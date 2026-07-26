import time
import numpy as np
import cv2
from typing import Optional
from pathlib import Path
import torch

from core.utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from core.utils.utilities import cxy_wh_2_rect

from .backend_manager import BackendManager


class DaSiamRPNTracker:
    """
    Stateless w.r.t. video I/O — caller feeds frames one at a time via track_step().
    track_live() is available for local webcam/file playback with cv2 display.
    One instance = one tracking session.

    Frames arrive as numpy BGR (from cv2/FastAPI upload) and are converted to a
    CUDA float32 tensor exactly once per call, at the top of init_from_mask/track_step.
    Everything downstream (run_SiamRPN.py) operates on that tensor with a single
    .cpu() sync per frame, inside tracker_eval.
    """

    def __init__(self,
                 model_path: str = 'models/SiamRPNOTB.model',
                 onnx_path:  str = 'weights/search.onnx',
                 trt_path:   str = 'weights/search.engine',
                 use_onnx:   bool = True,
                 backend_manager: BackendManager = None):
        self.backend = backend_manager or BackendManager(
            model_path=model_path,
            onnx_path=onnx_path,
            trt_path=trt_path,
            use_onnx=use_onnx,
            benchmark=True
        )
        self.device = self.backend.device
        self._pinned_buffer: Optional[torch.Tensor] = None


        self.state           = None
        self.last_good_state = None
        self.score_ema       = None
        self.alpha           = 0.7
        self.fps_ema         = None
        self.alpha_fps       = 0.9
        self.last_tracking_fps = 0.0

        self.CONF_THRESH = 0.35
        self.MAX_LOST    = 15
        self.lost_count  = 0

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def reset(self):
        self.state = None
        self.last_good_state = None
        self.score_ema = None
        self.fps_ema = None
        self.lost_count = 0

    def _frame_to_gpu(self, frame: np.ndarray) -> torch.Tensor:
        # Convert numpy array to a lightweight torch tensor view
        # (Note: from_numpy creates a view; we defer .float() to avoid a redundant copy)
        cpu_tensor = torch.from_numpy(frame)

        # Check if the target device is an accelerator (CUDA/MPS)
        if self.device.type != "cpu":
            # Lazily allocate or resize the pinned staging buffer
            if self._pinned_buffer is None or self._pinned_buffer.shape != cpu_tensor.shape:
                # Allocate page-locked (pinned) memory on the host matching the frame shape
                self._pinned_buffer = torch.empty(
                    cpu_tensor.shape,
                    dtype=torch.float32,
                    pin_memory=True
                )

            # Copy the numpy view data in-place into the pinned buffer and cast to float
            self._pinned_buffer.copy_(cpu_tensor)

            # Asynchronously transfer to the GPU
            return self._pinned_buffer.to(self.device, non_blocking=True)

        return cpu_tensor.float().to(self.device)

    @staticmethod
    def _clone_state(state: dict) -> dict:
        """
        dict.copy() is shallow — fine for scalars/config objects that are never
        mutated in place, but target_pos/target_sz get REASSIGNED (not
        mutated) every SiamRPN_track call, so aliasing the tensor reference here
        is safe. Being explicit about it rather than relying on that as an
        accident: we .clone() the two tensors that matter so last_good_state
        can never be silently affected by a future in-place edit to state.
        """
        new_state = state.copy()
        new_state['target_pos'] = state['target_pos'].clone()
        new_state['target_sz'] = state['target_sz'].clone()
        if "r1_kernel" in state:
            new_state["r1_kernel"] = state["r1_kernel"].clone()

        if "cls1_kernel" in state:
            new_state["cls1_kernel"] = state["cls1_kernel"].clone()
        return new_state

    # -----------------------------------------------------------
    # INIT FROM MASK
    # -----------------------------------------------------------
    def init_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> tuple:
        if frame is None or mask is None:
            raise ValueError("frame and mask are required")
        self.init_shape = frame.shape

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Mask is empty — nothing to track")

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w  = max(10, x_max - x_min)
        h  = max(10, y_max - y_min)
        cx = x_min + w / 2
        cy = y_min + h / 2

        target_pos = [cx, cy]   # plain python list — SiamRPN_init expects this now
        target_sz  = [w, h]

        im_t = self._frame_to_gpu(frame)
        self.state           = SiamRPN_init(im_t, target_pos, target_sz, self.backend.get_pt_net())
        self.last_good_state = self._clone_state(self.state)
        self.score_ema       = None
        self.lost_count      = 0
        self.backend.export_and_build(self.state["r1_kernel"], self.state["cls1_kernel"])

        print(f"[INFO] Tracker initialised | box: ({x_min},{y_min},{w},{h})")
        return (x_min, y_min, w, h)

    # -----------------------------------------------------------
    # TRACK STEP  (FastAPI / per-frame API)
    # -----------------------------------------------------------
    def track_step(self, frame: np.ndarray) -> dict:
        if self.state is None:
            raise RuntimeError("Call init_from_mask() before track_step()")

        t0 = time.perf_counter()
        active_net, backend = self.backend.active_net
        self.state["net"] = active_net

        im_t = self._frame_to_gpu(frame)

        # Runs inference and decodes anchors
        self.state = SiamRPN_track(self.state, im_t)

        # TARGETED STREAM SYNCHRONIZATION
        if self.device.type == "cuda":
            if backend == "TensorRT" and hasattr(active_net, 'stream'):
                # Force CPU host to block until ONLY the custom TRT stream finishes processing.
                # This guarantees that the buffers are filled and time.perf_counter() is completely accurate.
                active_net.stream.synchronize()
            else:
                # Secure fallback for default PyTorch stream and ONNX
                torch.cuda.current_stream().synchronize()

        # Host-side timer now accurately reflects combined (GPU compute + CPU preprocessing) time
        self.last_tracking_fps = 1.0 / (time.perf_counter() - t0)

        raw_score = float(self.state.get("score", 1.0))
        if np.isnan(raw_score):
            raw_score = 0.0

        self.score_ema = (
            raw_score
            if self.score_ema is None
            else self.alpha * self.score_ema + (1 - self.alpha) * raw_score
        )

        score = self.score_ema
        coords_nan = torch.isnan(self.state["target_pos"]).any() or torch.isnan(self.state["target_sz"]).any()
        weak = (score < self.CONF_THRESH) or coords_nan

        H, W = frame.shape[:2]
        if weak:
            self.lost_count += 1
            if self.last_good_state is not None:
                self.state = self._clone_state(self.last_good_state)
        else:
            self.lost_count = 0
            self.last_good_state = self._clone_state(self.state)

        # Mapping to integer is completely protected from NaN states
        x, y, w, h = map(
            int,
            cxy_wh_2_rect(self.state["target_pos"], self.state["target_sz"]).tolist(),
        )

        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))

        lost = self.lost_count >= self.MAX_LOST

        fps_inst = self.last_tracking_fps
        self.fps_ema = (
            fps_inst
            if self.fps_ema is None
            else self.alpha_fps * self.fps_ema + (1 - self.alpha_fps) * fps_inst
        )

        return {
            "bbox": (x, y, w, h),
            "score": float(score),
            "lost": lost,
            "tracker_fps": float(self.fps_ema),
            "model_fps": float(self.backend.model_fps),
            "backend": backend,
        }
        # -----------------------------------------------------------
        # TRACK LIVE
        # -----------------------------------------------------------

    def track_live(self, video_src="input.mp4", display: bool = False):
        if self.state is None:
            raise RuntimeError("Call init_from_mask() before track_live()")

        _, backend = self.backend.active_net
        print(f"[INFO] track_live started | backend: {backend}")

        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_src}")

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        output_path = results_dir / f"{backend}_backend_tracked.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        # ------------------------------------------------------

        tracker_fps_sum = 0.0
        model_fps_sum = 0.0
        frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.track_step(frame)

                tracker_fps_sum += result["tracker_fps"]
                model_fps_sum += result["model_fps"]
                frames += 1

                bbox = result["bbox"]
                score = result["score"]
                lost = result["lost"]

                if lost:
                    color = (0, 0, 255)
                elif score >= self.CONF_THRESH:
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)

                if not lost:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.putText(
                    frame,
                    (
                        f'{result["backend"]} | '
                        f'Tracker:{result["tracker_fps"]:.0f} | '
                        f'Model:{result["model_fps"]:.0f} | '
                        f'S:{score:.2f}'
                    ),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

                writer.write(frame)

                if display:
                    cv2.imshow("DaSiamRPN", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break

                yield None if lost else result

        finally:
            cap.release()
            writer.release()

            if display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

            print(f"[INFO] Saved tracked video to: {output_path}")

            if frames:
                print("=" * 50)
                print(f"Frames processed    : {frames}")
                print(f"Average Tracker FPS : {tracker_fps_sum / frames:.2f}")
                print(f"Average Model FPS   : {model_fps_sum / frames:.2f}")
                print("=" * 50)