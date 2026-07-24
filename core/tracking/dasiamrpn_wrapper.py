import time
import numpy as np
import cv2
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
        )
        self.device = self.backend.device

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
        cpu_tensor = torch.from_numpy(frame).float()
        pinned_tensor = cpu_tensor.pin_memory()
        """The ONE H2D copy per frame. Everything downstream reuses this tensor."""
        return pinned_tensor.to(self.device, non_blocking=True)

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

        self.state = SiamRPN_track(self.state, im_t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_tracking_fps = 1.0 / (time.perf_counter() - t0)
        raw_score = float(self.state.get("score", 1.0))

        # 1. Safely handle NaN scores
        if np.isnan(raw_score):
            raw_score = 0.0

        self.score_ema = (
            raw_score
            if self.score_ema is None
            else self.alpha * self.score_ema + (1 - self.alpha) * raw_score
        )

        score = self.score_ema

        # 2. Check if the output bounding boxes contain any NaN values
        coords_nan = torch.isnan(self.state["target_pos"]).any() or torch.isnan(self.state["target_sz"]).any()
        weak = (score < self.CONF_THRESH) or coords_nan

        H, W = frame.shape[:2]

        # 3. INTERCEPT AND ROLLBACK BEFORE CONVERTING TO INT
        if weak:
            self.lost_count += 1
            if self.last_good_state is not None:
                self.state = self._clone_state(self.last_good_state)
        else:
            self.lost_count = 0
            self.last_good_state = self._clone_state(self.state)

        # 4. Now mapping to integer is safe because NaN states have been reverted
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
            "model_fps": float(getattr(active_net, "last_model_fps", 0.0)),
            "backend": backend,
        }

    # -----------------------------------------------------------
    # TRACK LIVE  (local display / debug)
    # -----------------------------------------------------------

    def track_live(self, video_src=0, display: bool = True):
        if self.state is None:
            raise RuntimeError("Call init_from_mask() before track_live()")

        _, backend = self.backend.active_net
        print(f"[INFO] track_live started | backend: {backend}")

        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_src}")

        SCREEN_W = 512
        SCREEN_H = 512

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))
                result = self.track_step(frame)  # numpy in, GPU tensor conversion happens inside

                bbox         = result["bbox"]
                score        = result["score"]
                lost         = result["lost"]
                tracker_fps  = result["tracker_fps"]
                model_fps    = result["model_fps"]
                label         = result["backend"]
                x, y, w, h = bbox

                if display:
                    if not lost:
                        color = (0, 255, 0) if score >= self.CONF_THRESH else (0, 165, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    hud_color = (0, 0, 255) if lost else (
                        (0, 255, 0) if score >= self.CONF_THRESH else (0, 165, 255)
                    )
                    cv2.putText(
                        frame,
                        f"{label} | Tracker:{int(tracker_fps)} | Model:{int(model_fps)} | S:{score:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        hud_color,
                        2,
                    )
                    cv2.imshow("DaSiamRPN", frame)

                    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                        break

                yield None if lost else result
        finally:
            cap.release()
            cv2.destroyAllWindows()