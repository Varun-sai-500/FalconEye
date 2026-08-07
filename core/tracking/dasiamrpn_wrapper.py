import time
import numpy as np
import cv2
from typing import Optional
from pathlib import Path
import torch

from core.tracking.pipeline import DaSiamRPN_init, DaSiamRPN_track
from .backend_manager import BackendManager


class DaSiamRPNTracker:
    """
    Stateless w.r.t. video I/O — caller feeds frames one at a time via tracking().
    track_live() is available for local webcam/file playback with cv2 display.
    One instance = one tracking session.

    Frames arrive as numpy BGR (from cv2/FastAPI upload) and are converted to a
    CUDA/MPS half/bfloat16 tensor exactly once per call, at the top of init_from_bbox/tracking.
    Everything downstream (pipeline.py) operates on that tensor with a single
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
            benchmark=False
        )
        self.device = self.backend.device
        self._active_pt_net = None
        self._pinned_buffer: Optional[torch.Tensor] = None
        self.dtype = self.backend.dtype

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

        # Keep track of center for kinematic noise analytics
        self._prev_bench_center = None
        self._was_lost_state = False
        self._current_lost_run = 0

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def reset(self):
        self.state = None
        self._active_pt_net = None
        self.last_good_state = None
        self.score_ema = None
        self.fps_ema = None
        self.lost_count = 0
        self._prev_bench_center = None
        self._was_lost_state = False
        self._current_lost_run = 0

    @staticmethod
    def _center_to_rect(center, size):
        return torch.stack((
            center[0] - size[0] * 0.5,
            center[1] - size[1] * 0.5,
            size[0],
            size[1],
        ))

    def _frame_to_gpu(self, frame: np.ndarray) -> torch.Tensor:
        cpu_tensor = torch.from_numpy(frame)

        if self.device.type != "cpu":
            # Ensure pinned buffer tracks both size changes and dtype configurations (CUDA optimized)
            if self.device.type == "cuda":
                if (self._pinned_buffer is None
                        or self._pinned_buffer.shape != cpu_tensor.shape
                        or self._pinned_buffer.dtype != self.dtype):
                    self._pinned_buffer = torch.empty(
                        cpu_tensor.shape,
                        dtype=self.dtype,
                        pin_memory=True
                    )

                # Cast type on host using pinned memory to optimize HtoD bandwidth execution
                self._pinned_buffer.copy_(cpu_tensor.to(self.dtype))
                return self._pinned_buffer.to(self.device, non_blocking=True)

            # Direct transfer strategy for hardware architectures without memory pinning (MPS)
            return cpu_tensor.to(device=self.device, dtype=self.dtype)

        return cpu_tensor.to(dtype=self.dtype, device=self.device)

    @staticmethod
    def _clone_state(state: dict) -> dict:
        new_state = state.copy()
        new_state['target_pos'] = state['target_pos'].clone()
        new_state['target_sz'] = state['target_sz'].clone()
        if "r1_kernel" in state:
            new_state["r1_kernel"] = state["r1_kernel"].clone()
        if "cls1_kernel" in state:
            new_state["cls1_kernel"] = state["cls1_kernel"].clone()
        return new_state

    # -----------------------------------------------------------
    # INIT FROM BBOX
    # -----------------------------------------------------------
    def init_from_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:

        if frame is None:
            raise ValueError("frame is required")
        if bbox is None:
            raise ValueError("bbox is required")

        x, y, w, h = map(int, bbox)

        if w <= 0 or h <= 0:
            raise ValueError("Invalid bbox dimensions")

        cx = x + w / 2
        cy = y + h / 2

        target_pos = [cx, cy]
        target_sz = [w, h]
        im_t = self._frame_to_gpu(frame)

        # CRITICAL FIX: Ensure the PyTorch template extraction network matches the current session precision parameters
        net = self.backend.get_pt_net()

        if isinstance(net, torch.nn.Module):
            if self._active_pt_net is None:
                self._active_pt_net = net.to(
                    device=self.device,
                    dtype=self.dtype,
                ).eval()

        net = self._active_pt_net

        self.state = DaSiamRPN_init(
            im_t,
            target_pos,
            target_sz,
            net,
        )
        self.last_good_state = self._clone_state(self.state)
        self.score_ema = None
        self.lost_count = 0

        self.backend.export_and_build(
            self.state["r1_kernel"],
            self.state["cls1_kernel"],
        )
        print(f"[INFO] Tracker initialised | box: ({x}, {y}, {w}, {h})")
        return (x, y, w, h)

    # -----------------------------------------------------------
    # TRACK STEP (Per-frame API pipeline)
    # -----------------------------------------------------------
    @torch.inference_mode()
    def tracking(self, frame: np.ndarray) -> dict:
        if self.state is None:
            raise RuntimeError("Call init_from_bbox() before tracking()")

        t0 = time.perf_counter()

        # CRITICAL FIX: If running tracking natively through PyTorch, cast network dynamically
        active_net, backend = self.backend.active_net

        if backend == "PyTorch":
            active_net = self._active_pt_net

        self.state["net"] = active_net
        im_t = self._frame_to_gpu(frame)

        # Execute forward pass within the targeted precision context
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=autocast_enabled):
            self.state = DaSiamRPN_track(self.state, im_t)

        # TARGETED STREAM SYNCHRONIZATION
        if self.device.type == "cuda":
            if backend == "TensorRT" and hasattr(active_net, 'stream'):
                active_net.stream.synchronize()
            else:
                torch.cuda.current_stream().synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        tracking_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Instantaneous FPS (UI only)
        self.last_tracking_fps = 1000.0 / tracking_latency_ms

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

        x, y, w, h = map(
            int,
            self._center_to_rect(self.state["target_pos"], self.state["target_sz"]).tolist(),
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

        # --- Kinematics & Stability Telemetry ---
        center_x, center_y = x + w / 2.0, y + h / 2.0
        jerk_delta = 0.0

        if not lost:
            if self._prev_bench_center is not None:
                jerk_delta = float(np.sqrt((center_x - self._prev_bench_center[0])**2 + (center_y - self._prev_bench_center[1])**2))
            self._prev_bench_center = (center_x, center_y)
        else:
            self._prev_bench_center = None

        is_recovery_frame = False
        recovery_duration = 0

        if not lost:
            if self._was_lost_state:
                is_recovery_frame = True
                recovery_duration = self._current_lost_run
                self._current_lost_run = 0
                self._was_lost_state = False
        else:
            self._current_lost_run += 1
            self._was_lost_state = True

        return {
            "bbox": (x, y, w, h),
            "score": float(score),
            "lost": lost,
            "tracker_fps": float(self.fps_ema),
            "backend": backend,
            "metrics": {
                "tracking_latency_ms": tracking_latency_ms,
                "jerk_delta": jerk_delta,
                "is_recovery": is_recovery_frame,
                "recovery_duration": recovery_duration,
                "raw_score": raw_score
            }
        }

    # -----------------------------------------------------------
    # BATCH OFFLINE PROCESSING & PROFILING
    # -----------------------------------------------------------
    @torch.inference_mode()
    def track_offline(self, video_src: str, bbox: tuple[int, int, int, int], display: bool = False):
        """
        Processes a full video sequence sequentially.
        Extracts the first frame to auto-initialize tracking using the provided bbox parameters.
        Saves the resulting tracked video to the 'results' directory with isolated latency calculations.
        """
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_src}")

        # --- Setup Video Writer Target ---
        Path("results").mkdir(parents=True, exist_ok=True)

        out_filename = f"tracked_results.mp4"
        out_filepath = str(Path("results") / out_filename)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0  # Safe fallback if metadata is missing

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_filepath, fourcc, fps, (width, height))

        # --- Telemetry Profiles ---
        latencies_ms = []
        scores = []
        lost_flags = []
        bbox_history = []
        tracklet_lengths = []
        recovery_latencies = []

        current_tracklet_len = 0
        current_lost_len = 0
        was_lost = False
        frames = 0

        try:
            # Step 1: Handle first frame initialization natively
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                raise RuntimeError("Failed to read the initial frame from the video source.")

            # Anchor initial states (Mutates template but skips model inference)
            self.init_from_bbox(first_frame, bbox)

            _, backend = self.backend.active_net
            print(f"[INFO] Headless benchmark started | Backend: {backend}")

            # Log initial baseline metric frame
            frames += 1
            scores.append(1.0)
            lost_flags.append(False)
            bbox_history.append(bbox)
            current_tracklet_len += 1

            # Draw & write the initialization frame (Outside loop timing constraints)
            draw_frame = first_frame.copy()
            ix, iy, iw, ih = map(int, bbox)
            cv2.rectangle(draw_frame, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)
            writer.write(draw_frame)

            # Step 2: Continuous batch inference loop for remaining frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # === HIGH-PRECISION RUNTIME TIMING ZONE ===
                t_start = time.perf_counter()
                result = self.tracking(frame)
                dt = (time.perf_counter() - t_start) * 1000.0
                # ==========================================

                # Immediately catch time metrics before doing any file I/O operations
                latencies_ms.append(dt)
                frames += 1

                res_bbox = result["bbox"]
                score = result["score"]
                lost = result["lost"]

                scores.append(score)
                lost_flags.append(lost)
                bbox_history.append(res_bbox if not lost else None)

                # --- Defer Video Output Execution Tasks (Outside timing zone) ---
                draw_frame = frame.copy()
                if not lost:
                    x, y, w, h = res_bbox
                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                writer.write(draw_frame)

                # Optional debug rendering loop
                if display:
                    cv2.imshow("Tracking Profile Loop", draw_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # --- State Tracklet Diagnostics ---
                if not lost:
                    current_tracklet_len += 1
                    if was_lost:
                        recovery_latencies.append(current_lost_len)
                        current_lost_len = 0
                        was_lost = False
                else:
                    if current_tracklet_len > 0:
                        tracklet_lengths.append(current_tracklet_len)
                        current_tracklet_len = 0
                    current_lost_len += 1
                    was_lost = True
        finally:
            cap.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"[INFO] Tracked video saved to: {out_filepath}")

        if current_tracklet_len > 0:
            tracklet_lengths.append(current_tracklet_len)

        # --- Compute Analytics ---
        if frames > 1:
            latencies_ms = np.asarray(latencies_ms, dtype=np.float64)
            lost_flags = np.array(lost_flags)
            lost_count = np.sum(lost_flags)
            failure_rate = (lost_count / frames) * 100.0

            jerk_deltas = []
            prev_center = None
            for b in bbox_history:
                if b is not None:
                    cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                    if prev_center is not None:
                        jerk_deltas.append(np.sqrt((cx - prev_center[0])**2 + (cy - prev_center[1])**2))
                    prev_center = (cx, cy)
                else:
                    prev_center = None

            avg_jerk = np.mean(jerk_deltas) if jerk_deltas else 0.0
            mean_latency = float(np.mean(latencies_ms))
            median_latency = float(np.median(latencies_ms))
            p95 = float(np.percentile(latencies_ms, 95))
            p99 = float(np.percentile(latencies_ms, 99))
            jitter = float(np.std(latencies_ms))
            avg_tracking_fps = 1000.0 / mean_latency


            avg_tracklet = np.mean(tracklet_lengths) if tracklet_lengths else 0.0
            max_tracklet = np.max(tracklet_lengths) if tracklet_lengths else 0
            avg_recovery_frames = np.mean(recovery_latencies) if recovery_latencies else 0.0

        print("\n" + "=" * 60)
        print(f"ROBOTICS TRACKING PROFILING | {backend.upper()}")
        print("=" * 60)

        print("ENVIRONMENT")
        print(f"  Backend                : {backend}")
        print(f"  Device                 : {self.backend.device_name}")
        print(f"  Precision              : {str(self.backend.dtype).replace('torch.', '')}")
        print(f"  Frames                 : {frames}")

        print("-" * 60)
        print("PERFORMANCE")
        print(f"  Effective FPS          : {avg_tracking_fps:.2f}")
        print(f"  Mean Latency           : {mean_latency:.2f} ms")
        print(f"  Median Latency         : {median_latency:.2f} ms")
        print(f"  P95 Latency            : {p95:.2f} ms")
        print(f"  P99 Latency            : {p99:.2f} ms")
        print(f"  Jitter                : {jitter:.2f} ms")

        print("-" * 60)
        print("TRACKING QUALITY")
        print(f"  Failure Rate           : {failure_rate:.2f}% ({lost_count}/{frames})")
        print(f"  Mean Confidence        : {np.mean(scores):.3f}")
        print(f"  Mean Tracklet          : {avg_tracklet:.1f} frames")
        print(f"  Max Tracklet           : {max_tracklet} frames")
        print(f"  Mean Recovery          : {avg_recovery_frames:.1f} frames")
        print(f"  Mean Motion Jitter     : {avg_jerk:.2f} px/frame")

        print("=" * 60 + "\n")

        return {
            "backend": backend,
            "device": self.backend.device_name,
            "precision": str(self.backend.dtype).replace("torch.", ""),

            "performance": {
                "fps": avg_tracking_fps,
                "latency": {
                    "mean_ms": mean_latency,
                    "p50_ms": median_latency,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "jitter_ms": jitter,
                },
            },

            "tracking": {
                "failure_rate": failure_rate,
                "avg_tracklet": avg_tracklet,
                "max_tracklet": int(max_tracklet),
                "avg_recovery_frames": avg_recovery_frames,
                "avg_score": float(np.mean(scores)),
                "avg_jerk_px": avg_jerk,
            },

            "output_video": out_filepath,
        }
