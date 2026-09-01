import time
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

import torch

from core.tracking.pipeline import DaSiamRPN_init, DaSiamRPN_track
from .backend_manager import BackendManager


class DaSiamRPNTracker:
    """
    Stateless w.r.t. video I/O — caller feeds frames one at a time via tracking().
    track_offline() is available for headless batch processing with metrics export.
    One instance = one tracking session.

    Frame pipeline:
        numpy BGR (uint8) → pinned uint8 HtoD (non_blocking) → cast to dtype on GPU
    Everything downstream (pipeline.py) operates on GPU tensors.
    One .cpu() sync per frame inside tracker_eval.
    """

    def __init__(
        self,
        model_path: str = 'models/SiamRPNOTB.model',
        onnx_path:  str = 'weights/search.onnx',
        trt_path:   str = 'weights/search.engine',
        use_onnx:   bool = True,
        backend_manager: BackendManager = None,
    ):
        self.backend = backend_manager or BackendManager(
            model_path=model_path,
            onnx_path=onnx_path,
            trt_path=trt_path,
            use_onnx=use_onnx,
            benchmark=True,
        )
        self.device = self.backend.device
        self.dtype  = self.backend.dtype

        self._active_pt_net              = None
        self._pinned_u8: Optional[torch.Tensor] = None   # uint8 pinned staging buffer
        self._frame_gpu: Optional[torch.Tensor] = None

        self.state           = None
        self.last_good_state = None
        self.score_ema       = None
        self.alpha           = 0.7
        self.fps_ema         = None
        self.alpha_fps       = 0.9

        self.CONF_THRESH = 0.35
        self.MAX_LOST    = 15
        self.lost_count  = 0

        # Kinematics telemetry
        self._prev_center        = None
        self._was_lost           = False
        self._current_lost_run   = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def reset(self):
        self.state           = None
        self._active_pt_net  = None
        self.last_good_state = None
        self.score_ema       = None
        self.fps_ema         = None
        self.lost_count      = 0
        self._prev_center    = None
        self._was_lost       = False
        self._current_lost_run = 0

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _center_to_rect(center, size) -> torch.Tensor:
        return torch.stack((
            center[0] - size[0] * 0.5,
            center[1] - size[1] * 0.5,
            size[0],
            size[1],
        ))

    def _frame_to_gpu(self, frame: np.ndarray) -> torch.Tensor:
        """
        numpy BGR uint8 → persistent GPU tensor in self.dtype.

        CUDA:
            numpy → pinned uint8 staging buffer
                → persistent GPU buffer via async HtoD + dtype conversion // bottleneck

        CPU/MPS:
            numpy → tensor → device/dtype conversion directly.
        """
        if self.device.type == "cuda":

            # --------------------------------------------------------------
            # Persistent pinned CPU staging buffer
            # --------------------------------------------------------------
            if (
                self._pinned_u8 is None
                or self._pinned_u8.shape != torch.Size(frame.shape)
            ):
                self._pinned_u8 = torch.empty(
                    frame.shape,
                    dtype=torch.uint8,
                    pin_memory=True,
                )

            # numpy → pinned CPU buffer
            self._pinned_u8.copy_(torch.from_numpy(frame))

            H, W, C = frame.shape

            # --------------------------------------------------------------
            # Persistent GPU frame buffer
            # --------------------------------------------------------------
            if (
                self._frame_gpu is None
                or self._frame_gpu.shape != torch.Size((H, W, C))
                or self._frame_gpu.dtype != self.dtype
                or self._frame_gpu.device != self.device
            ):
                self._frame_gpu = torch.empty(
                    (H, W, C),
                    device=self.device,
                    dtype=self.dtype,
                )

            # --------------------------------------------------------------
            # H2D + dtype conversion on BackendManager-owned stream
            # --------------------------------------------------------------
            with self.backend.stream_context():

                # Copy uint8 → GPU dtype directly into persistent buffer.
                self._frame_gpu.copy_(
                    self._pinned_u8,
                    non_blocking=True,
                )

            return self._frame_gpu

        # ------------------------------------------------------------------
        # CPU / MPS
        # ------------------------------------------------------------------
        return torch.from_numpy(frame).to(
            device=self.device,
            dtype=self.dtype,
        )
    
    @staticmethod
    def _clone_state(state: dict) -> dict:
        """Shallow-copy state dict, cloning only the tensor leaves."""
        new_state = state.copy()
        new_state['target_pos'] = state['target_pos'].clone()
        new_state['target_sz']  = state['target_sz'].clone()
        if 'r1_kernel'  in state:
            new_state['r1_kernel']  = state['r1_kernel'].clone()
        if 'cls1_kernel' in state:
            new_state['cls1_kernel'] = state['cls1_kernel'].clone()
        return new_state

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

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
            raise ValueError(f"Invalid bbox dimensions: ({w}, {h})")

        cx, cy = x + w / 2.0, y + h / 2.0

        im_t = self._frame_to_gpu(frame)
        net  = self.backend.get_pt_net()

        if isinstance(net, torch.nn.Module) and self._active_pt_net is None:
            self._active_pt_net = net.to(device=self.device, dtype=self.dtype).eval()

        self.state = DaSiamRPN_init(
            im_t,
            [cx, cy],
            [float(w), float(h)],
            self._active_pt_net,
        )
        self.last_good_state = self._clone_state(self.state)
        self.score_ema  = None
        self.lost_count = 0

        self.backend.export_and_build(
            self.state['r1_kernel'],
            self.state['cls1_kernel'],
        )
        print(f"[INFO] Tracker initialised | box: ({x}, {y}, {w}, {h})")
        return (x, y, w, h)

    # ------------------------------------------------------------------
    # Per-frame tracking
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def tracking(self, frame: np.ndarray) -> dict:
        if self.state is None:
            raise RuntimeError("Call init_from_bbox() before tracking()")

        t0 = time.perf_counter()

        active_net, backend = self.backend.active_net
        if backend == "PyTorch":
            active_net = self._active_pt_net
        self.state['net'] = active_net

        im_t = self._frame_to_gpu(frame)

        self.state = self.backend.track_step(
            self.state,
            im_t,
        )
        self.backend.synchronize()

        tracking_latency_ms = (time.perf_counter() - t0) * 1000.0

        # ── Score EMA ─────────────────────────────────────────────────
        raw_score = float(self.state.get('score', 1.0))
        if np.isnan(raw_score):
            raw_score = 0.0

        self.score_ema = (
            raw_score
            if self.score_ema is None
            else self.alpha * self.score_ema + (1.0 - self.alpha) * raw_score
        )
        score = self.score_ema

        # ── Lost detection ────────────────────────────────────────────
        coords_nan = (
            torch.isnan(self.state['target_pos']).any()
            or torch.isnan(self.state['target_sz']).any()
        )
        weak = (score < self.CONF_THRESH) or coords_nan

        if weak:
            self.lost_count += 1
            if self.last_good_state is not None:
                self.state = self._clone_state(self.last_good_state)
        else:
            self.lost_count = 0
            # Only snapshot last_good_state when the tracker is healthy.
            self.last_good_state = self._clone_state(self.state)

        H, W = frame.shape[:2]
        x, y, w, h = map(
            int,
            self._center_to_rect(
                self.state['target_pos'], self.state['target_sz']
            ).tolist(),
        )
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))

        lost = self.lost_count >= self.MAX_LOST

        # ── FPS EMA ───────────────────────────────────────────────────
        fps_inst = 1000.0 / tracking_latency_ms
        self.fps_ema = (
            fps_inst
            if self.fps_ema is None
            else self.alpha_fps * self.fps_ema + (1.0 - self.alpha_fps) * fps_inst
        )

        # ── Kinematics telemetry (cheap scalar, no sqrt on lost frames) ──
        center_x, center_y = x + w / 2.0, y + h / 2.0
        jerk_delta          = 0.0

        if not lost:
            if self._prev_center is not None:
                dx = center_x - self._prev_center[0]
                dy = center_y - self._prev_center[1]
                jerk_delta = float(np.sqrt(dx * dx + dy * dy))
            self._prev_center = (center_x, center_y)
        else:
            self._prev_center = None

        is_recovery_frame = False
        recovery_duration = 0
        if not lost:
            if self._was_lost:
                is_recovery_frame  = True
                recovery_duration  = self._current_lost_run
                self._current_lost_run = 0
                self._was_lost     = False
        else:
            self._current_lost_run += 1
            self._was_lost = True

        return {
            'bbox':        (x, y, w, h),
            'score':       score,
            'lost':        lost,
            'tracker_fps': float(self.fps_ema),
            'backend':     backend,
            'metrics': {
                'tracking_latency_ms': tracking_latency_ms,
                'jerk_delta':          jerk_delta,
                'is_recovery':         is_recovery_frame,
                'recovery_duration':   recovery_duration,
                'raw_score':           raw_score,
            },
        }

    # ------------------------------------------------------------------
    # Batch offline processing
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def track_offline(
        self,
        video_src: str,
        bbox: tuple[int, int, int, int]
    ) -> dict:
        """
        Processes a full video sequence sequentially.
        Initialises from the first frame, tracks all subsequent frames,
        writes an annotated output video to results/, and returns metrics.
        """
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_src}")

        Path("results").mkdir(parents=True, exist_ok=True)
        out_filepath = str(Path("results") / "tracked_results.mp4")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            out_filepath,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )

        # ── Telemetry accumulators ────────────────────────────────────
        latencies_ms:       list[float] = []
        scores:             list[float] = []
        lost_flags:         list[bool]  = []
        bbox_history:       list        = []
        tracklet_lengths:   list[int]   = []
        recovery_latencies: list[int]   = []

        current_tracklet_len = 0
        current_lost_len     = 0
        was_lost             = False
        frames               = 0

        try:
            # ── Frame 0: init ─────────────────────────────────────────
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                raise RuntimeError("Failed to read the initial frame.")

            self.init_from_bbox(first_frame, bbox)
            _, backend = self.backend.active_net
            print(f"[INFO] Offline tracking started | Backend: {backend}")

            frames += 1
            scores.append(1.0)
            lost_flags.append(False)
            bbox_history.append(bbox)
            current_tracklet_len += 1

            # Write init frame with bbox drawn
            ix, iy, iw, ih = map(int, bbox)
            draw = first_frame.copy()
            cv2.rectangle(draw, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)
            writer.write(draw)

            # ── Tracking loop ─────────────────────────────────────────
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.tracking(frame)

                latencies_ms.append(result['metrics']['tracking_latency_ms'])
                frames += 1

                res_bbox = result['bbox']
                score    = result['score']
                lost     = result['lost']

                scores.append(score)
                lost_flags.append(lost)
                bbox_history.append(res_bbox if not lost else None)

                if not lost:
                    draw = frame.copy()
                    x, y, w, h = res_bbox
                    cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    writer.write(draw)
                else:
                    writer.write(frame)

                # ── Tracklet diagnostics ──────────────────────────────
                if not lost:
                    current_tracklet_len += 1
                    if was_lost:
                        recovery_latencies.append(current_lost_len)
                        current_lost_len = 0
                        was_lost         = False
                else:
                    if current_tracklet_len > 0:
                        tracklet_lengths.append(current_tracklet_len)
                        current_tracklet_len = 0
                    current_lost_len += 1
                    was_lost          = True

        finally:
            cap.release()
            writer.release()
            print(f"[INFO] Tracked video saved → {out_filepath}")

        if current_tracklet_len > 0:
            tracklet_lengths.append(current_tracklet_len)

        # ── Analytics ─────────────────────────────────────────────────
        lat    = np.asarray(latencies_ms, dtype=np.float64)
        lflags = np.array(lost_flags)

        lost_count   = int(np.sum(lflags))
        failure_rate = (lost_count / frames) * 100.0

        mean_latency   = float(np.mean(lat))
        median_latency = float(np.median(lat))
        p95            = float(np.percentile(lat, 95))
        p99            = float(np.percentile(lat, 99))
        jitter         = float(np.std(lat))
        avg_tracking_fps = 1000.0 / mean_latency

        # Jerk computed post-hoc from bbox_history
        jerk_deltas = []
        prev_center = None
        for b in bbox_history:
            if b is not None:
                cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
                if prev_center is not None:
                    dx = cx - prev_center[0]
                    dy = cy - prev_center[1]
                    jerk_deltas.append(np.sqrt(dx * dx + dy * dy))
                prev_center = (cx, cy)
            else:
                prev_center = None

        avg_jerk            = float(np.mean(jerk_deltas)) if jerk_deltas else 0.0
        avg_tracklet        = float(np.mean(tracklet_lengths)) if tracklet_lengths else 0.0
        max_tracklet        = int(np.max(tracklet_lengths))    if tracklet_lengths else 0
        avg_recovery_frames = float(np.mean(recovery_latencies)) if recovery_latencies else 0.0

        # ── Console report ────────────────────────────────────────────
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
        print(f"  Jitter                 : {jitter:.2f} ms")

        print("-" * 60)
        print("TRACKING QUALITY")
        print(f"  Failure Rate           : {failure_rate:.2f}% ")
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