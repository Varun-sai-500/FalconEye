import time
import numpy as np

from core.utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from core.utils.utilities import cxy_wh_2_rect

from .backend_manager import BackendManager


class DaSiamRPNTracker:
    """
    Stateless w.r.t. video I/O — caller feeds frames one at a time via track_step().
    track_live() is available for local webcam/file playback with cv2 display.
    One instance = one tracking session.

    All model loading, ONNX export, and TensorRT build/backend-selection logic
    lives in BackendManager — this class only tracks.
    """
    def __init__(self,
                 model_path: str = 'models/SiamRPNOTB.model',
                 onnx_path:  str = 'weights/search.onnx',
                 trt_path:   str = 'weights/search.engine',
                 use_onnx:   bool = True,
                 backend_manager: BackendManager = None):
        """
        Args:
            model_path, onnx_path, trt_path, use_onnx : forwarded to BackendManager
                if backend_manager isn't supplied directly.
            backend_manager : optionally inject an existing BackendManager
                (e.g. to share one across multiple tracker sessions).
        """
        self.backend = backend_manager or BackendManager(
            model_path=model_path,
            onnx_path=onnx_path,
            trt_path=trt_path,
            use_onnx=use_onnx,
        )

        # tracking state
        self.state           = None
        self.last_good_state = None
        self.score_ema       = None
        self.alpha           = 0.7
        self.fps_ema         = None
        self.alpha_fps       = 0.9
        self.last_tracking_fps = 0.0

        # thresholds (instance attrs so FastAPI callers can override per-session)
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

    # -----------------------------------------------------------
    # INIT FROM MASK
    # -----------------------------------------------------------
    def init_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> tuple:
        """
        Initialise tracker from a segmentation mask.

        Args:
            frame : HxWxC numpy BGR
            mask  : HxW binary (255 or True = object pixels)
        Returns:
            (x_min, y_min, w, h)
        """
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

        target_pos = np.array([cx, cy])
        target_sz  = np.array([w,  h])

        # SiamRPN_init calls pt_net.temple(real_z_crop) internally —
        # r1_kernel / cls1_kernel are REAL after this line
        self.state           = SiamRPN_init(frame, target_pos, target_sz, self.backend.get_pt_net())
        self.last_good_state = self.state.copy()
        self.score_ema       = None
        self.lost_count      = 0

        # export NOW — kernels are real at this exact point. No-op if already done.
        self.backend.export_and_build()

        print(f"[INFO] Tracker initialised | box: ({x_min},{y_min},{w},{h})")
        return (x_min, y_min, w, h)

    # -----------------------------------------------------------
    # TRACK STEP  (FastAPI / per-frame API)
    # -----------------------------------------------------------
    def track_step(self, frame: np.ndarray) -> dict:
        if self.state is None:
            raise RuntimeError("Call init_from_mask() before track_step()")

        active_net, backend = self.backend.active_net

        self.state["net"] = active_net
        t0 = time.perf_counter()
        self.state = SiamRPN_track(self.state, frame)
        self.last_tracking_fps = 1.0 / (time.perf_counter() - t0)

        raw_score = float(self.state.get("score", 1.0))
        self.score_ema = (
            raw_score
            if self.score_ema is None
            else self.alpha * self.score_ema + (1 - self.alpha) * raw_score
        )

        score = self.score_ema
        weak = score < self.CONF_THRESH

        H, W = frame.shape[:2]

        x, y, w, h = map(
            int,
            cxy_wh_2_rect(
                self.state["target_pos"],
                self.state["target_sz"],
            ),
        )

        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))

        if weak:
            self.lost_count += 1
            if self.last_good_state is not None:
                self.state = self.last_good_state.copy()
                x, y, w, h = map(
                    int,
                    cxy_wh_2_rect(
                        self.state["target_pos"],
                        self.state["target_sz"],
                    ),
                )
        else:
            self.lost_count = 0
            self.last_good_state = self.state.copy()

        lost = self.lost_count >= self.MAX_LOST

        fps_inst = self.last_tracking_fps

        self.fps_ema = (
            fps_inst
            if self.fps_ema is None
            else self.alpha_fps * self.fps_ema
            + (1 - self.alpha_fps) * fps_inst
        )

        return {
            "bbox": (x, y, w, h),
            "score": float(score),
            "lost": lost,
            "fps": float(self.fps_ema),
            "model_fps": float(getattr(active_net, "last_model_fps", 0.0)),
            "backend": backend,
        }

    # -----------------------------------------------------------
    # TRACK LIVE  (local display / debug)
    # -----------------------------------------------------------
    def track_live(self, video_src=0, display: bool = True):
        """
        Convenience generator for local webcam or file playback.
        Calls track_step() internally — no duplicated logic.
        Yields the track_step() dict each frame, or None when lost.

        Args:
            video_src : cv2.VideoCapture source (int or path)
            display   : draw bbox + HUD via cv2.imshow
        """
        import cv2

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
                result = self.track_step(frame)

                bbox    = result["bbox"]
                score   = result["score"]
                lost    = result["lost"]
                fps     = result["fps"]
                mfps    = result["model_fps"]
                label   = result["backend"]
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
                        f"{label} | E2E:{int(fps)} | Model:{int(mfps)} | S:{score:.2f}",
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