import os
import sys
import cv2
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.net import SiamRPNvot
from utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils.utilities import cxy_wh_2_rect

torch.set_grad_enabled(False)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[WARN] onnxruntime not installed — using PyTorch inference")


# ---------------------------------------------------------------
# ONNX NET WRAPPER
# ---------------------------------------------------------------
class _ONNXNet:
    """
    Drop-in for SiamRPNvot during track_live().
    temple() is a no-op — real kernels already baked into the graph.
    """
    def __init__(self, onnx_path):
        self.fps_ema = None
        self.alpha_fps = 0.9
        self.last_model_fps = 0.0
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if ORT_AVAILABLE and 'CUDAExecutionProvider' in ort.get_available_providers()
            else ['CPUExecutionProvider']
        )
        so = ort.SessionOptions()
        self.session    = ort.InferenceSession(onnx_path,sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[INFO] ONNX session loaded | provider: {self.session.get_providers()[0]}")

    def __call__(self, x_crop):
        x_np = x_crop if x_crop.dtype == np.float32 else x_crop.astype(np.float32)

        t0 = time.perf_counter()   # ✅ model timing start

        regression, classification = self.session.run(
            None, {self.input_name: x_np}
        )

        dt = time.perf_counter() - t0
        self.last_model_fps = 1.0 / dt   # ✅ store model FPS

        return regression, classification

    def temple(self, z):
        pass   # no-op — kernels already baked in

    cfg = {}


# ---------------------------------------------------------------
# MAIN TRACKER CLASS
# ---------------------------------------------------------------
class DaSiamRPNTracker:
    def __init__(self,
                 model_path='models/SiamRPNVOT.model',
                 onnx_path='search.onnx',
                 use_onnx=True):
        """
        Args:
            model_path : PyTorch .model weights
            onnx_path  : where to save/load search.onnx
            use_onnx   : False → pure PyTorch the whole way
        """
        self.model_path = model_path
        self.fps_ema = None
        self.alpha_fps = 0.9
        self.onnx_path  = onnx_path
        self.use_onnx   = use_onnx and ORT_AVAILABLE
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # always load PyTorch net — needed for temple() during init
        self.pt_net = SiamRPNvot()
        self.pt_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pt_net.eval().to(self.device)
        print(f"[INFO] PyTorch model loaded | device: {self.device}")

        # onnx_net stays None until init_from_mask() exports with real kernels
        self.onnx_net        = None
        self.state           = None
        self.last_good_state = None
        self.score_ema       = None
        self.alpha           = 0.7

    # -----------------------------------------------------------
    # INTERNAL — export search.onnx AFTER real temple() has run
    # -----------------------------------------------------------
    def _export_with_real_kernels(self):
        """
        Called inside init_from_mask() AFTER SiamRPN_init() has already
        called pt_net.temple(real_crop).
        At this point r1_kernel and cls1_kernel inside pt_net are REAL.
        Exporting now bakes those real kernels as constants into the graph.
        """
        print("[INFO] Exporting search.onnx with real template kernels ...")

        # only the search crop is the input — kernels are now constants
        dummy_x = torch.zeros(1, 3, 271, 271).to(self.device)

        with torch.no_grad():
            torch.onnx.export(
                self.pt_net,
                dummy_x,
                self.onnx_path,
                input_names=['search_crop'],
                output_names=['regression', 'classification'],
                opset_version=18,
                do_constant_folding=True,   # bakes REAL r1_kernel/cls1_kernel
            )

        print(f"[INFO] Exported → '{self.onnx_path}'")
        self.onnx_net = _ONNXNet(self.onnx_path)

    # -----------------------------------------------------------
    # INIT FROM MASK
    # -----------------------------------------------------------
    def init_from_mask(self, rgb_frame, mask):
        """
        Args:
            rgb_frame : HxWxC numpy BGR
            mask      : HxW binary (255 or True = object pixels)
        Returns:
            (x_min, y_min, w, h)
        """
        if rgb_frame is None or mask is None:
            raise ValueError("rgb_frame and mask are required")

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

        # SiamRPN_init internally calls pt_net.temple(real_z_crop)
        # after this line r1_kernel and cls1_kernel are REAL
        self.state           = SiamRPN_init(rgb_frame, target_pos, target_sz, self.pt_net)
        self.last_good_state = self.state.copy()
        self.score_ema       = None

        # export NOW — kernels are real at this exact point
        if self.use_onnx:
            self._export_with_real_kernels()

        print(f"[INFO] Tracker initialized | box: ({x_min},{y_min},{w},{h})")
        return (x_min, y_min, w, h)

    # -----------------------------------------------------------
    # LIVE TRACKING
    # -----------------------------------------------------------
    def track_live(self, video_src=0, display=True):
        """
        Yields (x, y, w, h) every frame.
        """
        if self.state is None:
            raise RuntimeError("Call init_from_mask() before track_live()")

        # use ONNX net if exported, else fall back to PyTorch
        active_net = self.onnx_net if (self.use_onnx and self.onnx_net) else self.pt_net
        mode_label = 'ONNX' if (self.use_onnx and self.onnx_net) else 'PyTorch'

        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_src}")

        SCREEN_W    = 512
        SCREEN_H    = 512
        CONF_THRESH = 0.35
        MAX_LOST    = 15
        lost_count  = 0

        print(f"[INFO] Tracking started | mode: {mode_label}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))

            # ✅ START TIMER (correct place)
            t0 = time.perf_counter()

            # ---------------- TRACKING ----------------
            self.state['net'] = active_net
            self.state        = SiamRPN_track(self.state, frame)

            raw_score = float(self.state.get('score', 1.0))
            if self.score_ema is None:
                self.score_ema = raw_score
            else:
                self.score_ema = self.alpha * self.score_ema + (1 - self.alpha) * raw_score

            score = self.score_ema
            weak  = score < CONF_THRESH

            x, y, w, h = map(
                int,
                cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
            )
            x = max(0, min(x, SCREEN_W - w))
            y = max(0, min(y, SCREEN_H - h))

            if weak:
                lost_count += 1
                print(f"[WARN] Weak | score={score:.2f} | lost={lost_count}/{MAX_LOST}")
                if self.last_good_state is not None:
                    self.state = self.last_good_state.copy()
                    x, y, w, h = map(
                        int,
                        cxy_wh_2_rect(
                            self.state['target_pos'],
                            self.state['target_sz']
                        )
                    )
            else:
                lost_count = 0
                self.last_good_state = self.state.copy()

            if lost_count >= MAX_LOST:
                print("[ERROR] Target LOST — holding last known bbox")
                lost_count = MAX_LOST
                yield (x, y, w, h)
                continue

            # ✅ END TIMER (correct place)
            dt = time.perf_counter() - t0

            # --- Instant FPS ---
            fps_inst = 1.0 / dt

            # --- Smoothed FPS ---
            if self.fps_ema is None:
                self.fps_ema = fps_inst
            else:
                self.fps_ema = self.alpha_fps * self.fps_ema + (1 - self.alpha_fps) * fps_inst

            # --- Model FPS ---
            model_fps = getattr(active_net, "last_model_fps", 0.0)

            # ---------------- DISPLAY ----------------
            if display:
                color = (0, 255, 0) if not weak else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{mode_label} | E2E:{int(self.fps_ema)} | Inst:{int(fps_inst)} | Model:{int(model_fps)} | S:{score:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )
                cv2.imshow("DaSiamRPN", frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                break

            yield (x, y, w, h)