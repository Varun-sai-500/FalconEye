import os
import time
import numpy as np
import torch

from core.utils.net import SiamRPNotb
from core.utils.run_SiamRPN import SiamRPN_init, SiamRPN_track
from core.utils.utilities import cxy_wh_2_rect

torch.set_grad_enabled(False)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[WARN] onnxruntime not installed — using PyTorch inference")

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_AVAILABLE = False
    print("[WARN] tensorrt not installed — TRT backend disabled")


# ---------------------------------------------------------------
# TRT HELPERS
# ---------------------------------------------------------------
def load_engine(engine_path):
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT not installed")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return runtime.deserialize_cuda_engine(engine_data)


def trt_infer(context, x_crop):
    if isinstance(x_crop, np.ndarray):
        x_crop = torch.from_numpy(x_crop)

    x_crop = x_crop.contiguous().cuda().float()

    regression = torch.empty((1, 20, 19, 19), device="cuda", dtype=torch.float32)
    classification = torch.empty((1, 10, 19, 19), device="cuda", dtype=torch.float32)

    context.set_tensor_address("search_crop",    x_crop.data_ptr())
    context.set_tensor_address("regression",     regression.data_ptr())
    context.set_tensor_address("classification", classification.data_ptr())

    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    return regression.cpu().numpy(), classification.cpu().numpy()


# ---------------------------------------------------------------
# ONNX NET WRAPPER
# ---------------------------------------------------------------
class _ONNXNet:
    """
    temple() is a no-op — real kernels already baked into the graph.
    """
    def __init__(self, onnx_path):
        self.last_model_fps = 0.0
        providers = ( ['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if ORT_AVAILABLE and 'CUDAExecutionProvider' in ort.get_available_providers()
                     else ['CPUExecutionProvider'] )
        so = ort.SessionOptions()
        self.session    = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[INFO] ONNX session loaded | provider: {self.session.get_providers()[0]}")

    def __call__(self, x_crop):
        x_np = x_crop if x_crop.dtype == np.float32 else x_crop.astype(np.float32)
        t0 = time.perf_counter()
        regression, classification = self.session.run(
            None,
            {self.input_name: x_np},
        )
        self.last_model_fps = 1.0 / (time.perf_counter() - t0)
        return regression, classification

    def temple(self, z):
        pass   # no-op — kernels already baked in

    cfg = {}


# ---------------------------------------------------------------
# TRT NET WRAPPER
# ---------------------------------------------------------------
class _TRTNet:
    """
    temple() is a no-op — kernels baked into the engine at build time.
    """
    def __init__(self, engine_path):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not installed — cannot load engine")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — cannot run TRT engine")

        self.engine  = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.last_model_fps = 0.0
        print(f"[INFO] TRT engine loaded from '{engine_path}'")

    def __call__(self, x_crop):
        t0 = time.perf_counter()
        regression, classification = trt_infer(self.context, x_crop)
        self.last_model_fps = 1.0 / (time.perf_counter() - t0)
        return regression, classification

    def temple(self, z):
        pass   # no-op — kernels baked into engine

    cfg = {}


# ---------------------------------------------------------------
# MAIN TRACKER CLASS
# ---------------------------------------------------------------
class DaSiamRPNTracker:
    """
    Stateless w.r.t. video I/O — caller feeds frames one at a time via track_step().
    track_live() is available for local webcam/file playback with cv2 display.
    One instance = one tracking session.
    Backend priority: TensorRT → ONNX → PyTorch.
    """
    def __init__(self,
                 model_path: str = 'models/SiamRPNOTB.model',
                 onnx_path:  str = 'weights/search.onnx',
                 trt_path:   str = 'weights/search.engine',
                 use_onnx:   bool = True):
        """
        Args:
            model_path : PyTorch .model weights (always loaded for temple() at init)
            onnx_path  : path to save/load search.onnx
            trt_path   : path to pre-built TensorRT engine
            use_onnx   : False → pure PyTorch the whole way
        """
        self.model_path = model_path
        self.onnx_path  = onnx_path
        self.trt_path   = trt_path
        self.use_onnx   = use_onnx and ORT_AVAILABLE
        self.use_trt    = (
            TRT_AVAILABLE
            and torch.cuda.is_available()
        )
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        # PyTorch net always loaded — needed for temple() during init_from_mask()
        self.pt_net = SiamRPNotb()
        self.pt_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pt_net.eval().to(self.device)
        print(f"[INFO] PyTorch model loaded | device: {self.device}")

        self.onnx_net = None
        self.trt_net  = None

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

    # -----------------------------------------------------------
    # ACTIVE BACKEND RESOLVER
    # -----------------------------------------------------------
    @property
    def initialized(self) -> bool:
        return self.state is not None
    def reset(self):
        self.state = None
        self.last_good_state = None
        self.score_ema = None
        self.fps_ema = None
        self.lost_count = 0

    @property
    def _active_net(self):
        """Returns the best available net. Priority: TRT → ONNX → PyTorch."""
        if self.trt_net is not None:
            return self.trt_net, "TensorRT"
        if self.onnx_net is not None:
            return self.onnx_net, "ONNX"
        return self.pt_net, "PyTorch"

    # -----------------------------------------------------------
    # INTERNAL — export search.onnx AFTER real temple() has run
    # -----------------------------------------------------------
    def _export_with_real_kernels(self):
        """
        Called inside init_from_mask() AFTER SiamRPN_init() has already
        called pt_net.temple(real_z_crop).
        r1_kernel and cls1_kernel are REAL at this point — do_constant_folding
        bakes them as frozen constants into the exported graph.
        """
        print("[INFO] Exporting search.onnx with real template kernels ...")
        dummy_x = torch.zeros(1, 3, 271, 271).to(self.device)
        with torch.no_grad():
            torch.onnx.export(
                self.pt_net,
                dummy_x,
                self.onnx_path,
                input_names=['search_crop'],
                output_names=['regression', 'classification'],
                opset_version=18,
                do_constant_folding=True,
            )
        print(f"[INFO] Exported → '{self.onnx_path}'")
        if TRT_AVAILABLE and not os.path.exists(self.trt_path):
            try:
                self._build_trt_engine()
            except Exception as e:
                print(f"[WARN] Failed to build TensorRT engine: {e}")

        self.onnx_net = _ONNXNet(self.onnx_path)

        if self.use_trt:
            try:
                self.trt_net = _TRTNet(self.trt_path)
            except Exception as e:
                print(f"[WARN] TRT engine load failed: {e} — falling back to ONNX")
                self.trt_net = None
    def _build_trt_engine(self):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not installed")

        print(f"[INFO] Building TensorRT engine from '{self.onnx_path}'...")

        logger = trt.Logger(trt.Logger.INFO)

        builder = trt.Builder(logger)
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()

        serialized_engine = builder.build_serialized_network(
            network,
            config
        )

        with open(self.trt_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[INFO] TensorRT engine saved to '{self.trt_path}'")
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
        self.state           = SiamRPN_init(frame, target_pos, target_sz, self.pt_net)
        self.last_good_state = self.state.copy()
        self.score_ema       = None
        self.lost_count      = 0

        # export NOW — kernels are real at this exact point
        if self.use_onnx and self.onnx_net is None:
            self._export_with_real_kernels()

        print(f"[INFO] Tracker initialised | box: ({x_min},{y_min},{w},{h})")
        return (x_min, y_min, w, h)

    # -----------------------------------------------------------
    # TRACK STEP  (FastAPI / per-frame API)
    # -----------------------------------------------------------
    def track_step(self, frame: np.ndarray) -> dict:
        if self.state is None:
            raise RuntimeError("Call init_from_mask() before track_step()")

        active_net, backend = self._active_net

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

        _, backend = self._active_net
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