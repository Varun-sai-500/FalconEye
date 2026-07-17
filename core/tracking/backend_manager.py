import os
import time
import numpy as np
import torch

from core.utils.net import SiamRPNotb

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
# BACKEND MANAGER
# ---------------------------------------------------------------
class BackendManager:
    """
    Owns model lifecycle: PyTorch weight loading, ONNX export (post-temple,
    kernels frozen as constants), TensorRT engine build/load, and backend
    selection. DaSiamRPNTracker only ever talks to this class to get a net
    to run inference on — it has zero knowledge of onnx/trt internals.

    Backend priority once available: TensorRT → ONNX → PyTorch.
    """
    def __init__(self,
                 model_path: str = 'models/SiamRPNOTB.model',
                 onnx_path:  str = 'weights/search.onnx',
                 trt_path:   str = 'weights/search.engine',
                 use_onnx:   bool = True,
                 device=None):
        self.model_path = model_path
        self.onnx_path  = onnx_path
        self.trt_path   = trt_path
        self.use_onnx   = use_onnx and ORT_AVAILABLE
        self.use_trt    = TRT_AVAILABLE and torch.cuda.is_available()

        self.device = device or torch.device(
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

    @property
    def active_net(self):
        """Returns (net, backend_label). Priority: TRT → ONNX → PyTorch."""
        if self.trt_net is not None:
            return self.trt_net, "TensorRT"
        if self.onnx_net is not None:
            return self.onnx_net, "ONNX"
        return self.pt_net, "PyTorch"

    def get_pt_net(self):
        """Used by the tracker to run SiamRPN_init (calls pt_net.temple() internally)."""
        return self.pt_net

    # -----------------------------------------------------------
    # Export ONNX AFTER real temple() has run on pt_net
    # -----------------------------------------------------------
    def export_and_build(self):
        """
        Call this immediately after SiamRPN_init() has run pt_net.temple(real_z_crop),
        i.e. r1_kernel / cls1_kernel are real. do_constant_folding bakes them as
        frozen constants into the exported graph, leaving only the search branch
        dynamic. Builds/loads ONNX and, if available, TensorRT.

        No-op if ONNX export is disabled or already done.
        """
        if not self.use_onnx or self.onnx_net is not None:
            return

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
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, logger)

        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        # Without an explicit workspace limit, build_serialized_network can
        # silently return None on some TRT versions/GPUs instead of raising.
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

        serialized_engine = builder.build_serialized_network(
            network,
            config
        )

        with open(self.trt_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[INFO] TensorRT engine saved to '{self.trt_path}'")