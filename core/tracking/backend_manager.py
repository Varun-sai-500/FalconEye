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


def trt_infer(context, x_crop, r1_kernel, cls1_kernel, regression_buf, classification_buf):
    if isinstance(x_crop, np.ndarray):
        x_crop = torch.from_numpy(x_crop).cuda()

    x_crop = x_crop.contiguous().float()
    if not x_crop.is_cuda:
        x_crop = x_crop.cuda()

    # Zero Host-to-Device / Device-to-Host overhead via static GPU virtual address pointers
    context.set_tensor_address("search_crop", x_crop.data_ptr())
    context.set_tensor_address("r1_kernel", r1_kernel.data_ptr())
    context.set_tensor_address("cls1_kernel", cls1_kernel.data_ptr())
    context.set_tensor_address("regression", regression_buf.data_ptr())
    context.set_tensor_address("classification", classification_buf.data_ptr())

    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    return regression_buf, classification_buf


class _TRTNet:
    def __init__(self, engine_path, score_size, anchor_num):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not installed — cannot load engine")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — cannot run TRT engine")

        self.engine  = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.last_model_fps = 0.0

        self.score_size = score_size
        self.anchor_num = anchor_num

        # Pre-allocated permanent tracking buffers (Avoids runtime frame-by-frame memory pooling spikes)
        self.regression_buf = torch.empty(
            (1, 4 * anchor_num, score_size, score_size), device="cuda", dtype=torch.float32
        )
        self.classification_buf = torch.empty(
            (1, 2 * anchor_num, score_size, score_size), device="cuda", dtype=torch.float32
        )
        print(f"[INFO] TRT engine initialized | score_size={score_size}")

    def __call__(self, x_crop, r1_kernel, cls1_kernel):
        t0 = time.perf_counter()
        regression, classification = trt_infer(
            self.context,
            x_crop,
            r1_kernel,
            cls1_kernel,
            self.regression_buf,
            self.classification_buf
        )
        self.last_model_fps = 1.0 / (time.perf_counter() - t0)
        return regression, classification

    def temple(self, z):
        pass

    cfg = {}


# ---------------------------------------------------------------
# ZERO-COPY ONNX RUNTIME INTERFACE (VIA IO-BINDING MAPPING)
# ---------------------------------------------------------------
class _ONNXNet:
    def __init__(self, onnx_path, device=None):
        self.last_model_fps = 0.0
        self.device = device or torch.device('cpu')
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if ORT_AVAILABLE and 'CUDAExecutionProvider' in ort.get_available_providers()
                     else ['CPUExecutionProvider'])

        so = ort.SessionOptions()
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

        inputs = self.session.get_inputs()
        self.search_name = inputs[0].name
        self.r1_name = inputs[1].name
        self.cls1_name = inputs[2].name

        self.output_names = [o.name for o in self.session.get_outputs()]
        self.using_cuda = 'CUDAExecutionProvider' in self.session.get_providers()

        if self.using_cuda:
            self.io_binding = self.session.io_binding()

        print(f"[INFO] ONNX session initialized | provider: {self.session.get_providers()[0]}")

    def _helper_bind_tensor(self, name, tensor):
        self.io_binding.bind_input(
            name=name,
            device_type='cuda',
            device_id=tensor.device.index or 0,
            element_type=np.float32,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    def __call__(self, x_crop, r1_kernel, cls1_kernel):
        if not torch.is_tensor(x_crop):
            x_crop = torch.from_numpy(x_crop).to(self.device)
        x_crop = x_crop.contiguous().float()

        t0 = time.perf_counter()

        if self.using_cuda:
            x_crop = x_crop.to(self.device, non_blocking=True)
            r1_kernel = r1_kernel.to(self.device, non_blocking=True)
            cls1_kernel = cls1_kernel.to(self.device, non_blocking=True)

            self.io_binding.clear_binding_inputs()
            self.io_binding.clear_binding_outputs()

            self._helper_bind_tensor(self.search_name, x_crop)
            self._helper_bind_tensor(self.r1_name, r1_kernel)
            self._helper_bind_tensor(self.cls1_name, cls1_kernel)

            for name in self.output_names:
                self.io_binding.bind_output(name, device_type='cuda', device_id=x_crop.device.index or 0)

            self.session.run_with_iobinding(self.io_binding)
            outs = self.io_binding.get_outputs()

            regression = torch.utils.dlpack.from_dlpack(outs[0].to_dlpack()) if hasattr(outs[0], "to_dlpack") \
                else torch.as_tensor(outs[0].numpy(), device=self.device)
            classification = torch.utils.dlpack.from_dlpack(outs[1].to_dlpack()) if hasattr(outs[1], "to_dlpack") \
                else torch.as_tensor(outs[1].numpy(), device=self.device)
        else:
            x_np = x_crop.cpu().numpy()
            r1_np = r1_kernel.cpu().numpy()
            cls1_np = cls1_kernel.cpu().numpy()

            feeds = {self.search_name: x_np, self.r1_name: r1_np, self.cls1_name: cls1_np}
            regression, classification = self.session.run(None, feeds)
            regression = torch.from_numpy(regression).to(self.device)
            classification = torch.from_numpy(classification).to(self.device)

        self.last_model_fps = 1.0 / (time.perf_counter() - t0)
        return regression, classification

    def temple(self, z):
        pass

    cfg = {}


# ---------------------------------------------------------------
# UNIFIED COMPILATION & BACKEND ARCHITECTURE LIFE-CYCLE MANAGER
# ---------------------------------------------------------------
class BackendManager:
    def __init__(self,
                 model_path: str = 'models/SiamRPNOTB.model',
                 onnx_path:  str = 'weights/search.onnx',
                 trt_path:   str = 'weights/search.engine',
                 use_onnx:   bool = True,
                 instance_size: int = 271,
                 exemplar_size: int = 127,
                 total_stride: int = 8,
                 anchor_num: int = 5,
                 device=None):
        self.model_path = model_path
        self.onnx_path  = onnx_path
        self.trt_path   = trt_path
        self.use_onnx   = use_onnx and ORT_AVAILABLE
        self.use_trt    = TRT_AVAILABLE and torch.cuda.is_available()

        self.instance_size = instance_size
        self.exemplar_size = exemplar_size
        self.score_size = (instance_size - exemplar_size) // total_stride + 1
        self.anchor_num = anchor_num

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        # Baseline PyTorch initialization path
        self.pt_net = SiamRPNotb()
        if os.path.exists(model_path):
            self.pt_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pt_net.eval().to(self.device)
        print(f"[INFO] Base PyTorch network mounted | device: {self.device}")

        self.onnx_net = None
        self.trt_net  = None

    @property
    def active_net(self):
        """Returns the absolute highest-priority ready acceleration backend."""
        if self.trt_net is not None:
            return self.trt_net, "TensorRT"
        if self.onnx_net is not None:
            return self.onnx_net, "ONNX"
        return self.pt_net, "PyTorch"

    def get_pt_net(self):
        return self.pt_net

    def export_and_build(self, r1_kernel, cls1_kernel):
        assert r1_kernel.device == self.device
        assert cls1_kernel.device == self.device
        """
        Traces and saves the structural execution graph using the real tracking kernels
        computed directly from Frame 0. Eliminates dummy image generation bottlenecks entirely.
        """
        if not self.use_onnx or self.onnx_net is not None:
            return

        print(f"[INFO] Exporting search.onnx using real frame-0 target context tensors...")
        dummy_x = torch.zeros(1, 3, self.instance_size, self.instance_size).to(self.device)

        with torch.no_grad():
            os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)

            torch.onnx.export(
                self.pt_net,
                (dummy_x, r1_kernel, cls1_kernel),
                self.onnx_path,
                input_names=["search_crop", "r1_kernel", "cls1_kernel"],
                output_names=["regression", "classification"],
                opset_version=18,
                do_constant_folding=True,
            )
        print(f"[INFO] Structural trace completed → saved to '{self.onnx_path}'")

        if TRT_AVAILABLE and not os.path.exists(self.trt_path):
            try:
                self._build_trt_engine()
            except Exception as e:
                print(f"[WARN] TensorRT automatic compilation aborted: {e}")

        self.onnx_net = _ONNXNet(self.onnx_path, device=self.device)

        if self.use_trt:
            try:
                self.trt_net = _TRTNet(self.trt_path, self.score_size, self.anchor_num)
            except Exception as e:
                print(f"[WARN] TensorRT Context map failed: {e} — falling back securely to ONNX runtime engine.")
                self.trt_net = None

    def _build_trt_engine(self):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT backend driver unavailable.")

        print(f"[INFO] Executing offline local TensorRT serialization engine compilation...")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, logger)

        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX Parser parsing integrity failure.")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # Allocates exactly 1 GiB memory pool window

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT compilation interface returned null pointer.")

        os.makedirs(os.path.dirname(self.trt_path), exist_ok=True)
        tmp_path = self.trt_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(serialized_engine)
        os.replace(tmp_path, self.trt_path)

        print(f"[INFO] TensorRT compilation successful → persistent engine mapped: '{self.trt_path}'")