import os
import time
import numpy as np
import torch
from core.tracking.dasiamrpn import DaSiamRPNotb

torch.set_grad_enabled(False)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[WARN] onnxruntime not installed — using PyTorch inference")

try:
    import tensorrt as trt
    TRT_INSTALLED = True
except ImportError:
    TRT_INSTALLED = False
    print("[WARN] tensorrt not installed.")

CUDA_AVAILABLE = torch.cuda.is_available()
TRT_AVAILABLE = TRT_INSTALLED and CUDA_AVAILABLE


class TRTNet:
    def __init__(self, engine_path, score_size, anchor_num, device, dtype=torch.float16, logger=None):
        self.logger = logger if logger is not None else trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                "Failed to create TensorRT execution context."
            )

        self.score_size = score_size
        self.anchor_num = anchor_num
        self.device = device
        self.dtype = dtype
        self._validate_engine()

        # Pre-allocate output buffers natively on target device
        self.regression_buf = torch.empty(
            (1, 4 * anchor_num, score_size, score_size), device=self.device, dtype=self.dtype
        )
        self.classification_buf = torch.empty(
            (1, 2 * anchor_num, score_size, score_size), device=self.device, dtype=self.dtype
        )
        print(f"[INFO] TRT engine initialized successfully | score_size={score_size} | dtype={self.dtype}")

    def _load_engine(self, engine_path):
        runtime = trt.Runtime(self.logger)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
        return engine

    def _validate_engine(self):
        expected_dtype = (
            trt.DataType.BF16
            if self.dtype == torch.bfloat16
            else trt.DataType.HALF
        )

        expected = {
            "search_crop": trt.TensorIOMode.INPUT,
            "r1_kernel": trt.TensorIOMode.INPUT,
            "cls1_kernel": trt.TensorIOMode.INPUT,
            "regression": trt.TensorIOMode.OUTPUT,
            "classification": trt.TensorIOMode.OUTPUT,
        }

        available = {
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        }

        missing = set(expected) - available
        if missing:
            raise RuntimeError(
                f"Missing engine tensors: {sorted(missing)}"
            )

        for name, expected_mode in expected.items():
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)

            if mode != expected_mode:
                raise RuntimeError(
                    f"{name}: expected {expected_mode.name}, got {mode.name}"
                )

            if dtype != expected_dtype:
                raise RuntimeError(
                    f"{name}: expected {expected_dtype.name}, got {dtype.name}"
                )
            shape = tuple(self.engine.get_tensor_shape(name))

            print(
                f"[OK] {name:15}"
                f"{mode.name:6}"
                f"{dtype.name:6}"
                f"{shape}"
            )

    @torch.inference_mode()
    def forward(self, x_crop, r1_kernel, cls1_kernel):
        # Cast inputs straight to execution stream
        x_crop = x_crop.contiguous().to(self.device, dtype=self.dtype)
        r1_kernel = r1_kernel.contiguous().to(self.device, dtype=self.dtype)
        cls1_kernel = cls1_kernel.contiguous().to(self.device, dtype=self.dtype)

        # Zero-copy execution via direct PyTorch stream interaction
        current_stream = torch.cuda.current_stream().cuda_stream

        self.context.set_tensor_address("search_crop", x_crop.data_ptr())
        self.context.set_tensor_address("r1_kernel", r1_kernel.data_ptr())
        self.context.set_tensor_address("cls1_kernel", cls1_kernel.data_ptr())
        self.context.set_tensor_address("regression", self.regression_buf.data_ptr())
        self.context.set_tensor_address("classification", self.classification_buf.data_ptr())

        if not self.context.execute_async_v3(stream_handle=current_stream):
            raise RuntimeError("TensorRT execute_async_v3() failed.")

        return self.regression_buf, self.classification_buf

    def __call__(self, x_crop, r1_kernel, cls1_kernel):
        return self.forward(x_crop, r1_kernel, cls1_kernel)

    @staticmethod
    def build_trt_engine(onnx_path, trt_path, workspace_size=1 << 30, dtype=torch.float16):
        print(f"[INFO] Executing offline local TensorRT serialization engine compilation ({dtype})...")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)

        # TensorRT 10/11 implicitly uses EXPLICIT_BATCH; create_network() needs no extra flags
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                print(f"[ONNX Parser Error]: {parser.get_error(i)}")
            raise RuntimeError("ONNX Parser parsing integrity failure.")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        print(f"[INFO] Building TensorRT engine (requested dtype={dtype})")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT compilation interface returned null pointer.")

        trt_dir = os.path.dirname(trt_path)
        if trt_dir:
            os.makedirs(trt_dir, exist_ok=True)

        tmp_path = trt_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(serialized_engine)
        os.replace(tmp_path, trt_path)
        print(f"[INFO] TensorRT compilation successful → persistent engine mapped: '{trt_path}'")


class ONNXNet:
    def __init__(self, onnx_path, score_size, anchor_num,device, dtype=torch.float16):
        from onnx import TensorProto
        self.device = device
        self.dtype = dtype

        if self.dtype == torch.bfloat16:
            self.ort_element_type = TensorProto.BFLOAT16
        elif self.dtype == torch.float16:
            self.ort_element_type = np.float16
        else:
            self.ort_element_type = np.float32

        if ORT_AVAILABLE and CUDA_AVAILABLE:
            current_stream_ptr = torch.cuda.current_stream().cuda_stream
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': '0',
                    'user_compute_stream': str(current_stream_ptr)
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']

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

            # Pre-allocate output buffers permanently
            self.reg_buf = torch.empty((1, 4 * anchor_num, score_size, score_size), device=self.device, dtype=self.dtype)
            self.cls_buf = torch.empty((1, 2 * anchor_num, score_size, score_size), device=self.device, dtype=self.dtype)

            # TRUE ZERO-COPY: Bind outputs permanently right here in init
            self.io_binding.bind_output(
                name=self.output_names[0], device_type='cuda', device_id=0,
                element_type=self.ort_element_type, shape=tuple(self.reg_buf.shape),
                buffer_ptr=self.reg_buf.data_ptr()
            )
            self.io_binding.bind_output(
                name=self.output_names[1], device_type='cuda', device_id=0,
                element_type=self.ort_element_type, shape=tuple(self.cls_buf.shape),
                buffer_ptr=self.cls_buf.data_ptr()
            )

        print(f"[INFO] ONNX session initialized | provider: {self.session.get_providers()[0]}")

    def __call__(self, x_crop, r1_kernel, cls1_kernel):
        if not torch.is_tensor(x_crop):
            x_crop = torch.from_numpy(x_crop).to(self.device)

        if self.using_cuda:
            x_crop = x_crop.contiguous().to(device=self.device, dtype=self.dtype)
            r1_kernel = r1_kernel.contiguous().to(device=self.device, dtype=self.dtype)
            cls1_kernel = cls1_kernel.contiguous().to(device=self.device, dtype=self.dtype)

            # Clean only the changing input registers
            self.io_binding.clear_binding_inputs()

            # Dynamic input mappings
            self.io_binding.bind_input(
                name=self.search_name, device_type='cuda', device_id=0,
                element_type=self.ort_element_type, shape=tuple(x_crop.shape),
                buffer_ptr=x_crop.data_ptr()
            )
            self.io_binding.bind_input(
                name=self.r1_name, device_type='cuda', device_id=0,
                element_type=self.ort_element_type, shape=tuple(r1_kernel.shape),
                buffer_ptr=r1_kernel.data_ptr()
            )
            self.io_binding.bind_input(
                name=self.cls1_name, device_type='cuda', device_id=0,
                element_type=self.ort_element_type, shape=tuple(cls1_kernel.shape),
                buffer_ptr=cls1_kernel.data_ptr()
            )
            self.session.run_with_iobinding(self.io_binding)

            return self.reg_buf, self.cls_buf
        else:
            x_np = x_crop.cpu().to(self.dtype).numpy()
            r1_np = r1_kernel.cpu().to(self.dtype).numpy()
            cls1_np = cls1_kernel.cpu().to(self.dtype).numpy()

            feeds = {self.search_name: x_np, self.r1_name: r1_np, self.cls1_name: cls1_np}
            regression, classification = self.session.run(None, feeds)
            return (
                torch.from_numpy(regression).to(device=self.device, dtype=self.dtype),
                torch.from_numpy(classification).to(device=self.device, dtype=self.dtype),
            )


class BackendManager:
    def __init__(self,
                 model_path: str = 'models/SiamRPNOTB.model',
                 onnx_path:  str = 'weights/search.onnx',
                 trt_path:   str = 'weights/search.engine',
                 use_onnx:   bool = True,
                 instance_size: int = 271,
                 custom_stride_calc: bool = False,
                 exemplar_size: int = 127,
                 total_stride: int = 8,
                 anchor_num: int = 5,
                 device=None,
                 benchmark: bool = False):

        self.model_path = model_path
        self.onnx_path  = onnx_path
        self.trt_path   = trt_path
        self.use_onnx   = use_onnx and ORT_AVAILABLE
        self.use_trt    = TRT_AVAILABLE and torch.cuda.is_available()

        self.instance_size = instance_size
        self.exemplar_size = exemplar_size
        self.score_size = (instance_size - exemplar_size) // total_stride + 1
        self.anchor_num = anchor_num
        self.model_fps = 0.0
        self.benchmark = benchmark

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            major, _ = torch.cuda.get_device_capability(0)
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
            self.device_name = torch.cuda.get_device_name(0)

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self.device = torch.device("mps")
            self.dtype = torch.float16
            self.device_name = "Apple Metal (MPS)"

        else:
            import platform

            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.device_name = platform.processor() or "CPU"

        self.pt_net = DaSiamRPNotb()
        if os.path.exists(model_path):
            self.pt_net.load_state_dict(torch.load(model_path, map_location=self.device))

        self.pt_net.eval().to(device=self.device, dtype=self.dtype)
        print(f"[INFO] Base PyTorch network mounted | device: {self.device} | dtype: {self.dtype}")

        self.onnx_net = None
        self.trt_net  = None

    @property
    def active_net(self):
        if self.trt_net is not None:
            return self.trt_net, "TensorRT"
        if self.onnx_net is not None:
            return self.onnx_net, "ONNX"
        return self.pt_net, "PyTorch"

    def get_pt_net(self):
        return self.pt_net

    def export_and_build(self, r1_kernel, cls1_kernel):
        r1_kernel_half = r1_kernel.to(self.dtype)
        cls1_kernel_half = cls1_kernel.to(self.dtype)

        if self.use_onnx and self.onnx_net is None:
            print(f"[INFO] Exporting search.onnx...")
            dummy_x = torch.zeros(1, 3, self.instance_size, self.instance_size, device=self.device, dtype=self.dtype)

            with torch.inference_mode():
                onnx_dir = os.path.dirname(self.onnx_path)
                if onnx_dir:
                    os.makedirs(onnx_dir, exist_ok=True)

                torch.onnx.export(
                    self.pt_net,
                    (dummy_x, r1_kernel_half, cls1_kernel_half),
                    self.onnx_path,
                    export_params=True,
                    input_names=["search_crop", "r1_kernel", "cls1_kernel"],
                    output_names=["regression", "classification"],
                    opset_version=18,
                    do_constant_folding=True,
                )
            print(f"[INFO] Structural trace completed → saved to '{self.onnx_path}'")

            if TRT_AVAILABLE and not os.path.exists(self.trt_path):
                try:
                    print("[WARN] Compiling TensorRT runtime workspace engine...")
                    TRTNet.build_trt_engine(self.onnx_path, self.trt_path, dtype=self.dtype)
                except Exception as e:
                    print(f"[WARN] TensorRT automatic compilation aborted: {e}")

            self.onnx_net = ONNXNet(
                onnx_path=self.onnx_path,
                score_size=self.score_size,
                anchor_num=self.anchor_num,
                device = self.device,
                dtype=self.dtype
            )

            if self.use_trt and os.path.exists(self.trt_path):
                try:
                    self.trt_net = TRTNet(
                        engine_path=self.trt_path,
                        score_size=self.score_size,
                        anchor_num=self.anchor_num,
                        device=self.device,
                        dtype=self.dtype
                    )
                except Exception as e:
                    print(f"[WARN] TensorRT Context map failed: {e} — falling back securely to ONNX.")
                    self.trt_net = None

        if self.benchmark:
            dummy_x = torch.zeros(1, 3, self.instance_size, self.instance_size, device=self.device, dtype=self.dtype)
            self.run_benchmark(dummy_x, r1_kernel_half, cls1_kernel_half)

    @torch.inference_mode()
    def run_benchmark(self, x_crop, r1_kernel, cls1_kernel, iterations=300, warmup=30):
        net, name = self.active_net
        print(f"\n[BENCHMARK] Starting isolation sweep for active backend: {name}...")

        x_crop = x_crop.contiguous().to(self.device, dtype=self.dtype)
        r1_kernel = r1_kernel.contiguous().to(self.device, dtype=self.dtype)
        cls1_kernel = cls1_kernel.contiguous().to(self.device, dtype=self.dtype)

        for _ in range(warmup):
            net(x_crop, r1_kernel, cls1_kernel)

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

            target_stream = torch.cuda.current_stream(self.device)

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            for i in range(iterations):
                start_events[i].record(target_stream)
                net(x_crop, r1_kernel, cls1_kernel)
                end_events[i].record(target_stream)

            torch.cuda.synchronize()

            latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        else:
            latencies = []

            for _ in range(iterations):
                start = time.perf_counter()
                net(x_crop, r1_kernel, cls1_kernel)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        self.model_fps = fps

        print("\n" + "=" * 60)
        print(f" BENCHMARK RUNTIME REPORT: {name.upper()} ({self.dtype})")
        print("=" * 60)
        print(f" * Device             : {self.device_name}")
        print(f" * Pure Inference FPS : {fps:.2f} FPS")
        print(f" * Mean Latency       : {avg_latency:.3f} ms")
        print(f" * P95 Latency        : {p95_latency:.3f} ms")
        print(f" * P99 Latency        : {p99_latency:.3f} ms")
        print("=" * 60 + "\n")