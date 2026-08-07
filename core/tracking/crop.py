import torch
import torch.nn.functional as F

class SubwindowCropper:
    def __init__(self, model_sz, device):
        self.model_sz = model_sz
        self.device = device
        self._base_idx_cache = {}
        idx = torch.arange(model_sz, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(idx, idx, indexing="ij")
        self.base_idx = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        self._size_wh_cache = {}  # keyed by (W, H, dtype) -> cached tensor

    def _get_image_size_tensor(self, W, H, device, dtype):
        key = (W, H, dtype)
        if key not in self._size_wh_cache:
            self._size_wh_cache[key] = torch.tensor([float(W), float(H)], device=device, dtype=dtype)
        return self._size_wh_cache[key]

    @torch.inference_mode()
    def crop(self, im_t, pos_t, original_sz, avg_chans_t):
        """
        im_t         : (H, W, C) float32/bfloat16/float16 CUDA or CPU tensor
        pos_t        : (2,) [cx, cy]
        original_sz  : scalar float/tensor
        avg_chans_t  : (3,)
        """
        H, W, C = im_t.shape
        target_dtype = im_t.dtype # Dynamically capture BFloat16/Float32

        size_wh = self._get_image_size_tensor(W, H, im_t.device, target_dtype)
        avg = avg_chans_t.reshape(1, C, 1, 1)

        # Center image around channel mean so zero padding corresponds to mean-value padding.
        im_centered = (im_t - avg_chans_t).permute(2, 0, 1).unsqueeze(0)

        # Ensure geometric inputs match the core image precision context
        pos_t = pos_t.to(dtype=target_dtype)
        original_sz = original_sz.to(dtype=target_dtype) if isinstance(original_sz, torch.Tensor) else original_sz

        context_min = pos_t - original_sz * 0.5
        scale = original_sz / self.model_sz

        # Matches OpenCV resize convention - cast base_idx to match target precision
        if target_dtype not in self._base_idx_cache:
            self._base_idx_cache[target_dtype] = self.base_idx.to(dtype=target_dtype)
        base_idx = self._base_idx_cache[target_dtype]
        src_local = (base_idx + 0.5) * scale - 0.5
        abs_coords = context_min[None, None, None] + src_local

        # grid calculation perfectly mirrors the data type of the input tensor
        grid = (abs_coords + 0.5) * (2.0 / size_wh) - 1.0

        patch = F.grid_sample(
            im_centered,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return patch + avg