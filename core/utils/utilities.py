import torch
import torch.nn.functional as F


class SubwindowCropper:
    def __init__(self, model_sz, device="cuda"):
        self.model_sz = model_sz
        self.device = device
        idx = torch.arange(model_sz, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(idx, idx, indexing="ij")
        self.base_idx = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        self._size_wh_cache = {}  # keyed by (W, H) -> cached tensor, no re-alloc for steady-state fixed resolution

    def _get_size_wh(self, W, H, device):
        key = (W, H)
        if key not in self._size_wh_cache:
            self._size_wh_cache[key] = torch.tensor([float(W), float(H)], device=device, dtype=torch.float32)
        return self._size_wh_cache[key]


    @torch.no_grad()
    def crop(self, im_t, pos_t, original_sz, avg_chans_t):
        """
        im_t        : (H, W, C) float32 CUDA tensor
        pos_t       : (2,) [cx, cy]
        original_sz : scalar float/tensor
        avg_chans_t : (3,)
        """
        H, W, C = im_t.shape
        size_wh = self._get_size_wh(W, H, im_t.device)
        C = im_t.shape[-1]

        avg = avg_chans_t.reshape(1, C, 1, 1)

        # Mean-pad trick
        im_centered = (im_t - avg_chans_t).permute(2, 0, 1).unsqueeze(0)

        context_min = pos_t - original_sz * 0.5

        scale = original_sz / self.model_sz

        # Matches OpenCV resize convention
        src_local = (self.base_idx + 0.5) * scale - 0.5
        abs_coords = context_min[None, None, None] + src_local

        grid = (abs_coords + 0.5) * (2.0 / size_wh) - 1.0

        patch = F.grid_sample(
            im_centered,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return patch + avg


def cxy_wh_2_rect(pos, sz):
    return torch.stack((
        pos[0] - sz[0] * 0.5,
        pos[1] - sz[1] * 0.5,
        sz[0],
        sz[1],
    ))