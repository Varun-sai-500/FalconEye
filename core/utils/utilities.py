import torch
import torch.nn.functional as F


class SubwindowCropper:
    def __init__(self, model_sz, device):
        self.model_sz = model_sz
        self.device = device
        idx = torch.arange(model_sz, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(idx, idx, indexing="ij")
        self.base_idx = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        self._size_wh_cache = {}  # keyed by (W, H) -> cached tensor, no re-alloc for steady-state fixed resolution

    def _get_image_size_tensor(self, W, H, device):
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
        size_wh = self._get_image_size_tensor(W, H, im_t.device)

        avg = avg_chans_t.reshape(1, C, 1, 1)

        # Center image around channel mean so zero padding corresponds to mean-value padding.
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

def generate_anchors(total_stride, scales, ratios, score_size, device):
    anchor_num = len(ratios) * len(scales)
    score_size = int(score_size)

    ratios_t = torch.tensor(ratios, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)

    size = total_stride * total_stride

    ws = torch.sqrt(size / ratios_t).int().float()
    hs = (ws * ratios_t).int().float()

    wws = (ws.unsqueeze(1) * scales_t).flatten()
    hhs = (hs.unsqueeze(1) * scales_t).flatten()

    base_anchors = torch.stack([
        torch.zeros_like(wws),
        torch.zeros_like(hhs),
        wws,
        hhs
    ], dim=-1)  # (anchor_num, 4)

    ori = -(score_size / 2.0) * total_stride
    grid_linear = torch.arange(score_size, dtype=torch.float32, device=device) * total_stride + ori
    yy, xx = torch.meshgrid(grid_linear, grid_linear, indexing='ij')

    xx = xx.flatten().repeat(anchor_num)
    yy = yy.flatten().repeat(anchor_num)
    anchors = base_anchors.repeat_interleave(score_size * score_size, dim=0)
    anchors[:, 0] = xx
    anchors[:, 1] = yy

    return anchors

def cxy_wh_2_rect(pos, sz):
    return torch.stack((
        pos[0] - sz[0] * 0.5,
        pos[1] - sz[1] * 0.5,
        sz[0],
        sz[1],
    ))
