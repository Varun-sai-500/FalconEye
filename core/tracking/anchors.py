import torch

def generate_anchors(total_stride, scales, ratios, score_size, device, dtype=torch.float32):
    """
    Generates tracking anchors matching the running precision of the pipeline.
    Passed dtype determines if it builds in Float32, BFloat16, or Float16.
    """
    anchor_num = len(ratios) * len(scales)
    score_size = int(score_size)

    # Dynamically apply targeting precision format
    ratios_t = torch.tensor(ratios, dtype=dtype, device=device)
    scales_t = torch.tensor(scales, dtype=dtype, device=device)

    size = total_stride * total_stride

    ws = torch.sqrt(size / ratios_t).int().to(dtype=dtype)
    hs = (ws * ratios_t).int().to(dtype=dtype)

    wws = (ws.unsqueeze(1) * scales_t).flatten()
    hhs = (hs.unsqueeze(1) * scales_t).flatten()

    base_anchors = torch.stack([
        torch.zeros_like(wws),
        torch.zeros_like(hhs),
        wws,
        hhs
    ], dim=-1)  # (anchor_num, 4)

    ori = -(score_size / 2.0) * total_stride

    # Match the coordinate space grid directly to the model execution precision
    grid_linear = torch.arange(score_size, dtype=dtype, device=device) * total_stride + ori
    yy, xx = torch.meshgrid(grid_linear, grid_linear, indexing='ij')

    xx = xx.flatten().repeat(anchor_num)
    yy = yy.flatten().repeat(anchor_num)
    anchors = base_anchors.repeat_interleave(score_size * score_size, dim=0)
    anchors[:, 0] = xx
    anchors[:, 1] = yy

    return anchors