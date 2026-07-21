import torch
import torch.nn.functional as F
from .utilities import SubwindowCropper, cxy_wh_2_rect


def generate_anchor_torch(total_stride, scales, ratios, score_size, device="cuda"):
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


class TrackerConfig(object):
    windowing = 'cosine'
    max_model_fps = 0.0
    exemplar_size = 127
    instance_size = 271
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1
    context_amount = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8]
    anchor_num = len(ratios) * len(scales)
    anchor = None
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1


def tracker_eval(net, x_crop, r1_kernel, cls1_kernel, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop, r1_kernel, cls1_kernel)
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0)[1, :]

    cx = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    cy = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    w = torch.exp(delta[2, :]) * p.anchor[:, 2]
    h = torch.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return torch.max(r, 1.0 / r)

    def sz(w_val, h_val):
        pad = (w_val + h_val) * 0.5
        return torch.sqrt((w_val + pad) * (h_val + pad))

    s_c = change(sz(w, h) / sz(target_sz[0], target_sz[1]))
    r_c = change((target_sz[0] / target_sz[1]) / (w / h))

    penalty = torch.exp(-(r_c * s_c - 1.0) * p.penalty_k)
    pscore = penalty * score
    pscore = pscore * (1.0 - p.window_influence) + window * p.window_influence
    best_pscore_id = torch.argmax(pscore)

    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = cx[best_pscore_id] / scale_z + target_pos[0]
    res_y = cy[best_pscore_id] / scale_z + target_pos[1]
    res_w = target_sz[0] / scale_z * (1.0 - lr) + (w[best_pscore_id] / scale_z) * lr
    res_h = target_sz[1] / scale_z * (1.0 - lr) + (h[best_pscore_id] / scale_z) * lr

    target_pos = torch.stack([res_x, res_y])
    target_sz = torch.stack([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net):
    device = im.device
    state = dict()
    p = TrackerConfig()
    p.update(getattr(net, 'cfg', {}))

    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if not isinstance(target_pos, torch.Tensor):
        target_pos = torch.tensor(target_pos, dtype=torch.float32, device=device)
    if not isinstance(target_sz, torch.Tensor):
        target_sz = torch.tensor(target_sz, dtype=torch.float32, device=device)

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287
        else:
            p.instance_size = 271
        p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1

    p.anchor = generate_anchor_torch(p.total_stride, p.scales, p.ratios, int(p.score_size), device=device)

    avg_chans = im.mean(dim=(0, 1))

    cropper_z = SubwindowCropper(model_sz=p.exemplar_size, device=device)
    state['cropper_z'] = cropper_z

    cropper_x = SubwindowCropper(model_sz=p.instance_size, device=device)
    state['cropper_x'] = cropper_x

    wc_z = target_sz[0] + p.context_amount * target_sz.sum()
    hc_z = target_sz[1] + p.context_amount * target_sz.sum()
    s_z = torch.round(torch.sqrt(wc_z * hc_z))

    z_crop = cropper_z.crop(im, target_pos, s_z, avg_chans)
    r1_kernel, cls1_kernel = net.extract_template(z_crop)

    state["r1_kernel"] = r1_kernel
    state["cls1_kernel"] = cls1_kernel

    if p.windowing == 'cosine':
        hanning_1d = torch.hann_window(int(p.score_size), periodic=False, device=device)
        window_2d = torch.outer(hanning_1d, hanning_1d)
    elif p.windowing == 'uniform':
        window_2d = torch.ones((int(p.score_size), int(p.score_size)), device=device)
    window = window_2d.flatten().repeat(p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    # FIXED: Extract kernels safely from state storage map to satisfy tracker_eval signatures
    r1_kernel = state["r1_kernel"]
    cls1_kernel = state["cls1_kernel"]

    wc_z = target_sz[1] + p.context_amount * target_sz.sum()
    hc_z = target_sz[0] + p.context_amount * target_sz.sum()
    s_z = torch.sqrt(wc_z * hc_z)

    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = torch.round(s_z + 2 * pad)

    cropper_x = state['cropper_x']
    x_crop = cropper_x.crop(im, target_pos, s_x, avg_chans)

    # FIXED: Patched invocation positional arguments to map properly into eval loop
    target_pos, target_sz, score = tracker_eval(
        net, x_crop, r1_kernel, cls1_kernel, target_pos, target_sz * scale_z, window, scale_z, p
    )

    target_pos[0] = torch.clamp(target_pos[0], min=0.0, max=float(state['im_w']))
    target_pos[1] = torch.clamp(target_pos[1], min=0.0, max=float(state['im_h']))
    target_sz[0] = torch.clamp(target_sz[0], min=10.0, max=float(state['im_w']))
    target_sz[1] = torch.clamp(target_sz[1], min=10.0, max=float(state['im_h']))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state