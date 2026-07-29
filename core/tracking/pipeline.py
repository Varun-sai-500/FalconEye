import torch
import torch.nn.functional as F
from .crop import SubwindowCropper
from .anchors import generate_anchors

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
    target_dtype = x_crop.dtype  # Dynamically capture current running type (BFloat16/Float32)

    delta, score = net(x_crop, r1_kernel, cls1_kernel)
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0)[1, :]

    # Ensure anchor tensors dynamically match network output dtype
    anchor_t = p.anchor.to(dtype=target_dtype)

    cx = delta[0, :] * anchor_t[:, 2] + anchor_t[:, 0]
    cy = delta[1, :] * anchor_t[:, 3] + anchor_t[:, 1]
    w = torch.exp(delta[2, :]) * anchor_t[:, 2]
    h = torch.exp(delta[3, :]) * anchor_t[:, 3]

    def change(r):
        return torch.max(r, 1.0 / r)

    def sz(w_val, h_val):
        pad = (w_val + h_val) * 0.5
        return torch.sqrt((w_val + pad) * (h_val + pad))

    s_c = change(sz(w, h) / sz(target_sz[0], target_sz[1]))
    r_c = change((target_sz[0] / target_sz[1]) / (w / h))

    # Match literal scales to precision
    penalty = torch.exp(-(r_c * s_c - 1.0) * p.penalty_k)
    pscore = penalty * score
    pscore = pscore * (1.0 - p.window_influence) + window.to(dtype=target_dtype) * p.window_influence
    best_pscore_id = torch.argmax(pscore)

    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = cx[best_pscore_id] / scale_z + target_pos[0]
    res_y = cy[best_pscore_id] / scale_z + target_pos[1]
    res_w = target_sz[0] / scale_z * (1.0 - lr) + (w[best_pscore_id] / scale_z) * lr
    res_h = target_sz[1] / scale_z * (1.0 - lr) + (h[best_pscore_id] / scale_z) * lr

    target_pos = torch.stack([res_x, res_y])
    target_sz = torch.stack([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def DaSiamRPN_init(im, target_pos, target_sz, net):
    device = im.device
    target_dtype = im.dtype  # Capture image array entry type context
    state = dict()
    p = TrackerConfig()
    p.update(getattr(net, 'cfg', {}))

    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    # Initialize coordinate tracking values to target precision format
    if not isinstance(target_pos, torch.Tensor):
        target_pos = torch.tensor(target_pos, dtype=target_dtype, device=device)
    else:
        target_pos = target_pos.to(dtype=target_dtype, device=device)

    if not isinstance(target_sz, torch.Tensor):
        target_sz = torch.tensor(target_sz, dtype=target_dtype, device=device)
    else:
        target_sz = target_sz.to(dtype=target_dtype, device=device)

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287
        else:
            p.instance_size = 271
        p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1

    # Keep structural generation floating point, cast down inside runtime loops
    p.anchor = generate_anchors(p.total_stride, p.scales, p.ratios, int(p.score_size), device=device)

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
        hanning_1d = torch.hann_window(int(p.score_size), periodic=False, device=device, dtype=target_dtype)
        window_2d = torch.outer(hanning_1d, hanning_1d)
    elif p.windowing == 'uniform':
        window_2d = torch.ones((int(p.score_size), int(p.score_size)), device=device, dtype=target_dtype)
    window = window_2d.flatten().repeat(p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def DaSiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    target_dtype = im.dtype

    r1_kernel = state["r1_kernel"]
    cls1_kernel = state["cls1_kernel"]

    # Align state geometry constants to current input array frame data layout
    target_pos = target_pos.to(dtype=target_dtype)
    target_sz = target_sz.to(dtype=target_dtype)

    wc_z = target_sz[1] + p.context_amount * target_sz.sum()
    hc_z = target_sz[0] + p.context_amount * target_sz.sum()
    s_z = torch.sqrt(wc_z * hc_z)

    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = torch.round(s_z + 2 * pad)

    cropper_x = state['cropper_x']
    x_crop = cropper_x.crop(im, target_pos, s_x, avg_chans)

    target_pos, target_sz, score = tracker_eval(
        net, x_crop, r1_kernel, cls1_kernel, target_pos, target_sz * scale_z, window, scale_z, p
    )

    # Force continuous floating points for target geometry clamp limits
    target_pos[0] = torch.clamp(target_pos[0], min=0.0, max=float(state['im_w']))
    target_pos[1] = torch.clamp(target_pos[1], min=0.0, max=float(state['im_h']))
    target_sz[0] = torch.clamp(target_sz[0], min=10.0, max=float(state['im_w']))
    target_sz[1] = torch.clamp(target_sz[1], min=10.0, max=float(state['im_h']))

    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state