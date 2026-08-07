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

def tracker_geometry_update(
    delta,
    score,
    anchor,
    target_sz,
    target_pos,
    window,
    scale_z,
    penalty_k,
    window_influence,
    lr_coef,
):

    cx = (delta[0] * anchor[:, 2] + anchor[:, 0]).to(anchor.dtype)
    cy = (delta[1] * anchor[:, 3] + anchor[:, 1]).to(anchor.dtype)
    w  = (torch.exp(delta[2]) * anchor[:, 2]).to(anchor.dtype)
    h  = (torch.exp(delta[3]) * anchor[:, 3]).to(anchor.dtype)

    def change(r):
        return torch.maximum(r, 1.0 / r)

    def sz(w_val, h_val):
        pad = (w_val + h_val) * 0.5
        return torch.sqrt((w_val + pad) * (h_val + pad))

    s_c = change(
        sz(w, h) /
        sz(target_sz[0], target_sz[1])
    )

    r_c = change(
        (target_sz[0] / target_sz[1]) /
        (w / h)
    )

    penalty = torch.exp(
        -(r_c * s_c - 1.0) * penalty_k
    )

    pscore = penalty * score

    pscore = (
        pscore * (1.0 - window_influence)
        +
        window * window_influence
    )

    best = torch.argmax(pscore)

    lr = (
        penalty[best]
        *
        score[best]
        *
        lr_coef
    )

    res_x = cx[best] / scale_z + target_pos[0]
    res_y = cy[best] / scale_z + target_pos[1]

    res_w = (
        target_sz[0] / scale_z * (1.0 - lr)
        +
        w[best] / scale_z * lr
    )

    res_h = (
        target_sz[1] / scale_z * (1.0 - lr)
        +
        h[best] / scale_z * lr
    )

    return (
        torch.stack([res_x, res_y]),
        torch.stack([res_w, res_h]),
        score[best]
    )

@torch.inference_mode()
def DaSiamRPN_init(im, target_pos, target_sz, net):
    device = im.device
    target_dtype = im.dtype  # Capture image array entry type context
    state = dict()
    p = TrackerConfig()
    p.update(getattr(net, 'cfg', {}))

    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    if device.type == "cpu":
        state["geometry"] = tracker_geometry_update
    else:
        state["geometry"] = torch.compile(
            tracker_geometry_update,
            mode="reduce-overhead",
        )
    if not isinstance(target_pos, torch.Tensor):
        target_pos = torch.tensor(target_pos, dtype=torch.float32, device=device)
    else:
        target_pos = target_pos.to(dtype=torch.float32, device=device)

    if not isinstance(target_sz, torch.Tensor):
        target_sz = torch.tensor(target_sz, dtype=torch.float32, device=device)
    else:
        target_sz = target_sz.to(dtype=torch.float32, device=device)

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287
        else:
            p.instance_size = 271
        p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1

    # Keep structural generation floating point, cast down inside runtime loops
    p.anchor = generate_anchors(
        p.total_stride,
        p.scales,
        p.ratios,
        int(p.score_size),
        device=device,
        dtype=torch.float32
    )
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
    state["anchor"] = p.anchor.to(dtype=r1_kernel.dtype)


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

@torch.inference_mode()
def DaSiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    anchor = state['anchor']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

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

    delta, score = net(
        x_crop,
        r1_kernel,
        cls1_kernel
    )
    delta = delta.permute(1,2,3,0).contiguous().view(4,-1)

    score = F.softmax(
        score.permute(1,2,3,0)
        .contiguous()
        .view(2,-1),
        dim=0
    )[1]

    target_pos, target_sz, score = state["geometry"](
        delta,
        score,
        anchor,
        target_sz * scale_z,
        target_pos,
        window,
        scale_z,
        p.penalty_k,
        p.window_influence,
        p.lr,
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