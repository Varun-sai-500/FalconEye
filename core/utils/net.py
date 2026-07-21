import torch
import torch.nn as nn
import torch.nn.functional as F

class SiamRPN(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        super(SiamRPN, self).__init__()
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x == 3 else x * size, configs))
        feat_in = configs[-1]

        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )

        self.anchor = anchor
        self.feature_out = feature_out

        # Template branches (kernel generators)
        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * anchor, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * anchor, 3)

        # Search branches
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)

        self.regress_adjust = nn.Conv2d(4 * anchor, 4 * anchor, 1)

    def extract_template(self, z):
        """
        Extracts template features once on frame 0.
        Returns the raw tracking kernels instead of storing them as state.
        """
        z_f = self.featureExtract(z)
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)

        # Get shape safely using standard tensor tracking
        kernel_size = r1_kernel_raw.shape[-1]

        r1_kernel = r1_kernel_raw.view(self.anchor * 4, self.feature_out, kernel_size, kernel_size)
        cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)

        return r1_kernel, cls1_kernel

    def forward(self, x, r1_kernel, cls1_kernel):
        """
        Pure stateless tracking forward pass.
        Fully compatible with torch.compile() and CUDA graphs.
        """
        x_f = self.featureExtract(x)

        # 1. Extract search features
        r2_out = self.conv_r2(x_f)
        cls2_out = self.conv_cls2(x_f)

        # 2. Reshape for grouped cross-correlation.
        # NOTE: template is frozen post-init (extract_template), one search patch
        # per frame -> N is always 1. groups=N reduces to plain conv2d here.
        # This does NOT support N>1 (multiple simultaneous search patches against
        # one template) without first tiling r1_kernel/cls1_kernel per group.
        N, C, H, W = r2_out.shape
        r2_out = r2_out.view(1, N * C, H, W)
        cls2_out = cls2_out.view(1, N * C, H, W)

        # 3. Fused group cross-correlation
        regress_raw = F.conv2d(r2_out, r1_kernel, groups=N)
        cls_raw = F.conv2d(cls2_out, cls1_kernel, groups=N)

        # 4. Final format restoration
        regress_channels = r1_kernel.shape[0]
        cls_channels = cls1_kernel.shape[0]

        regress = self.regress_adjust(regress_raw.view(N, regress_channels, regress_raw.shape[-2], regress_raw.shape[-1]))
        cls = cls_raw.view(N, cls_channels, cls_raw.shape[-2], cls_raw.shape[-1])

        return regress, cls

class SiamRPNBIG(SiamRPN):
    cfg = {
        "lr": 0.295,
        "window_influence": 0.42,
        "penalty_k": 0.055,
        "instance_size": 271,
        "adaptive": True,
    }

    def __init__(self):
        super().__init__(size=2)


class SiamRPNotb(SiamRPN):
    cfg = {
        "lr": 0.30,
        "window_influence": 0.40,
        "penalty_k": 0.22,
        "instance_size": 271,
        "adaptive": False,
    }

    def __init__(self):
        super().__init__(size=1, feature_out=256)