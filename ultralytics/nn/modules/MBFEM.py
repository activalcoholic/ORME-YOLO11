import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """
    Basic Conv-BN-SiLU block.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)

        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """
    Depthwise separable convolution with optional dilation.
    It is used to enlarge the receptive field with low computational cost.
    """
    def __init__(self, c1, c2, k=3, s=1, dilation=1, act=True):
        super().__init__()
        padding = dilation * (k - 1) // 2

        self.dw = nn.Conv2d(
            c1,
            c1,
            kernel_size=k,
            stride=s,
            padding=padding,
            dilation=dilation,
            groups=c1,
            bias=False
        )
        self.dw_bn = nn.BatchNorm2d(c1)
        self.dw_act = nn.SiLU(inplace=True) if act else nn.Identity()

        self.pw = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.pw_act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


class MSStructureBranch(nn.Module):
    """
    Multi-scale structural branch.
    Different dilation rates are used to simulate different receptive fields.
    """
    def __init__(self, c1, c2, depth=1, kernel_size=3, dilation=1):
        super().__init__()

        layers = []
        for i in range(depth):
            in_ch = c1 if i == 0 else c2
            layers.append(
                DSConv(
                    in_ch,
                    c2,
                    k=kernel_size,
                    s=1,
                    dilation=dilation,
                    act=True
                )
            )

        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        return self.branch(x)


class MBFEM(nn.Module):
    """
    MBFEM feature enhancement module.

    Main ideas:
    1. Multi-scale structural modeling branch.
    2. Occlusion-aware spatial attention branch.
    3. Boundary/detail enhancement branch.
    4. Channel attention with exponential amplification.
    5. Residual re-calibration to avoid over-suppression.
    """

    def __init__(
        self,
        c1,
        c2=None,
        depth=1,
        kernel_size=3,
        patch_size=(2, 3, 4),
        reduction=16,
        spatial_kernel=7,
        edge_kernel=7,
        gamma_init=0.05
    ):
        super().__init__()

        # Keep output channel consistent with input by default.
        # This makes it safe to insert before Detect.
        if c2 is None or c2 != c1:
            c2 = c1

        self.c1 = c1
        self.c2 = c2

        if isinstance(patch_size, int):
            patch_size = (patch_size,)
        self.patch_size = patch_size

        # 1. Multi-scale structure branches.
        # Here patch_size is used as dilation rate to enlarge receptive field.
        self.ms_branches = nn.ModuleList([
            MSStructureBranch(
                c1=c1,
                c2=c2,
                depth=depth,
                kernel_size=kernel_size,
                dilation=p
            )
            for p in patch_size
        ])

        # Fuse multi-scale features.
        self.fuse = ConvBNAct(c2 * len(patch_size), c2, k=1, s=1)

        # 2. Channel attention branch.
        hidden = max(c2 // reduction, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 3. Boundary/detail enhancement branch.
        # Directional depthwise convolutions are useful for elongated fish contours.
        self.edge_branch = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=(1, edge_kernel),
                padding=(0, edge_kernel // 2),
                groups=c2,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                c2,
                c2,
                kernel_size=(edge_kernel, 1),
                padding=(edge_kernel // 2, 0),
                groups=c2,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

        # 4. Occlusion-aware spatial attention.
        # Input: average response map + max response map + edge response map.
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(
                3,
                1,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                bias=False
            ),
            nn.Sigmoid()
        )

        # Residual scaling coefficient.
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # If channel changes are needed in future, this keeps shortcut valid.
        self.shortcut = nn.Identity() if c1 == c2 else ConvBNAct(c1, c2, k=1, s=1)

    def forward(self, x):
        identity = self.shortcut(x)

        # Multi-scale structural features.
        ms_feats = [branch(x) for branch in self.ms_branches]
        fm = self.fuse(torch.cat(ms_feats, dim=1))

        # Channel attention.
        # Sigmoid output is in [0, 1], exp maps it to approximately [1, 2.718].
        # mc = torch.exp(self.channel_attn(fm))
        mc = 1.0 + self.channel_attn(fm)

        # Boundary/detail response.
        edge_feat = self.edge_branch(fm)

        # Spatial attention maps.
        avg_map = torch.mean(fm, dim=1, keepdim=True)
        max_map, _ = torch.max(fm, dim=1, keepdim=True)
        edge_map = torch.mean(edge_feat, dim=1, keepdim=True)

        ms = self.spatial_attn(torch.cat([avg_map, max_map, edge_map], dim=1))

        # Joint channel-spatial enhancement.
        enhanced = fm * mc * (1.0 +0.5 * ms)

        # Residual re-calibration.
        # This avoids destroying useful features when attention is imperfect.
        out = identity + self.gamma * (enhanced - identity)

        return out
