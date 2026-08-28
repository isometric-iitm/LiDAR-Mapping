import torch
import torch.nn as nn


class CBR(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1,
                 use_groupnorm: bool = True, groups: int = 8):
        super().__init__()
        norm = nn.GroupNorm(groups, out_ch) if use_groupnorm else nn.BatchNorm2d(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=False),
            norm,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 use_groupnorm: bool = True, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            CBR(channels, channels, kernel_size, use_groupnorm=use_groupnorm, groups=groups),
            CBR(channels, channels, kernel_size, use_groupnorm=use_groupnorm, groups=groups),
        )

    def forward(self, x):
        return x + self.block(x)


class RangeImageUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 4,
        base_channels: int = 32,
        use_groupnorm: bool = True,
        groups: int = 8,
    ):
        super().__init__()
        c = base_channels
        self.enc0 = CBR(in_channels, c, use_groupnorm=use_groupnorm, groups=groups)
        self.pool0 = nn.MaxPool2d(2, 2)

        self.enc1 = nn.Sequential(
            CBR(c, c * 2, use_groupnorm=use_groupnorm, groups=groups),
            ResBlock(c * 2, use_groupnorm=use_groupnorm, groups=groups),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            CBR(c * 2, c * 4, use_groupnorm=use_groupnorm, groups=groups),
            ResBlock(c * 4, use_groupnorm=use_groupnorm, groups=groups),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            CBR(c * 4, c * 8, use_groupnorm=use_groupnorm, groups=groups),
            ResBlock(c * 8, use_groupnorm=use_groupnorm, groups=groups),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            CBR(c * 8, c * 16, use_groupnorm=use_groupnorm, groups=groups),
            ResBlock(c * 16, use_groupnorm=use_groupnorm, groups=groups),
            CBR(c * 16, c * 16, dilation=2, padding=2, use_groupnorm=use_groupnorm, groups=groups),
        )

        self.up3 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2)
        self.dec3 = CBR(c * 16, c * 8, use_groupnorm=use_groupnorm, groups=groups)

        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec2 = CBR(c * 8, c * 4, use_groupnorm=use_groupnorm, groups=groups)

        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec1 = CBR(c * 4, c * 2, use_groupnorm=use_groupnorm, groups=groups)

        self.up0 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec0 = CBR(c * 2, c, use_groupnorm=use_groupnorm, groups=groups)

        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)                        # [B, c, H, W]
        p0 = self.pool0(e0)                       # [B, c, H/2, W/2]

        e1 = self.enc1(p0)                        # [B, c*2, H/2, W/2]
        p1 = self.pool1(e1)                       # [B, c*2, H/4, W/4]

        e2 = self.enc2(p1)                        # [B, c*4, H/4, W/4]
        p2 = self.pool2(e2)                       # [B, c*4, H/8, W/8]

        e3 = self.enc3(p2)                        # [B, c*8, H/8, W/8]
        p3 = self.pool3(e3)                       # [B, c*8, H/16, W/16]

        b = self.bottleneck(p3)                   # [B, c*16, H/16, W/16]

        d3 = self.up3(b)                          # [B, c*8, H/8, W/8]
        d3 = torch.cat([d3, e3], dim=1)            # [B, c*16, H/8, W/8]
        d3 = self.dec3(d3)                        # [B, c*8, H/8, W/8]

        d2 = self.up2(d3)                         # [B, c*4, H/4, W/4]
        d2 = torch.cat([d2, e2], dim=1)            # [B, c*8, H/4, W/4]
        d2 = self.dec2(d2)                        # [B, c*4, H/4, W/4]

        d1 = self.up1(d2)                         # [B, c*2, H/2, W/2]
        d1 = torch.cat([d1, e1], dim=1)            # [B, c*4, H/2, W/2]
        d1 = self.dec1(d1)                        # [B, c*2, H/2, W/2]

        d0 = self.up0(d1)                         # [B, c, H, W]
        d0 = torch.cat([d0, e0], dim=1)            # [B, c*2, H, W]
        d0 = self.dec0(d0)                        # [B, c, H, W]

        return self.head(d0)                       # [B, num_classes, H, W]
