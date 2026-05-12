import torch
import torch.nn as nn


class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels, expansion=4):
        super().__init__()

        hidden_dim = channels * expansion

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7,
                      padding=3, groups=channels),
            nn.GroupNorm(1, channels),
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            GRN(hidden_dim),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[ConvNeXtV2Block(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.down(x)
        return self.blocks(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=2):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels,
                      out_channels, kernel_size=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[ConvNeXtV2Block(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.blocks(x)


class ConvNeXtV2Generator(nn.Module):
    """
    ConvNeXt V2-inspired encoder-decoder generator with:
    - depthwise 7x7 ConvNeXt blocks
    - GRN
    - U-Net skip connections
    - tanh output for [-1, 1] image range
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(1, c),
            nn.GELU(),
            ConvNeXtV2Block(c),
        )

        self.down1 = DownBlock(c, c * 2, num_blocks=1)
        self.down2 = DownBlock(c * 2, c * 4, num_blocks=1)
        self.down3 = DownBlock(c * 4, c * 8, num_blocks=1)
        self.down4 = DownBlock(c * 8, c * 8, num_blocks=1)

        self.bottleneck = nn.Sequential(
            ConvNeXtV2Block(c * 8),
            ConvNeXtV2Block(c * 8),
        )

        self.up1 = UpBlock(c * 8, c * 8, c * 8, num_blocks=1)
        self.up2 = UpBlock(c * 8, c * 4, c * 4, num_blocks=1)
        self.up3 = UpBlock(c * 4, c * 2, c * 2, num_blocks=1)
        self.up4 = UpBlock(c * 2, c, c, num_blocks=1)

        self.out = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        s0 = self.stem(x)

        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)

        x = self.bottleneck(x)

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)

        return self.out(x)


if __name__ == "__main__":
    model = ConvNeXtV2Generator()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
