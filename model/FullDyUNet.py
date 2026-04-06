"""FullDyUNet: Encoder + Decoder 모두 DyConvBlock 사용."""
import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop
from torchinfo import summary
import torch.nn.functional as F
import einops

from ._base import OutputBlock


class DyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, num_dy_conv=4, attn_temp=30):
        super().__init__()
        self.num_dy_conv = num_dy_conv
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.attn_temp = attn_temp

        if in_channels == 3:
            hidden_channels = num_dy_conv
        else:
            hidden_channels = int(in_channels * 0.25) + 1

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_dy_conv, kernel_size=1, bias=True)
        )

        self.weights = nn.Parameter(
            torch.randn(num_dy_conv, out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=True
        )
        self.gn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x_shape = einops.parse_shape(x, 'b c h w')
        batch_size, in_channels = x_shape['b'], x_shape['c']

        attn_scores = self.attention(x)
        attn_scores = attn_scores.view(batch_size, -1)
        attn_scores = F.softmax(attn_scores / self.attn_temp, 1)

        weights = self.weights.view(self.num_dy_conv, -1)
        filters = torch.mm(attn_scores, weights)
        filters = einops.rearrange(
            filters, 'b (out_c in_c kh kw) -> (b out_c) in_c kh kw',
            out_c=self.out_channels, in_c=in_channels, kh=self.kernel_size, kw=self.kernel_size
        )

        x = einops.rearrange(x, 'b c h w -> 1 (b c) h w')
        x = F.conv2d(x, filters, stride=self.stride, padding=self.padding, bias=None, groups=batch_size)
        x = einops.rearrange(x, '1 (b c) h w -> b c h w', b=batch_size)
        x = self.leaky_relu(self.gn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, size=6):
        super().__init__()
        self.encoder_layers = nn.ModuleList()

        for _ in range(size):
            self.encoder_layers.append(DyConvBlock(in_channels, out_channels, padding=padding, num_dy_conv=2))
            self.encoder_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.encoder_layers.append(DyConvBlock(in_channels, out_channels, padding=padding, num_dy_conv=2))

    def forward(self, x):
        route_connection = []
        for layer in self.encoder_layers:
            if isinstance(layer, DyConvBlock):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        return x, route_connection


class DecoderDyConv(nn.Module):
    """Decoder with DyConvBlock instead of MultiCNNBlock."""
    def __init__(self, in_channels, out_channels, num_class, padding, size=6, upsample=False):
        super().__init__()
        self.decoder_layers = nn.ModuleList()
        self.dy_blocks = nn.ModuleList()
        self.padding = padding

        for _ in range(size):
            if upsample:
                self.decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            else:
                self.decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            # DyConvBlock: 2*in_channels (concat) -> out_channels
            self.dy_blocks.append(DyConvBlock(2 * in_channels, out_channels, padding=padding, num_dy_conv=2))

        self.output = OutputBlock(out_channels, num_class)

    def forward(self, x, routes_connection):
        routes_connection.pop(-1)
        block_idx = 0
        for layer in self.decoder_layers:
            x = layer(x)
            if self.padding == 0:
                routes_connection[-1] = center_crop(routes_connection[-1], x.shape[-1])
            x = torch.cat([x, routes_connection.pop(-1)], dim=1)
            x = self.dy_blocks[block_idx](x)
            block_idx += 1
        x = self.output(x)
        return x


class FullDyUNet(nn.Module):
    """ThinDyUNet with DyConvBlock in both Encoder and Decoder."""
    def __init__(self, in_channels, start_out_channels, num_class, size, padding=0, upsample=False):
        super().__init__()
        self.encoder = Encoder(in_channels, start_out_channels, padding=padding, size=size)
        self.decoder = DecoderDyConv(
            start_out_channels, start_out_channels,
            num_class, padding=padding, size=size, upsample=upsample
        )

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    model = FullDyUNet(
        in_channels=3, start_out_channels=64,
        num_class=1, size=7, padding=1, upsample=True
    )
    summary(model, input_data=x, col_width=20, depth=5,
            row_settings=["depth", "var_names"],
            col_names=["input_size", "kernel_size", "output_size", "params_percent"])
