import torch
import torch.nn as nn

from .blocks import ConvBlock
from .blocks import DownConv


class Encoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        self.init_conv = nn.Conv2d(input_dim, filters[0], 3, 1, 1, bias=True)
        self.enc_down_0 = DownConv(filters[0], filters[1])

        self.enc_block_1 = nn.Sequential(
            ConvBlock(filters[1], filters[1]),
        )
        self.enc_down_1 = DownConv(filters[1], filters[2])

        module_list = []
        for i in range(2, len(filters)):
            in_channels = filters[i]

            if i == len(filters) - 1:
                out_channels = emb_dim

                module_list.extend([
                    ConvBlock(in_channels, in_channels),
                    ConvBlock(in_channels, out_channels),
                ])

            else:
                out_channels = filters[i + 1]

                module_list.extend([
                    ConvBlock(in_channels, in_channels),
                    DownConv(in_channels, out_channels),
                ])

        self.modules = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.enc_down_0(x)

        x = self.enc_block_1(x)
        x = self.enc_down_1(x)

        out = self.modules(x)

        return out
