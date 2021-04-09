import torch
import torch.nn as nn

from .blocks import ConvBlock
from .blocks import DownConv
from .blocks import Normalize


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

        enc_normal_list = []
        enc_abnormal_list = []

        for i in range(2, len(filters)):
            in_channels = filters[i]

            if i == len(filters) - 1:
                out_channels = emb_dim

                enc_normal_list.extend([
                    ConvBlock(in_channels, in_channels),
                    ConvBlock(in_channels, out_channels),
                ])

                enc_abnormal_list.extend([
                    ConvBlock(in_channels, in_channels),
                    ConvBlock(in_channels, out_channels),
                ])

            else:
                out_channels = filters[i + 1]

                enc_normal_list.extend([
                    ConvBlock(in_channels, in_channels),
                    DownConv(in_channels, out_channels),
                ])

                enc_abnormal_list.extend([
                    ConvBlock(in_channels, in_channels),
                    DownConv(in_channels, out_channels),
                ])

        self.enc_normal = nn.Sequential(*enc_normal_list)
        self.enc_abnormal = nn.Sequential(*enc_abnormal_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.enc_down_0(x)

        x = self.enc_block_1(x)
        x = self.enc_down_1(x)

        out_1 = self.enc_normal(x)
        out_2 = self.enc_abnormal(x)

        return out_1, out_2
