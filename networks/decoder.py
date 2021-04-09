import torch
import torch.nn as nn

from .blocks import ConvBlock
from .blocks import UpConv


class Decoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 filters: list,
                 ) -> None:
        super().__init__()

        self.initial_conv = nn.Conv2d(emb_dim, filters[0], 3, 1, 1)
        self.dec_block_0 = ConvBlock(filters[0], filters[0])

        self.len_filters = len(filters)

        for i in range(len(filters) - 1):
            in_channels = filters[i]
            out_channels = filters[i + 1]

            self.add_module('dec_up_{}'.format(str(i + 1)), UpConv(in_channels, out_channels))
            self.add_module('dec_block_{}'.format(str(i + 1)), ConvBlock(out_channels, out_channels))

        self.dec_end = nn.Conv2d(filters[-1], output_dim, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.dec_block_0(x)

        for i in range(self.len_filters - 1):
            x = getattr(self, 'dec_up_{}'.format(str(i + 1)))(x)
            x = getattr(self, 'dec_block_{}'.format(str(i + 1)))(x)

        x = self.dec_end(x)

        return x
