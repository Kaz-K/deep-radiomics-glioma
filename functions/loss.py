import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    c = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order).contiguous()
    return transposed.view(c, -1)


class SoftDiceLoss(nn.Module):

    def __init__(self, ignore_index: bool = None, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = torch.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target).float()

        intersect = (output * target).sum(-1)
        denominator = output.sum(-1) + target.sum(-1)

        if self.ignore_index is not None:
            intersect = torch.cat((intersect[0:self.ignore_index], intersect[self.ignore_index + 1:]))
            denominator = torch.cat((denominator[0:self.ignore_index], denominator[self.ignore_index + 1:]))

        intersect = intersect.sum()
        denominator = denominator.sum()
        dice = 2.0 * intersect / denominator.clamp(min=self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    epsilon = 1e-6

    def __init__(self, gamma: float = 2, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # output: (B, C, H, W)
        # target: (B, C, H, W)
        p = torch.softmax(output, dim=1)
        p = torch.clamp(p, min=self.epsilon, max=1-self.epsilon)
        log_p = torch.log_softmax(output, dim=1)

        loss_sce = - target * log_p
        loss_focal = torch.sum(loss_sce * (1.0 - p) ** self.gamma, dim=1)

        return loss_focal.mean()


class OneHotEncoder(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()

        self.n_classes = n_classes
        self.ones = torch.sparse.torch.eye(n_classes).cuda()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n_dim = t.dim()
        output_size = t.size() + torch.Size([self.n_classes])

        t = t.data.long().contiguous().view(-1).cuda()
        out = Variable(self.ones.index_select(0, t)).view(output_size)
        out = out.permute(0, -1, *range(1, n_dim)).contiguous().float()

        return out
