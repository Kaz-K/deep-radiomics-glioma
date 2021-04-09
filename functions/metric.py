import torch

from .loss import OneHotEncoder


class DiceCoefficient(object):
    epsilon = 1e-5

    def __init__(self, n_classes: int, index_to_class_name: dict, ignore_index: bool = None):
        super().__init__()
        self.one_hot_encoder = OneHotEncoder(n_classes).forward
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.index_to_class_name = index_to_class_name

    def __call__(self, pred: torch.Tensor, label: torch.Tensor, allow_none: bool = False) -> dict:
        batch_size = pred.shape[0]
        output = pred.argmax(1)

        output = self.one_hot_encoder(output)
        output = output.contiguous().view(batch_size, self.n_classes, -1)

        target = self.one_hot_encoder(label)
        target = target.contiguous().view(batch_size, self.n_classes, -1)

        assert output.shape == target.shape

        if not allow_none:
            dice = {}
            for i in range(self.n_classes):
                if i == self.ignore_index:
                    continue

                os = output[:, i, ...]
                ts = target[:, i, ...]

                inter = torch.sum(os * ts, dim=1)
                union = torch.sum(os, dim=1) + torch.sum(ts, dim=1)
                score = torch.sum(2.0 * inter / union.clamp(min=self.epsilon))
                score /= batch_size

                if self.index_to_class_name:
                    dice[self.index_to_class_name[i]] = score.item()
                else:
                    dice[str(i)] = score.item()

            return dice

        else:
            dice = {}
            for i in range(self.n_classes):
                if i == self.ignore_index:
                    continue

                os = output[:, i, ...]
                ts = target[:, i, ...]

                if ts.sum() == 0:
                    score = None

                else:
                    inter = torch.sum(os * ts, dim=1)
                    union = torch.sum(os, dim=1) + torch.sum(ts, dim=1)
                    score = torch.sum(2.0 * inter / union.clamp(min=self.epsilon))
                    score /= batch_size
                    score = score.item()

                if self.index_to_class_name:
                    dice[self.index_to_class_name[i]] = score
                else:
                    dice[str(i)] = score

            return dice
