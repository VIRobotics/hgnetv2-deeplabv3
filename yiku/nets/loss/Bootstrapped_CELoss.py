import torch
import torch.nn as nn
import torch.nn.functional as F
class BootstrappedCrossEntropyLoss(nn.Module):
    """
    Implements the cross entropy loss function.

    Args:
        min_K (int): the minimum number of pixels to be counted in loss computation.
        loss_th (float): the loss threshold. Only loss that is larger than the threshold
            would be calculated.
        weight (tuple|list, optional): The weight for different classes. Default: None.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: 255.
    """

    def __init__(self, min_K, loss_th, weight=None, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.K = min_K
        self.threshold = loss_th
        if weight is not None:
            weight = torch.from_numpy(weight, dtype='float32')
        self.weight = weight

    def forward(self, logit, label):

        n, c, h, w = logit.shape
        total_loss = 0.0
        if len(label.shape) != len(logit.shape):
            label = torch.unsqueeze(label, 1)

        for i in range(n):
            x = torch.unsqueeze(logit[i], 0)
            y = torch.unsqueeze(label[i], 0)
            x = torch.transpose(x, (0, 2, 3, 1))
            y = torch.transpose(y, (0, 2, 3, 1))
            x = torch.reshape(x, shape=(-1, c))
            y = torch.reshape(y, shape=(-1, ))
            loss = F.cross_entropy(
                x,
                y,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction="none")
            sorted_loss = torch.sort(loss, descending=True)
            if sorted_loss[self.K] > self.threshold:
                new_indices = torch.nonzero(sorted_loss > self.threshold)
                loss = torch.gather(sorted_loss, new_indices)
            else:
                loss = sorted_loss[:self.K]

            total_loss += torch.mean(loss)
        return total_loss / float(n)