from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def focal_loss(preds: Tensor, target: Tensor, alpha: Tensor, gamma: float, reduction: str="mean") -> Tensor:
    # compute weighted cross entropy term: -alpha * log(pt) [alpha is already part of F.nll_loss]
    log_p = F.log_softmax(preds, dim=-1)
    neg_log_likelihood = F.nll_loss(log_p, target=target, weight=alpha)

    # get true class column from each row
    all_rows = torch.arange(len(preds))
    log_pt = log_p[all_rows, target]

    # compute focal term: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = torch.pow(1 - pt, gamma)

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * neg_log_likelihood

    if reduction is not None:
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

    return loss

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.

    Args:
        alpha (Tensor, optional): Weights for each class. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
    """

    def __init__(self, alpha: Optional[Tensor] = None, gamma: float = 0.0, reduction: str = 'mean'):
        if reduction not in ('mean', 'sum', None):
            raise ValueError(f"Reduction must be one of: 'mean', 'sum', f{None}.")

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return focal_loss(preds=preds, target=target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
