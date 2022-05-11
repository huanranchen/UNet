import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(x, y, alpha=0.25, gamma=2, eps=1e-5, scale = 10):
    x = F.sigmoid(x)
    x = x.reshape(-1)
    y = y.reshape(-1)
    positive = (y == 1)
    negative = (y == 0)

    p = x[positive]
    positive_loss = torch.sum(- (1 - p) ** gamma * torch.log(p + eps))
    n = x[negative]
    negative_loss = torch.sum(-n ** gamma * torch.log(1 - n + eps))

    return scale * (alpha * negative_loss + (1 - alpha) * positive_loss)/x.shape[0]
