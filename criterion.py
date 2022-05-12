import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(x, y, alpha=0.07, gamma=2,):
    batch = x.shape[0]
    x = torch.sigmoid(x)
    x = x.clamp(min=0.001, max = 0.999)
    x = x.reshape(-1)
    y = y.reshape(-1)
    positive = (y == 1)
    negative = (y == 0)

    p = x[positive]
    positive_loss = torch.sum(- (1 - p) ** gamma * torch.log(p))
    n = x[negative]
    negative_loss = torch.sum(-n ** gamma * torch.log(1 - n))

    # print(positive_loss, negative_loss)

    return (alpha * negative_loss + (1 - alpha) * positive_loss)/batch



