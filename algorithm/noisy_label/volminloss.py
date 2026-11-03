import math

import torch
import torch.nn.functional as F
from torch import nn
from .losses import register_loss

@register_loss("volmin", requires_transition=False) # type: ignore[arg-type]
class VolMinLoss(nn.Module):
    def __init__(self, num_classes=3, reg=1e-4):
        super().__init__()
        C = num_classes
        self.w = nn.Parameter(torch.zeros((C, C)))  # zeros for C=3
        self.register_buffer("eye", torch.eye(C))
        self.reg = reg

    @property
    def T(self):
        T = self.eye + torch.sigmoid(self.w) * (1 - self.eye) # type: ignore[arg-type]
        return T / T.sum(dim=1, keepdim=True)

    def forward(self, logits, noisy_targets, transition_matrix):
        T = self.T
        noisy_post = F.softmax(logits, dim=1) @ T
        loss = F.nll_loss(torch.log(noisy_post + 1e-8), noisy_targets)
        if self.training:
            loss = loss + self.reg * torch.slogdet(T)[1] # type: ignore[arg-type]
        return loss