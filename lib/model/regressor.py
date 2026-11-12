from typing import List

import torch
import torch.nn as nn
from lib.model.utils import Sine


class Regressor(nn.Module):
    """Point-wise MLP predicting the sub-pixel offset (in [-1, 1]) to add to the
    integer disparity selected by the classifier."""

    def __init__(self, filter_channels: List[int]):
        super().__init__()
        self.nfeat = filter_channels[0]
        self.layer_1 = nn.Sequential(nn.Conv1d(self.nfeat, filter_channels[1], 1), Sine())
        self.layer_2 = nn.Sequential(
            nn.Conv1d(self.nfeat + filter_channels[1], filter_channels[2], 1), Sine()
        )
        self.layer_3 = nn.Conv1d(self.nfeat + filter_channels[2], 1, 1)

    def forward(self, feature: torch.Tensor):
        feat = self.layer_1(feature)
        feat = self.layer_2(torch.cat([feat, feature], 1))
        feat = self.layer_3(torch.cat([feat, feature], 1))
        return torch.tanh(feat)
