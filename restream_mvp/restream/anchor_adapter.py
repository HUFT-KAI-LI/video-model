import torch
from torch import nn


class GatedAnchorAdapter(nn.Module):
    """Public tensors use LongLive's B,T,C,H,W layout."""

    def __init__(self, channels=16, gate_init_logit=-2.0):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(nn.Conv3d(2 * channels, channels, 1), nn.SiLU(),
                                 nn.Conv3d(channels, channels, 1))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit)))

    def forward(self, predicted, anchor):
        if predicted.ndim != 5 or predicted.shape != anchor.shape or predicted.shape[2] != self.channels:
            raise ValueError("Expected matching B,T,C,H,W tensors")
        x = torch.cat((predicted, anchor), dim=2).permute(0, 2, 1, 3, 4)
        delta = self.net(x).permute(0, 2, 1, 3, 4)
        return predicted + self.gate_logit.sigmoid() * delta
