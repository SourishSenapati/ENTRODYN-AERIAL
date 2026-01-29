"""
PIKAN Math Core Model.
Physics-Informed Kolmogorov-Arnold Network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANSplineLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer (Spline-based).
    Optimized for Edge Inference (Jetson Orin/RTX 4050).
    """

    def __init__(self, in_features, out_features, grid_size=5):
        super(KANSplineLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable Spline Control Points
        self.grid = nn.Parameter(torch.randn(
            in_features, out_features, grid_size) * 0.1)
        self.base_weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.1)

    def forward(self, x):
        # Linear residual (Fast path)
        base = F.linear(x, self.base_weight)

        # B-Spline approximation (Physics path)
        # Simplified basis function for speed
        x_expanded = x.unsqueeze(-1)
        spline_basis = torch.sigmoid(
            x_expanded + self.grid)  # Activation function
        spline_out = torch.sum(spline_basis, dim=2)

        return base + spline_out


class EntropyNavNet(nn.Module):
    def __init__(self):
        super(EntropyNavNet, self).__init__()
        self.layer1 = KANSplineLayer(3, 16)  # Inputs: Gas, Temp, Wind
        self.layer2 = KANSplineLayer(16, 2)  # Outputs: Vector X, Vector Y

    def forward(self, x):
        x = F.silu(self.layer1(x))
        x = self.layer2(x)
        return x
