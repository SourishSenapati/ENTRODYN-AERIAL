import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANSplineLayer(nn.Module):
    """
    Hardware-Optimized Kolmogorov-Arnold Layer.
    Uses B-Spline basis functions instead of standard weights.
    Grid Size: 5 (Low res for edge speed) | Spline Order: 3 (Cubic)
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANSplineLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Learnable Control Points (The "Weights")
        self.grid = nn.Parameter(torch.Tensor(
            in_features, out_features, grid_size + spline_order))
        # Base Linear Residual (for stability)
        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.grid, mean=0.0, std=0.1)

    def forward(self, x):
        # 1. Base Linear Transform (Fast path)
        base_output = F.linear(x, self.base_weight)

        # 2. B-Spline Transform (The "Physics" path)
        # TODO: Optimize this matmul for Jetson Orin Nano
        x_uns = x.unsqueeze(-1)  # [Batch, In, 1]
        # Simplified Basis Function (Gaussian approximation for speed)
        basis = torch.exp(-torch.pow(x_uns - self.grid.mean(), 2))
        spline_output = torch.matmul(basis, self.grid).sum(dim=1)

        return base_output + spline_output
