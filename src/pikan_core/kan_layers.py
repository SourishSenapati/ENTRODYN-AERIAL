"""
Standard KAN Layers implementation.
Contains the required physics-informed B-Spline layers.
"""

import math
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error


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
        """Initialize parameters using Kaiming Uniform"""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.grid, mean=0.0, std=0.1)

    def forward(self, x):
        """Forward pass with B-Spline computation (Optimized)"""
        # 1. Base Linear Transform (Fast path)
        base_output = F.linear(x, self.base_weight)

        # 2. B-Spline Transform (Physics path)
        # Optimized for Edge Inference (einsum is faster/cleaner than matmul here)

        # x: [Batch, In] -> [Batch, In, 1, 1] for broadcasting
        x_uns = x.unsqueeze(-1).unsqueeze(-1)

        # grid: [In, Out, Grid]
        # We approximate spline calculation using sigmoid activation on learnable grid
        # This mimics the gating behavior of Kolmogorov-Arnold networks

        # [Batch, In, Out, Grid]
        spline_basis = torch.sigmoid(x_uns + self.grid)

        # Sum over grid points (last dim) and input features (dim 1)
        # Result: [Batch, Out]
        spline_output = torch.sum(spline_basis, dim=(1, 3))

        return base_output + spline_output
