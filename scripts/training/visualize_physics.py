"""
Physics Visualization Script.
Generates plots for KAN activations and Entropy fields.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn.functional as F

# 1. Setup Path for imports (CRITICAL: Must be first)
# Add src directory to system path so we can import pikan_core
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: sys.path[0] is {sys.path[0]}")

# 2. Import Custom Modules
try:
    from pikan_core.kan_layers import KANSplineLayer
    from pikan_core.thermo_physics import calculate_entropy_production
    print("DEBUG: Successfully imported pikan_core modules")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import pikan_core. Path issue? {e}")
    sys.exit(1)

# 3. GPU Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEBUG: Using Computation Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"DEBUG: GPU Model: {torch.cuda.get_device_name(0)}")


def plot_kan_basis(grid_size=5, spline_order=3):
    """Visualizes the B-Spline basis functions used in the KAN layer."""
    print("Generating KAN Basis visualization...")

    # Create x range for plotting
    x = torch.linspace(-1.5, 1.5, 300).to(DEVICE)

    plt.figure(figsize=(10, 6))
    plt.title(f"KAN Layer: B-Spline Basis Functions (Order={spline_order})")
    plt.xlabel("Input Value (Normalized)")
    plt.ylabel("Basis Activation")

    # Instantiate layer and move to GPU
    layer = KANSplineLayer(in_features=1, out_features=1,
                           grid_size=grid_size, spline_order=spline_order).to(DEVICE)

    # Visualize the output of the layer
    with torch.no_grad():
        x_in = x.unsqueeze(1)  # [300, 1]
        y = layer(x_in)
        base_out = F.linear(
            x_in, layer.base_weight)  # pylint: disable=not-callable

    # Move back to CPU for plotting
    x_cpu = x.cpu().numpy()
    y_cpu = y.cpu().numpy()
    base_cpu = base_out.cpu().numpy()

    plt.plot(x_cpu, y_cpu, label="KAN Output (Random Init)",
             linewidth=2.5, color='#00aaff')
    plt.plot(x_cpu, base_cpu, '--', label="Base Linear Component",
             color='gray', alpha=0.5)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Ensure docs directory exists
    docs_dir = os.path.abspath(os.path.join(current_dir, '../../docs'))
    os.makedirs(docs_dir, exist_ok=True)
    save_path = os.path.join(docs_dir, "viz_kan_activation.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


def plot_entropy_field():
    """Visualizes the Entropy Production Rate on synthetic data."""
    print("Generating Entropy Field visualization...")

    # 1. Load Synthetic Data (or generate on fly)
    grid_size = 100
    x = np.linspace(0, 100, grid_size)
    y = np.linspace(0, 100, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    source = (50, 50)
    concentration_np = np.exp(-((grid_x - source[0])
                              ** 2 + (grid_y - source[1])**2) / 200.0)

    # Move to GPU for calculation
    c_field = torch.tensor(concentration_np, dtype=torch.float32,
                           # [1, 1, 100, 100]
                           device=DEVICE).unsqueeze(0).unsqueeze(0)
    u_field = torch.zeros(1, 2, 100, 100, device=DEVICE)  # Zero velocity

    s_gen = calculate_entropy_production(c_field, u_field)

    # Plotting (CPU)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Fix unused variable warning by using fig (e.g. suptitle) or ignoring
    fig.suptitle("Thermodynamic Entropy Analysis")

    # Concentration
    im1 = axes[0].imshow(concentration_np, cmap='viridis', origin='lower')
    axes[0].set_title("Gas Concentration Field (C)")
    plt.colorbar(im1, ax=axes[0])

    # Entropy Production
    s_gen_np = s_gen.squeeze().cpu().numpy()
    im2 = axes[1].imshow(s_gen_np, cmap='inferno', origin='lower')
    axes[1].set_title(
        "Entropy Production Rate (S_gen)\n(The 'Scent' the drone follows)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    docs_dir = os.path.abspath(os.path.join(current_dir, '../../docs'))
    os.makedirs(docs_dir, exist_ok=True)
    save_path = os.path.join(docs_dir, "viz_entropy_field.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_kan_basis()
    plot_entropy_field()
