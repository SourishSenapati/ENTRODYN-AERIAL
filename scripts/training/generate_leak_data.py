import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def generate_gaussian_plume(grid_size=100, source=(50, 50), wind=(1.0, 0.5)):
    """
    Generates a synthetic gas plume for testing the PIKAN tracker.
    Uses Gaussian dispersion model.
    """
    x = np.linspace(0, 100, grid_size)
    y = np.linspace(0, 100, grid_size)
    X, Y = np.meshgrid(x, y)

    # Advection-Diffusion Approximation
    dist_x = X - source[0]
    dist_y = Y - source[1]

    # Rotating for wind direction
    concentration = np.exp(-((dist_x - wind[0])
                           ** 2 + (dist_y - wind[1])**2) / 200.0)
    return concentration


if __name__ == "__main__":
    print("Generating synthetic gas leak scenarios...")
    data = generate_gaussian_plume()

    os.makedirs("src/simulation_data", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    np.save("src/simulation_data/leak_scenario_01.npy", data)
    print("Saved scenario_01.npy")

    # Quick debug plot
    plt.imshow(data, cmap='hot')
    plt.title("Synthetic Leak Source (Ground Truth)")
    plt.savefig("docs/leak_vis_debug.png")
