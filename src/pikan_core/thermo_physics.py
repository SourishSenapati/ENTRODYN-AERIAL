import torch


def calculate_entropy_production(c_field, u_field, D=1.5e-5):
    """
    Computes Local Entropy Production Rate (S_gen).

    Args:
        c_field: Gas concentration tensor [Batch, 1, H, W]
        u_field: Velocity vector tensor [Batch, 2, H, W]
        D: Mass diffusivity (Default: Methane in Air)

    Returns:
        S_gen: Scalar field of thermodynamic irreversibility.
    """
    # 1. Gradient of Concentration (Nabla C)
    grad_c_x = torch.gradient(c_field, dim=2)[0]
    grad_c_y = torch.gradient(c_field, dim=3)[0]

    # 2. Magnitude of Gradient
    grad_mag_sq = grad_c_x**2 + grad_c_y**2

    # 3. Entropy Generation (Isothermal approx)
    # S_gen ~ (Flux^2) / Dissipation
    s_gen = grad_mag_sq / (D + 1e-8)  # Avoid div/0

    return s_gen
