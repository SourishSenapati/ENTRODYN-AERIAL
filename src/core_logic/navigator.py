"""
Gradient Ascent Navigation Logic for Single Agent.
"""

import numpy as np


def calculate_next_waypoint(current_pos, gas_gradient, step_size=0.5):
    """
    Single-Agent Gradient Ascent.
    Move in the direction where smell increases most.

    Args:
        current_pos (np.array): Current [x, y, z] position.
        gas_gradient (np.array): Gradient vector of gas concentration.
        step_size (float): Movement step magnitude.

    Returns:
        np.array: Next waypoint coordinates.
    """
    # Normalize gradient
    norm = np.linalg.norm(gas_gradient)
    if norm < 1e-5:
        return current_pos  # Hover if lost

    direction = gas_gradient / norm

    # Calculate setpoint
    next_pos = current_pos + (direction * step_size)
    return next_pos
