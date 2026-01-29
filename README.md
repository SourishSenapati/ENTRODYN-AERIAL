# ENTRODYN-AERIAL: Entropy-Gradient Navigation Swarm

**Status:** Prototype (Testing on RTX 4050)
**Build:** v0.1.2-alpha

## What is this?

This project forces a drone swarm to follow the **Second Law of Thermodynamics** to find gas leaks.

Standard sensors are too slow (20+ mins detection time). We wrote a custom **Kolmogorov-Arnold Network (KAN)** that calculates the _entropy production rate_ of the air. The drone follows the path of maximum disorder to find the leak source.

## Hardware Stack

- **Compute:** Laptop (RTX 4050) acting as Edge Station.
- **Drones:** Hexacopters with Pixhawk 4.
- **Sensors:** Thermal + MQ-135.

## Why Custom KAN?

Standard MLPs were too heavy for real-time physics solving. We switched to KANs (Splines).
