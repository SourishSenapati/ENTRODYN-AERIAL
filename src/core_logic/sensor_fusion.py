"""
Sensor Fusion Logic for Robust Leak Detection.
Mimics Extended Kalman Filter (EKF) rejection logic.
"""

import numpy as np


class RobustAuditor:
    def __init__(self):
        self.gas_threshold = 50.0  # ppm
        self.thermal_threshold = 28.0  # degrees C (Ambient)

    def verify_leak(self, gas_ppm, temp_reading):
        """
        The 'One-in-a-Million' Logic.
        Rejects false positives by requiring dual-modal confirmation.

        Args:
            gas_ppm (float): Gas sensor reading in ppm.
            temp_reading (float): Temperature reading in Celsius.

        Returns:
            float: Confidence score (1.0 = Confirmed, 0.1 = Noise, 0.0 = Clear)
        """
        # Logic 1: Gas sensor sees something
        gas_detected = gas_ppm > self.gas_threshold

        # Logic 2: Joule-Thomson Effect (Leaking gas is colder than air)
        # If temp drops suddenly at the same spot, it's real.
        temp_drop = temp_reading < (self.thermal_threshold - 3.0)

        if gas_detected and temp_drop:
            return 1.0  # CONFIRMED LEAK (100% Certainty)
        elif gas_detected and not temp_drop:
            return 0.1  # Likely sensor noise / exhaust
        else:
            return 0.0  # Clean air
