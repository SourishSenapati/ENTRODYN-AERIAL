"""
Deep Auditor Module.
Implements core logic for sensor saturation checks, material context analysis,
and complex leak verification combining gas, thermal, and visual data.
"""
import time


class DeepAuditor:
    """
    The DeepAuditor 'Judge' that validates sensor health and leak probability
    using multi-modal data (Gas, Thermal, Vision).
    """

    def __init__(self):
        # Thresholds
        self.gas_saturation_threshold = 0.9  # 90% of sensor max
        self.saturation_time_limit = 10.0   # Seconds
        self.thermal_reflection_risk = False

        # State tracking
        self.saturation_timer_start = 0
        self.is_saturated = False
        self.last_clean_baseline = 0.0

    def check_sensor_health(self, gas_normalized):
        """
        FAILURE MODE 1: Saturation Trap
        Returns: 'HEALTHY' or 'SATURATED_RESET_REQUIRED'
        """
        current_time = time.time()

        if gas_normalized > self.gas_saturation_threshold:
            if self.saturation_timer_start == 0:
                self.saturation_timer_start = current_time

            elapsed = current_time - self.saturation_timer_start
            if elapsed > self.saturation_time_limit:
                return "SATURATED_RESET_REQUIRED"
        else:
            self.saturation_timer_start = 0  # Reset timer if dip occurs

        return "HEALTHY"

    def analyze_material_context(self, visual_class_id):
        """
        FAILURE MODE 3: Shiny Pipe Problem
        Input: YOLO Class ID (0=Rust, 1=Insulation, 2=ShinyMetal)
        """
        if visual_class_id == 2:  # Shiny Metal
            self.thermal_reflection_risk = True
            return "HIGH_REFLECTIVITY_MODE"
        else:
            self.thermal_reflection_risk = False
            return "NORMAL_MODE"

    def verify_leak_complex(self, gas_val, temp_val, visual_rust_detected):
        """
        FAILURE MODE 2: Low Pressure Leaks
        Combines Gas, Thermal, and Visual Rust data.
        """
        # Scenario A: High Pressure (Standard)
        if gas_val > 0.5 and temp_val < 25.0 and not self.thermal_reflection_risk:
            return True, "TYPE-A: HIGH PRESSURE LEAK (Confident)"

        # Scenario B: Low Pressure / Rusted (The Fallback)
        if gas_val > 0.5 and visual_rust_detected:
            return True, "TYPE-B: LOW PRESSURE/CORROSION LEAK (Visual Confirm)"

        # Scenario C: Shiny Pipe Masking
        if gas_val > 0.8 and self.thermal_reflection_risk:
            return True, "TYPE-C: REFLECTIVE ZONE LEAK (Thermal Ignored)"

        return False, "SAFE"
