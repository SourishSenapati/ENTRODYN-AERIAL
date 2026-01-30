"""
Deep Auditor Module.
Implements core logic for sensor saturation checks, material context analysis,
and complex leak verification combining gas, thermal, and visual data.
"""
import time
import json
import os


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

        # Load previous state if rebooted recently
        self._load_state()

    def check_sensor_health(self, gas_normalized):
        """
        FAILURE MODE 1: Saturation Trap
        Returns: 'HEALTHY' or 'SATURATED_RESET_REQUIRED'
        """
        current_time = time.time()

        if gas_normalized > self.gas_saturation_threshold:
            if self.saturation_timer_start == 0:
                self.saturation_timer_start = current_time
                self._save_state()  # PERSISTENCE: Save start time

            elapsed = current_time - self.saturation_timer_start

            # Reset recovery if threshold exceeded again
            self.is_saturated = True

            if elapsed > self.saturation_time_limit:
                return "SATURATED_RESET_REQUIRED"
        else:
            if self.is_saturated:
                # Only reset if we were previously saturated
                self.saturation_timer_start = 0
                self.is_saturated = False
                self._save_state()  # PERSISTENCE: Clear state

        return "HEALTHY"

    def _save_state(self):
        """Persist saturation state to disk to survive brownouts/reboots."""
        state = {
            "saturation_timer_start": self.saturation_timer_start,
            "is_saturated": self.is_saturated,
            "timestamp": time.time()
        }
        try:
            with open("auditor_state.json", "w") as f:
                json.dump(state, f)
        except IOError:
            pass  # Creating a fail-safe

    def _load_state(self):
        """Recover state from disk."""
        if os.path.exists("auditor_state.json"):
            try:
                with open("auditor_state.json", "r") as f:
                    state = json.load(f)
                    # Only restore if recent (within 60 seconds)
                    if time.time() - state.get("timestamp", 0) < 60:
                        self.saturation_timer_start = state.get(
                            "saturation_timer_start", 0)
                        self.is_saturated = state.get("is_saturated", False)
            except (IOError, json.JSONDecodeError):
                pass

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
