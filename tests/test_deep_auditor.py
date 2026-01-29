"""
Test suite for DeepAuditor.
Verifies sensor saturation logic, material context analysis, and complex leak detection.
"""
import sys
import os
import unittest
import time
from core_logic.deep_auditor import DeepAuditor

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)


class TestDeepAuditor(unittest.TestCase):
    """
    Test suite for the DeepAuditor 'Judge' logic.
    Verifies sensor saturation handling, material context switching, and complex leak detection.
    """

    def setUp(self):
        self.auditor = DeepAuditor()

    def test_sensor_saturation_logic(self):
        """Test Failure Mode 1: Saturation Trap"""
        print("\nTEST: Sensor Saturation Logic")
        # 1. Normal reading
        status = self.auditor.check_sensor_health(0.5)
        self.assertEqual(status, "HEALTHY")

        # 2. Saturation starts
        status = self.auditor.check_sensor_health(0.95)
        self.assertEqual(status, "HEALTHY")  # Not timed out yet

        # 3. Fast forward time (simulate 11 seconds)
        self.auditor.saturation_timer_start -= 11.0
        status = self.auditor.check_sensor_health(0.95)
        self.assertEqual(status, "SATURATED_RESET_REQUIRED")
        print(">> Saturation Reset Triggered successfully")

    def test_material_context(self):
        """Test Failure Mode 3: Shiny Pipe Problem"""
        print("\nTEST: Material Context Analysis")
        # Class 2 = Shiny Metal
        mode = self.auditor.analyze_material_context(2)
        self.assertEqual(mode, "HIGH_REFLECTIVITY_MODE")
        self.assertTrue(self.auditor.thermal_reflection_risk)

        # Class 0 = Rust
        mode = self.auditor.analyze_material_context(0)
        self.assertEqual(mode, "NORMAL_MODE")
        self.assertFalse(self.auditor.thermal_reflection_risk)

    def test_complex_leak_verification(self):
        """Test Failure Mode 2: Complex Leak Scenarios"""
        print("\nTEST: Complex Leak Verification")

        # Scenario A: High Pressure (Gas High, Temp Low, Normal Surface)
        self.auditor.thermal_reflection_risk = False
        leak, msg = self.auditor.verify_leak_complex(0.6, 20.0, False)
        self.assertTrue(leak)
        self.assertIn("TYPE-A", msg)

        # Scenario B: Low Pressure / Rust (Gas High, Temp Normal, Rust Present)
        leak, msg = self.auditor.verify_leak_complex(0.6, 28.0, True)
        self.assertTrue(leak)
        self.assertIn("TYPE-B", msg)

        # Scenario C: Shiny Pipe (Gas very High, Temp Normal, Shiny)
        self.auditor.thermal_reflection_risk = True
        leak, msg = self.auditor.verify_leak_complex(0.85, 28.0, False)
        self.assertTrue(leak)
        self.assertIn("TYPE-C", msg)

        print(">> All leak scenarios verified")


if __name__ == '__main__':
    unittest.main()
