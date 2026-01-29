"""
Test suite for Six Sigma Reliability Guard.
Verifies that random noise is rejected and true leaks are confirmed.
"""
import sys
import os
import unittest
from reliability.six_sigma_guard import SixSigmaGuard

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)


class TestSixSigmaReliability(unittest.TestCase):
    """
    Unit tests for the Six Sigma Reliability Guard.
    Ensures that the Bayesian Filter correctly distinguishes between noise and true leaks.
    """

    def setUp(self):
        self.guard = SixSigmaGuard()

    def test_noise_rejection(self):
        """Test 1: Random Noise (Simulating Sensor Glitch)"""
        print("\nTEST 1: Random Noise (Simulating Sensor Glitch)")
        triggered = False
        for i in range(10):
            # Gas flashes high for 1 frame then drops
            gas = 100.0 if i == 5 else 10.0
            temp = 28.0
            leak, conf = self.guard.update(gas, temp, 0.1)
            print(f"Time {i}: Conf={conf:.6f} | Alarm={leak}")
            if leak:
                triggered = True

        self.assertFalse(
            triggered, "Alarm triggered on random noise (False Positive)")

    def test_true_leak_detection(self):
        """Test 2: True Leak (Simulating 1-second exposure)"""
        print("\nTEST 2: True Leak (Simulating exposure)")
        confirmed = False
        for i in range(20):
            # Gas High + Temp Low (Consistent)
            gas = 100.0
            temp = 24.0
            leak, conf = self.guard.update(gas, temp, 0.1)
            status = "ALARM!!!" if leak else "Analyzing..."
            print(f"Time {i}: Conf={conf:.6f} | {status}")
            if leak:
                confirmed = True
                break

        self.assertTrue(
            confirmed, "Alarm failed to trigger on true leak (False Negative)")


if __name__ == '__main__':
    unittest.main()
