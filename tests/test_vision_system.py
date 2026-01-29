"""
Unit tests for the Smart Vision System (Computer Vision).
Verifies that YOLOv8 loads and processes frames correctly without a live camera.
"""

from core_logic.smart_eye import SmartEye
import unittest
import numpy as np
import cv2
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))


class TestVisionSystem(unittest.TestCase):
    def setUp(self):
        print(
            "\n[Vision Test] Initializing YOLOv8 Model (this may take time on first run)...")
        self.eye = SmartEye()

    def test_model_loading(self):
        """Verify the model is loaded correctly."""
        self.assertIsNotNone(self.eye.model, "YOLO model failed to load.")
        print(">> Model Loaded Successfully.")

    def test_empty_frame_handling(self):
        """Test how the system handles a black frame (no objects)."""
        # Create a black 640x480 image
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        processed_frame, avoid_vector, obstacle_detected = self.eye.scan_frame(
            black_frame)

        self.assertFalse(obstacle_detected,
                         "Phantom obstacle detection on black frame.")
        self.assertEqual(avoid_vector, 0.0,
                         "Steering command generated for empty space.")
        print(">> Empty Frame Check: PASSED")

    def test_synthetic_obstacle(self):
        """
        Since we can't easily fake a 'person' for YOLO without downloading a real image,
        we verify the pipeline runs without crashing on random noise.
        """
        noise_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            _, _, _ = self.eye.scan_frame(noise_frame)
            print(">> Inference Pipeline Stability: PASSED")
        except Exception as e:
            self.fail(f"Vision pipeline crashed on noise frame: {e}")


if __name__ == '__main__':
    unittest.main()
