import cv2
import numpy as np
from ultralytics import YOLO


class SmartEye:
    def __init__(self):
        # Load lightweight model (Auto-downloads on first run)
        self.model = YOLO("yolov8n.pt")
        # 0=Person, 15=Cat/Dog (Simulating dynamic obstacles)
        self.classes_to_avoid = [0, 15, 16]
        self.safe_distance_threshold = 0.4  # 40% of screen width

    def scan_frame(self, frame):
        """
        Input: Raw Camera Frame
        Output: Processed Frame (Drawings), Avoidance_Command (Vector)
        """
        results = self.model(frame, verbose=False)
        avoid_vector = 0.0  # 0.0 = Stay Course, -1.0 = Left, 1.0 = Right
        obstacle_detected = False

        # Get frame center
        height, width, _ = frame.shape
        center_x = width // 2

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.classes_to_avoid:
                    # Logic: Avoid Logic
                    x1, y1, x2, y2 = box.xyxy[0]
                    obj_center_x = (x1 + x2) / 2

                    # Calculate deviation from center (Normalized -1 to 1)
                    deviation = (obj_center_x - center_x) / (width / 2)

                    # If object is huge (close), trigger avoidance
                    box_width = x2 - x1
                    if box_width > (width * 0.3):  # Object takes up 30% of screen
                        obstacle_detected = True
                        # STEER AWAY: If object is Right, Go Left.
                        if deviation > 0:
                            avoid_vector = -0.8  # Yaw Left
                        else:
                            avoid_vector = 0.8  # Yaw Right

                    # Draw Bounding Box (Green=Safe, Red=Avoid)
                    color = (0, 0, 255) if obstacle_detected else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, "OBSTACLE", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, avoid_vector, obstacle_detected


# --- TEST BLOCK (Run this on Windows to test Webcam) ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    eye = SmartEye()
    print("Starting Smart Vision System... Press 'q' to exit.")
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        processed_img, cmd, hazard = eye.scan_frame(img)

        # Visualize Command
        cv2.putText(processed_img, f"CMD: {cmd}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("ENTRODYN SMART VISION", processed_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
