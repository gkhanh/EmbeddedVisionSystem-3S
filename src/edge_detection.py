import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image at {image_path} not found!")

    def preprocess_image(self, gray):
        # Apply Gaussian blur to reduce noise and improve edge detection
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def detect_edges(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Regions of Interest (ROI)
        vertical_roi = gray[:, :int(width * 0.4)]  # For vertical edge detection
        horizontal_roi = gray[:int(height * 0.4), :]  # For horizontal edge detection

        # Adjusted Canny thresholds for better edge detection
        edges_vertical = cv2.Canny(vertical_roi, 50, 150)  # Tuned for better sensitivity
        edges_horizontal = cv2.Canny(horizontal_roi, 50, 150)

        return edges_vertical, edges_horizontal

    def detect_hough_lines(self, edges_vertical, edges_horizontal):
        # Adjusted Hough Line Transform parameters for better detection
        vertical_lines = cv2.HoughLinesP(edges_vertical, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100,
                                         maxLineGap=20)
        horizontal_lines = cv2.HoughLinesP(edges_horizontal, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100,
                                           maxLineGap=20)

        return vertical_lines, horizontal_lines

