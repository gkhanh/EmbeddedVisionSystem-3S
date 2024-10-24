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

    # def preprocess_image(self, gray):
    #     # Apply bilateral filter to reduce noise and preserve edges
    #     return cv2.bilateralFilter(gray, 9, 75, 75)

    def apply_roi(self, image, roi_coords=None):
        """
        Crop the image to a predefined region of interest (ROI).
        roi_coords: Tuple of (x_start, y_start, x_end, y_end) defining the ROI.
        """
        if roi_coords is None:
            height, width = image.shape[:2]
            roi_coords = (int(width * 0.1), int(height * 0.9), int(width * 0.9), int(height * 0.9))  # Central ROI
        x_start, y_start, x_end, y_end = roi_coords
        return image[y_start:y_end, x_start:x_end]

    # def apply_roi(self, image):
    #     # Select ROI to focus on areas where the chip boundary is likely to be.
    #     height, width = image.shape[:2]
    #     # These values might need tuning based on chip location
    #     vertical_roi = image[:, :int(width * 0.6)]  # Capture leftmost portion
    #     horizontal_roi = image[:int(height * 0.4), :]  # Capture topmost portion
    #     return vertical_roi, horizontal_roi

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

    # def detect_edges(self):
    #     gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #     height, width = gray.shape
    #
    #     # Regions of Interest (ROI)
    #     vertical_roi = gray[:, :int(width * 0.4)]  # For vertical edge detection
    #     horizontal_roi = gray[:int(height * 0.4), :]  # For horizontal edge detection
    #
    #     # Adaptive Canny thresholds based on image
    #     median_val = np.median(vertical_roi)
    #     lower_threshold = max(0, int((1.0 - 0.3) * median_val))
    #     upper_threshold = min(255, int((1.0 + 0.5) * median_val))
    #
    #     edges_vertical = cv2.Canny(vertical_roi, lower_threshold, upper_threshold)
    #     edges_horizontal = cv2.Canny(horizontal_roi, lower_threshold, upper_threshold)
    #
    #     return edges_vertical, edges_horizontal

    def detect_hough_lines(self, edges_vertical, edges_horizontal):
        # Adjusted Hough Line Transform parameters for better detection
        vertical_lines = cv2.HoughLinesP(edges_vertical, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100,
                                         maxLineGap=20)
        horizontal_lines = cv2.HoughLinesP(edges_horizontal, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100,
                                           maxLineGap=20)

        return vertical_lines, horizontal_lines