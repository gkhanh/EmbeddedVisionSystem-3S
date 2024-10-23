# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the image
# image_path = '../data/foto2.png'  # Replace with actual image path
# image = cv2.imread(image_path)
#
# # Check if image was loaded correctly
# if image is None:
#     print("Error: Could not load image!")
# else:
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Adjust region of interest (ROI) to focus more on the chip's boundary
#     height, width = gray.shape
#     vertical_roi = gray[:, :int(width * 0.3)]  # Loosened the region for vertical edge detection
#     horizontal_roi = gray[:int(height * 0.3), :]  # Loosened the region for horizontal edge detection
#
#     # Use Canny Edge Detection to identify edges (adjust thresholds)
#     edges_vertical = cv2.Canny(vertical_roi, 30, 90)  # Lower thresholds for better detection
#     edges_horizontal = cv2.Canny(horizontal_roi, 30, 90)
#
#     # Diagnostic step 1: Show the detected edges
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(edges_vertical, cmap='gray')
#     plt.title('Edges (Vertical ROI)')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(edges_horizontal, cmap='gray')
#     plt.title('Edges (Horizontal ROI)')
#     plt.show()
#
#     # Use Hough Line Transform to detect lines (adjust parameters for better detection)
#     vertical_lines = cv2.HoughLinesP(edges_vertical, rho=1, theta=np.pi / 180, threshold=60, minLineLength=150, maxLineGap=40)
#     horizontal_lines = cv2.HoughLinesP(edges_horizontal, rho=1, theta=np.pi / 180, threshold=60, minLineLength=150, maxLineGap=40)
#
#     # Diagnostic step 2: Show all raw Hough lines
#     line_image = np.copy(image)  # Create a copy of the image to draw the raw lines
#
#     if vertical_lines is not None:
#         for line in vertical_lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for vertical lines
#
#     if horizontal_lines is not None:
#         for line in horizontal_lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for horizontal lines
#
#     # Plot the image with raw Hough lines
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
#     plt.title("Raw Detected Lines (Green for Vertical, Blue for Horizontal)")
#     plt.show()

import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image at {image_path} not found!")

    def detect_edges(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Regions of Interest (ROI)
        vertical_roi = gray[:, :int(width * 0.4)]  # For vertical edge detection
        horizontal_roi = gray[:int(height * 0.4), :]  # For horizontal edge detection

        # Edge detection using Canny
        edges_vertical = cv2.Canny(vertical_roi, 50, 100)
        edges_horizontal = cv2.Canny(horizontal_roi, 50, 100)

        return edges_vertical, edges_horizontal

    def detect_hough_lines(self, edges_vertical, edges_horizontal):
        # Hough Line Transform
        vertical_lines = cv2.HoughLinesP(edges_vertical, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=30)
        horizontal_lines = cv2.HoughLinesP(edges_horizontal, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=30)

        return vertical_lines, horizontal_lines

