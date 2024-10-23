# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Function to filter lines based on axis and length
# def filter_lines_within_boundary(lines, axis='horizontal', min_length=150, margin=10):
#     filtered_lines = []
#     for line in lines:
#         x1, y1, x2, y2 = line
#         length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#         if axis == 'horizontal' and abs(y1 - y2) < margin and length >= min_length:
#             filtered_lines.append(line)
#         elif axis == 'vertical' and abs(x1 - x2) < margin and length >= min_length:
#             filtered_lines.append(line)
#     return filtered_lines
#
# # Perform linear regression to fit a line
# def fit_line_linear_regression(points, axis='horizontal'):
#     if len(points) == 0:
#         return None
#     if axis == 'horizontal':
#         x = np.array([p[0] for p in points])
#         y = np.array([p[1] for p in points])
#     else:
#         x = np.array([p[1] for p in points])
#         y = np.array([p[0] for p in points])
#     A = np.vstack([x, np.ones(len(x))]).T
#     m, b = np.linalg.lstsq(A, y, rcond=None)[0]
#     return m, b
#
# # Convert slope and intercept to a full line
# def get_line_from_regression(slope, intercept, image_shape, axis='horizontal', offset=0):
#     height, width = image_shape[:2]
#     if axis == 'horizontal':
#         x1, x2 = 0, width
#         y1 = int(slope * x1 + intercept)
#         y2 = int(slope * x2 + intercept)
#         return np.array([x1, y1, x2, y2], dtype=np.int32)
#     else:
#         y1, y2 = 0, height
#         x1 = int(slope * y1 + intercept) + offset
#         x2 = int(slope * y2 + intercept) + offset
#         return np.array([x1, y1, x2, y2], dtype=np.int32)
#
# # Load the image
# image_path = './data/foto2.png'  # Replace with actual image path
# image = cv2.imread(image_path)
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Adjust region of interest (ROI) to focus more on the chip's boundary
# height, width = gray.shape
# vertical_roi = gray[:, :int(width * 0.4)]  # Loosened the region for vertical edge detection
# horizontal_roi = gray[:int(height * 0.4), :]  # Loosened the region for horizontal edge detection
#
# # Use Canny Edge Detection to identify edges
# edges_vertical = cv2.Canny(vertical_roi, 50, 100)  # Lower thresholds for better edge detection
# edges_horizontal = cv2.Canny(horizontal_roi, 50, 100)
#
# # Use Hough Line Transform to detect lines
# vertical_lines = cv2.HoughLinesP(edges_vertical, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=30)
# horizontal_lines = cv2.HoughLinesP(edges_horizontal, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=30)
#
# # Process lines for vertical detection
# if vertical_lines is not None:
#     vertical_lines = [line[0] for line in vertical_lines if abs(line[0][2] - line[0][0]) < 20]  # Filter near-vertical lines
#     filtered_vertical_lines = filter_lines_within_boundary(vertical_lines, axis='vertical', min_length=200, margin=10)
# else:
#     filtered_vertical_lines = []
#
# # Process lines for horizontal detection
# if horizontal_lines is not None:
#     horizontal_lines = [line[0] for line in horizontal_lines if abs(line[0][3] - line[0][1]) < 20]  # Filter near-horizontal lines
#     filtered_horizontal_lines = filter_lines_within_boundary(horizontal_lines, axis='horizontal', min_length=200, margin=10)
# else:
#     filtered_horizontal_lines = []
#
# # Perform linear regression to fit lines
# if len(filtered_horizontal_lines) > 0:
#     horizontal_points = [(x1, y1) for x1, y1, x2, y2 in filtered_horizontal_lines]
#     slope_horizontal, intercept_horizontal = fit_line_linear_regression(horizontal_points, axis='horizontal')
#     middle_horizontal_line = get_line_from_regression(slope_horizontal, intercept_horizontal, image.shape, axis='horizontal')
# else:
#     middle_horizontal_line = None
#
# # Offset to move the vertical line slightly to the right
# vertical_offset = 20  # Adjust this value to fine-tune the rightward shift
#
# if len(filtered_vertical_lines) > 0:
#     vertical_points = [(x1, y1) for x1, y1, x2, y2 in filtered_vertical_lines]
#     slope_vertical, intercept_vertical = fit_line_linear_regression(vertical_points, axis='vertical')
#     middle_vertical_line = get_line_from_regression(slope_vertical, intercept_vertical, image.shape, axis='vertical', offset=vertical_offset)
# else:
#     middle_vertical_line = None
#
# # Plot the result using Matplotlib
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Detected Chip Boundary")
#
# # Draw the detected horizontal and vertical lines
# if middle_horizontal_line is not None:
#     x1, y1, x2, y2 = middle_horizontal_line
#     plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)  # Blue line for horizontal boundary
#
# if middle_vertical_line is not None:
#     x1, y1, x2, y2 = middle_vertical_line
#     plt.plot([x1, x2], [y1, y2], color='green', linewidth=2)  # Green line for vertical boundary
#
# plt.axis('on')  # Show coordinates
# plt.show()

from src.chip_boundary_detection import ChipBoundaryDetector

def main():
    image_path = './data/foto6.png'  # Path to the image
    boundary_detector = ChipBoundaryDetector(image_path)
    boundary_detector.visualize_chip_boundary()

if __name__ == "__main__":
    main()