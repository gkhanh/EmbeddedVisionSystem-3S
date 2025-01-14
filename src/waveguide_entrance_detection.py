import cv2
import numpy as np
from src.utils import filter_lines_within_boundary
from src.visualizer import Visualizer
from src.edge_detection import EdgeDetector
import matplotlib.pyplot as plt


class WaveguideEntranceDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.edge_detector = EdgeDetector(image_path)
        self.image = self.edge_detector.image

    def detect_edges(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # # Calculate the median of the pixel intensities
        # median_val = np.median(blurred)
        #
        # # Use automatic Canny edge detection based on median value
        # low_threshold = max(0, int((1.0 - 0.4) * median_val))
        # high_threshold = min(255, int((1.0 + 0.5) * median_val))

        # Use Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 150)
        # edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    # def filter_edges(self, edges, orientation='vertical', threshold=50):
    #     # Define a vertical and horizontal kernel to isolate edges in the desired direction
    #     if orientation == 'vertical':
    #         kernel = np.array([[1], [-1]])  # Vertical filter
    #     else:
    #         kernel = np.array([[1, -1]])  # Horizontal filter
    #
    #     filtered_edges = cv2.filter2D(edges, -1, kernel)
    #     _, binary_edges = cv2.threshold(filtered_edges, threshold, 255, cv2.THRESH_BINARY)
    #
    #     return binary_edges

    def detect_hough_lines(self, edges):
        # Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=30)

        return lines

    def detect_waveguide_entrance(self):
        # Step 1: Detect edges
        edges = self.detect_edges()

        # Step 2: Detect Hough lines
        lines = self.detect_hough_lines(edges)

        # Step 3: Filter vertical and horizontal lines
        vertical_lines = []
        horizontal_lines = []

        if lines is not None:
            vertical_lines = filter_lines_within_boundary(lines, axis='vertical', min_length=100, margin=10)
            horizontal_lines = filter_lines_within_boundary(lines, axis='horizontal', min_length=100, margin=10)

        # Step 4: Find the waveguide entrance (intersection of horizontal and vertical lines)
        waveguide_entrance_line = None # Placeholder for detected line
        waveguide_entrance_point = None # Placeholder for entrance point

        if horizontal_lines and vertical_lines:
            # Get the right-most vertical line
            right_vertical_line = sorted(vertical_lines, key=lambda line: line[0])[-1]

            # Get the top-most horizontal line
            top_horizontal_line = sorted(horizontal_lines, key=lambda line: line[1])[0]

            # The entrance point is the intersection of the top horizontal line and the right vertical line
            waveguide_entrance_point = (right_vertical_line[0], top_horizontal_line[1])

            # Create the waveguide entrance line extending from the entrance point downwards
            waveguide_entrance_line = np.array([waveguide_entrance_point[0], waveguide_entrance_point[1],
                                                waveguide_entrance_point[0], self.image.shape[0]])

        return waveguide_entrance_line, waveguide_entrance_point

    def visualize_waveguide_entrance(self):
        # Step 1: Detect the waveguide entrance line and point
        waveguide_entrance_line, waveguide_entrance_point = self.detect_waveguide_entrance()

        # Step 2: Visualize the results using the Visualizer class
        visualizer = Visualizer(self.image)
        visualizer.plot_waveguide_entrance(waveguide_entrance_line, waveguide_entrance_point)

    # def visualize_intermediate_results(self, vertical_edges, horizontal_edges):
    #     # Plot vertical and horizontal edges separately
    #     fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    #
    #     axs[0].imshow(vertical_edges, cmap='gray')
    #     axs[0].set_title('Vertical Edges')
    #
    #     axs[1].imshow(horizontal_edges, cmap='gray')
    #     axs[1].set_title('Horizontal Edges')
    #
    #     plt.show()
