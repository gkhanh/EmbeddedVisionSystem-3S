from src.edge_detection import EdgeDetector
from src.utils import filter_lines_within_boundary, fit_line_linear_regression, get_line_from_regression
from src.visualizer import Visualizer

class ChipBoundaryDetector:
    def __init__(self, image_path):
        self.edge_detector = EdgeDetector(image_path)
        self.image = self.edge_detector.image

    def detect_chip_boundary(self):
        # Detect edges
        edges_vertical, edges_horizontal = self.edge_detector.detect_edges()

        # Detect Hough lines
        vertical_lines, horizontal_lines = self.edge_detector.detect_hough_lines(edges_vertical, edges_horizontal)

        # Debug information for detected lines
        print(f"Detected vertical lines: {len(vertical_lines) if vertical_lines is not None else 0}")
        print(f"Detected horizontal lines: {len(horizontal_lines) if horizontal_lines is not None else 0}")

        # Filter and process lines
        filtered_vertical_lines = filter_lines_within_boundary(vertical_lines, axis='vertical', min_length=100, margin=15)
        filtered_horizontal_lines = filter_lines_within_boundary(horizontal_lines, axis='horizontal', min_length=100, margin=15)

        print(f"Filtered vertical lines: {len(filtered_vertical_lines)}")
        print(f"Filtered horizontal lines: {len(filtered_horizontal_lines)}")

        # Fit lines using linear regression (robust handling when there are not enough lines)
        vertical_offset = 20  # Adjust offset for vertical line
        vertical_line = None
        horizontal_line = None

        if len(filtered_vertical_lines) > 1:  # Check we have at least 2 points
            vertical_points = [(x1, y1) for x1, y1, x2, y2 in filtered_vertical_lines]
            slope_v, intercept_v = fit_line_linear_regression(vertical_points, axis='vertical')
            vertical_line = get_line_from_regression(slope_v, intercept_v, self.image.shape, axis='vertical', offset=vertical_offset)

        if len(filtered_horizontal_lines) > 1:  # Check we have at least 2 points
            horizontal_points = [(x1, y1) for x1, y1, x2, y2 in filtered_horizontal_lines]
            slope_h, intercept_h = fit_line_linear_regression(horizontal_points, axis='horizontal')
            horizontal_line = get_line_from_regression(slope_h, intercept_h, self.image.shape, axis='horizontal')

        return vertical_line, horizontal_line

    def visualize_chip_boundary(self):
        # Get detected boundary lines
        vertical_line, horizontal_line = self.detect_chip_boundary()

        # Use the visualizer to plot the results
        visualizer = Visualizer(self.image)
        visualizer.plot_boundaries(vertical_line, horizontal_line)