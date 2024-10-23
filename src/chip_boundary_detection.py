from src.edge_detection import EdgeDetector
from src.utils import filter_lines_within_boundary, fit_line_linear_regression, get_line_from_regression
from src.visualizer import Visualizer  # Import the new Visualizer class

class ChipBoundaryDetector:
    def __init__(self, image_path):
        self.edge_detector = EdgeDetector(image_path)
        self.image = self.edge_detector.image

    def detect_chip_boundary(self):
        # Detect edges
        edges_vertical, edges_horizontal = self.edge_detector.detect_edges()

        # Detect Hough lines
        vertical_lines, horizontal_lines = self.edge_detector.detect_hough_lines(edges_vertical, edges_horizontal)

        # Filter and process lines
        filtered_vertical_lines = filter_lines_within_boundary(vertical_lines, axis='vertical', min_length=200, margin=10)
        filtered_horizontal_lines = filter_lines_within_boundary(horizontal_lines, axis='horizontal', min_length=200, margin=10)

        # Fit lines using linear regression
        vertical_offset = 20  # Adjust offset for vertical line
        if filtered_vertical_lines:
            vertical_points = [(x1, y1) for x1, y1, x2, y2 in filtered_vertical_lines]
            slope_v, intercept_v = fit_line_linear_regression(vertical_points, axis='vertical')
            vertical_line = get_line_from_regression(slope_v, intercept_v, self.image.shape, axis='vertical', offset=vertical_offset)
        else:
            vertical_line = None

        if filtered_horizontal_lines:
            horizontal_points = [(x1, y1) for x1, y1, x2, y2 in filtered_horizontal_lines]
            slope_h, intercept_h = fit_line_linear_regression(horizontal_points, axis='horizontal')
            horizontal_line = get_line_from_regression(slope_h, intercept_h, self.image.shape, axis='horizontal')
        else:
            horizontal_line = None

        return vertical_line, horizontal_line

    def visualize_chip_boundary(self):
        vertical_line, horizontal_line = self.detect_chip_boundary()
        visualizer = Visualizer(self.image)
        visualizer.plot_boundaries(vertical_line, horizontal_line)
