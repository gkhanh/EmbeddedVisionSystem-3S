import matplotlib.pyplot as plt
import cv2

class Visualizer:
    def __init__(self, image):
        self.image = image

    def plot_edges(self, edges_vertical, edges_horizontal):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(edges_vertical, cmap='gray')
        plt.title("Vertical Edges")

        plt.subplot(1, 2, 2)
        plt.imshow(edges_horizontal, cmap='gray')
        plt.title("Horizontal Edges")

        plt.show()

    def plot_boundaries(self, vertical_line, horizontal_line):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Chip Boundary")

        # Draw vertical line if present
        if vertical_line is not None:
            x1, y1, x2, y2 = vertical_line
            plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)  # Green line for vertical boundary

        # Draw horizontal line if present
        if horizontal_line is not None:
            x1, y1, x2, y2 = horizontal_line
            plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)  # Blue line for horizontal boundary

        plt.axis('on')  # Show coordinates
        plt.show()

    def plot_waveguide_entrance(self, waveguide_entrance_line=None, entrance_point=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Waveguide Entrance")

        # Plot the waveguide entrance line
        if waveguide_entrance_line is not None:
            x1, y1, x2, y2 = waveguide_entrance_line
            plt.plot([x1, x2], [y1, y2], color='green', linewidth=2, label='Waveguide Entrance Line')

        # Plot the entrance point
        if entrance_point is not None:
            plt.scatter(entrance_point[0], entrance_point[1], color='red', s=20, label='Entrance Point')

        plt.legend(loc='upper right')
        plt.show()

    def plot_combined(self, vertical_line=None, horizontal_line=None, waveguide_entrance_line=None,
                      entrance_point=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Combined Chip Boundary and Waveguide Entrance Detection")

        # Plot Chip Boundaries
        if vertical_line is not None:
            x1, y1, x2, y2 = vertical_line
            plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2, label='Vertical Boundary')

        if horizontal_line is not None:
            x1, y1, x2, y2 = horizontal_line
            plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2, label='Horizontal Boundary')

        # Plot Waveguide Entrance
        if waveguide_entrance_line is not None:
            x1, y1, x2, y2 = waveguide_entrance_line
            plt.plot([x1, x2], [y1, y2], color='green', linewidth=2, label='Waveguide Entrance Line')

        if entrance_point is not None:
            plt.scatter(entrance_point[0], entrance_point[1], color='red', s=20, label='Waveguide Entrance Point')

        plt.legend(loc='upper right')
        plt.show()