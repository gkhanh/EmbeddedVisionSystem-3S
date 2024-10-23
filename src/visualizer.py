import matplotlib.pyplot as plt
import cv2

class Visualizer:
    def __init__(self, image):
        self.image = image

    def plot_boundaries(self, vertical_line, horizontal_line):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Chip Boundary")

        # Draw vertical line if present
        if vertical_line is not None:
            x1, y1, x2, y2 = vertical_line
            plt.plot([x1, x2], [y1, y2], color='green', linewidth=2)  # Green line for vertical boundary

        # Draw horizontal line if present
        if horizontal_line is not None:
            x1, y1, x2, y2 = horizontal_line
            plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)  # Blue line for horizontal boundary

        plt.axis('on')  # Show coordinates
        plt.show()
