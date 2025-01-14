from src.chip_boundary_detection import ChipBoundaryDetector
from src.waveguide_entrance_detection import WaveguideEntranceDetector
from src.json_output import JsonOutput
from src.visualizer import Visualizer  # Import your visualizer
import cv2


def main():
    image_path = './data/foto1.png'  # Path to the image

    # 1. Perform Chip Boundary Detection
    print(f"Performing Chip Boundary Detection on {image_path}...")
    boundary_detector = ChipBoundaryDetector(image_path)
    vertical_line, horizontal_line = boundary_detector.detect_chip_boundary()

    # 2. Perform Waveguide Entrance Detection
    print(f"Performing Waveguide Entrance Detection on {image_path}...")
    waveguide_detector = WaveguideEntranceDetector(image_path)
    waveguide_entrance_line, entrance_point = waveguide_detector.detect_waveguide_entrance()

    # 3. Use Visualizer to plot combined results
    image = cv2.imread(image_path)
    visualizer = Visualizer(image)

    # Plot both boundary and waveguide entrance
    visualizer.plot_combined(vertical_line, horizontal_line, waveguide_entrance_line, entrance_point)

    # 4. Save entrance point as JSON if detected
    if entrance_point:
        json_output = JsonOutput(image_path,
                                 output_json_path=f'./result/entrance_coordinates_{image_path.split("/")[-1]}.json')
        json_output.detect_and_save_entrance()
    else:
        print("No entrance point detected, nothing to save.")


if __name__ == "__main__":
    main()
