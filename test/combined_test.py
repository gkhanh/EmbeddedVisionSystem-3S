from src.EgdeDetectionAlgorithm.chip_boundary_detection import ChipBoundaryDetector
from src.EgdeDetectionAlgorithm.waveguide_entrance_detection import WaveguideEntranceDetector
from src.json_output import JsonOutput
from src.waveguide_entrance_detection.visualizer import Visualizer
import cv2


def test_combined_detection(image_path):
    # 1. Perform Chip Boundary Detection
    print(f"Testing Chip Boundary Detection and Waveguide Entrance Detection on {image_path}...")
    chip_boundary_detector = ChipBoundaryDetector(image_path)
    vertical_line, horizontal_line = chip_boundary_detector.detect_chip_boundary()

    # 2. Perform Waveguide Entrance Detection
    waveguide_detector = WaveguideEntranceDetector(image_path)
    waveguide_entrance_line, entrance_point = waveguide_detector.detect_waveguide_entrance()

    waveguide_saver = JsonOutput(image_path,
                                             output_json_path=f'../result/entrance_coordinates_{image_path.split("/")[-1]}.json')
    entrance_point = waveguide_saver.detect_and_save_entrance()  # This will save the entrance and print info

    # 3. Read the image
    image = cv2.imread(image_path)

    # 4. Plot the combined results (Chip Boundary + Waveguide Entrance)
    visualizer = Visualizer(image)
    visualizer.plot_combined(vertical_line, horizontal_line, waveguide_entrance_line, entrance_point)

if __name__ == "__main__":
    # List of test image paths
    test_images = [
        "../data/foto1.png",
        "../data/foto2.png",
        "../data/foto3.png",
        "../data/foto4.png",
        "../data/foto5.png",
        "../data/foto6.png",
        "../data/foto7.png",
        "../data/foto8.png",
        "../data/foto9.png",
        "../data/foto10.png"
    ]

    # Run the combined test for each image
    for image_path in test_images:
        test_combined_detection(image_path)
