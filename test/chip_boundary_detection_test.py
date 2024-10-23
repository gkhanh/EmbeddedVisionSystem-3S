import os
from src.chip_boundary_detection import ChipBoundaryDetector


def test_all_images_in_directory(directory_path):
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is an image based on its extension
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(directory_path, filename)
            print(f"Testing with image: {image_path}")

            # Create a ChipBoundaryDetector object for each image
            boundary_detector = ChipBoundaryDetector(image_path)

            # Visualize the chip boundary for each image
            boundary_detector.visualize_chip_boundary()


def main():
    # Set the directory containing the images
    image_directory = '../data/'  # Path to the folder containing images

    # Test all images in the directory
    test_all_images_in_directory(image_directory)


if __name__ == "__main__":
    main()