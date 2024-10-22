import cv2
import numpy as np
import json
import os


def load_image(image_path):
    # Load the image from the given file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def preprocess_image(image):
    # Convert image to grayscale and apply Gaussian blur to reduce noise
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image


def detect_waveguide_and_resonators(image):
    # Detect the waveguide entrance and resonator rings in the image
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image. Ensure the image contains the chip.")

    # Detect the waveguide entrance (assumed to be the largest contour)
    waveguide_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(waveguide_contour)
    if M["m00"] != 0:
        waveguide_x = int(M["m10"] / M["m00"])
        waveguide_y = int(M["m01"] / M["m00"])
    else:
        waveguide_x, waveguide_y = 0, 0

    # Detect resonator rings (assumed to be smaller contours)
    resonator_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 1000]
    resonator_locations = []
    for contour in resonator_contours:
        M_res = cv2.moments(contour)
        if M_res["m00"] != 0:
            resonator_x = int(M_res["m10"] / M_res["m00"])
            resonator_y = int(M_res["m01"] / M_res["m00"])
            resonator_locations.append({"x": resonator_x, "y": resonator_y})

    return {"waveguide_entrance": {"x": waveguide_x, "y": waveguide_y}, "resonator_locations": resonator_locations}


def save_result_to_json(data, output_path):
    # Save the detected coordinates in a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Results saved in {output_path}")


def main(image_path, output_json_path):
    try:
        image = load_image(image_path)
        processed_image = preprocess_image(image)
        detection_result = detect_waveguide_and_resonators(processed_image)
        save_result_to_json(detection_result, output_json_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Example usage
    image_path = 'chip_image.jpg'
    output_json_path = 'waveguide_resonator_output.json'
    main(image_path, output_json_path)
