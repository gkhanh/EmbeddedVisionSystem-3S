import json
from src.waveguide_entrance_detection import WaveguideEntranceDetector

class JsonOutput:
    def __init__(self, image_path, output_json_path='../entrance_coordinates.json'):
        self.image_path = image_path
        self.output_json_path = output_json_path

    def save_as_json(self, coordinate):
        # Save entrance point coordinates as a JSON file.

        if coordinate:
            # Convert NumPy integer types to Python int
            coordinate = [int(c) for c in coordinate]

            # Save to JSON file
            with open(self.output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump({'waveguide_entrance': {'x': coordinate[0], 'y': coordinate[1]}}, json_file)
            print(f"Entrance point saved to {self.output_json_path}.")
        else:
            print("No entrance point detected to save.")

    def detect_and_save_entrance(self):
        # Detect waveguide entrance and save coordinates to a JSON file.
        # Create the detector and perform the detection
        waveguide_detector = WaveguideEntranceDetector(self.image_path)
        waveguide_entrance_line, entrance_point = waveguide_detector.detect_waveguide_entrance()

        # Handle entrance point saving
        if entrance_point:
            print(f"Entrance Point Detected: {entrance_point}")
            self.save_as_json(entrance_point)
        else:
            print("No entrance point detected.")

        return entrance_point
