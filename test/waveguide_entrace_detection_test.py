import os
from src.waveguide_entrance_detection import WaveguideEntranceDetector

def test_waveguide_entrance_detection():
    data_dir = "../data"
    for filename in os.listdir(data_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print(f"Testing with image: {filename}")
            detector = WaveguideEntranceDetector(os.path.join(data_dir, filename))
            detector.visualize_waveguide_entrance()

if __name__ == "__main__":
    test_waveguide_entrance_detection()