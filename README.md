# Vision System for Detecting Waveguide Entrance on Photonic Chip

## Project Overview

This project is focused on developing a vision system capable of detecting the waveguide entrance and resonator rings on a photonic chip. The system processes an image of the chip and returns the coordinates of the detected features in a JSON file. This program is designed to work with a stationary image capture system.

## Features

- Detects waveguide entrance on a photonic chip
- Detects resonator rings on the chip
- Returns the detected positions in JSON format
- No real-time processing required
- Pre-assumes the image is taken, and gripper movement is handled separately

## Dependencies

The project uses the following Python libraries:

- **OpenCV**: For image processing
- **NumPy**: For numerical operations
- **json**: Standard library for handling JSON data

## Setup Instructions

To set up and run this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/gkhanh/EmbeddedVisionSystem-3S.git
    cd vision-system-photonic-chip
    ```

2. Install the necessary dependencies by running the shell script:

    ```bash
    chmod +x install_dependencies.sh
    ./install_dependencies.sh
    ```

3. Place the image of the photonic chip (e.g., `chip_image.jpg`) in the root folder.

4. Run the Python script to detect the waveguide entrance:

    ```bash
    python3 waveguide_vision_system.py
    ```

## Output

The program will process the image and return the coordinates of the waveguide entrance and resonator rings in the following format:

```json
{
    "waveguide_entrance": {
        "x": 250,
        "y": 300
    },
    "resonator_locations": [
        {"x": 120, "y": 150},
        {"x": 180, "y": 210}
    ]
}
```


## Notes ## 

- The gripper starts at position [0:0:0]
- Return the position of the waveguide entrance location as coordinates in
JSON format
- Reference plane: Based on the taken picture of the camera
- FOV(Field of view) & DOF(Depth of field) are not taken into account
- Picture only zoon in a part of the chip, not the whole chip

**Process**

1. Gripper grab the chip
2. Gripper moves the chip under the camera
3. Camera takes a picture
4. Program processes the image
5. Return result as JSON file

**Functional Requirements**
- Don't have to be real-time processing
- Only taken the icture and analyze it
- Return the result in JSON
