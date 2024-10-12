Embedded Vision System for Photonic Chip Assembly

-- Description --
  
  This project focuses on developing an embedded vision system for detecting the waveguide entrance on photonic chips during automated assembly. The system aims to achieve micrometer-level precision, and it will be integrated in real-time with an assembly machine. 

-- Objectives --
  1. Real-Time Detection: Identify the waveguide entrance on the photonic chip in real-time.
  2. Micrometer Precision: Ensure high precision to meet photonic assembly requirements.
  3. Coordinate System Mapping: Implement a coordinate system to align the camera’s output with the assembly machine's positioning system.

-- Key Features --
  1. Image Processing: Uses OpenCV for capturing and analyzing images.
  2. Coordinate Transformation: Maps detected features to the machine's coordinate system for precise placement.
  3. Calibration: Includes tools for both intrinsic and extrinsic camera calibration to ensure accurate measurements.

-- Project Structure --
├── src/                    # Core functionality
├── data/                   # Sample data and images
└── README.md               # Project overview
