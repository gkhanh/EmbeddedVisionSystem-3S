from src.camera_calibration.camera_calibration import extrapolateManipulatorPosition, calculateCameraMovementOffset, \
    visualize_points

if __name__ == "__main__":
    manipulationPoint1 = (2, 4) # Real manipulation points
    manipulationPoint2 = (2, 2.5)

    firstCameraPoint = (1, 3) # First camera point

    secondPoint, thirdPoint = extrapolateManipulatorPosition(firstCameraPoint)
    result = calculateCameraMovementOffset(firstCameraPoint, manipulationPoint1, manipulationPoint2)
    visualizationSet = [firstCameraPoint, secondPoint, thirdPoint]
    visualize_points(visualizationSet)