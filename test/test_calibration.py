from src.camera_calibration.camera_calibration import AxisMapper
from src.camera_calibration.visualizer_utils import visualize_full_axis_maps

# Tests the AxisMapper class and visualizes all three maps (camera, manipulator, and merged) clearly.
def test_full_axis_mapping_with_visualization():

    # Initialize the axis mapper
    mapper = AxisMapper()

    # Add corresponding points (camera coordinates and manipulator coordinates)
    mapper.add_points(camera_x=100, camera_y=200, manipulator_x=0, manipulator_y=0)
    mapper.add_points(camera_x=80, camera_y=200, manipulator_x=-10, manipulator_y=0)
    mapper.add_points(camera_x=120, camera_y=220, manipulator_x=10, manipulator_y=5)

    # Perform calculations
    alpha, beta = mapper.calculate_rotation()
    scale = mapper.calculate_scale()
    t1, t2 = mapper.calculate_translation(scale, alpha, beta)
    transformed_points = mapper.transform_points(scale, alpha, beta, t1, t2)

    # Visualize all three maps
    visualize_full_axis_maps(mapper.camera_points, mapper.manipulator_points, transformed_points)


# Run the test with full visualization
if __name__ == '__main__':
    test_full_axis_mapping_with_visualization()
