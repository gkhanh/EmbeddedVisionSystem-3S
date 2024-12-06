from src.camera_calibration.camera_calibration import AxisMapper
from src.camera_calibration.visualizer_utils import visualize_combined_axis_map

# def test_axis_mapping(structured_mode=False, move_y=10, move_x=10):
#     """
#     Test the AxisMapper:
#     - Generate 1 random point (first point).
#     - Derive the second and third points based on manipulator movements.
#     """
#     # Initialize the axis mapper
#     mapper = AxisMapper(structured_mode=structured_mode)
#
#     if structured_mode:
#         # Add the first point; the rest will be derived
#         mapper.add_points(
#             camera_x=20,
#             camera_y=30,
#             manipulator_x=0,
#             manipulator_y=0,
#             move_y=move_y,
#             move_x=move_x
#         )
#     else:
#         # Add random points
#         mapper.add_points(camera_x=100, camera_y=200, manipulator_x=0, manipulator_y=0)
#         mapper.add_points(camera_x=80, camera_y=200, manipulator_x=-10, manipulator_y=0)
#         mapper.add_points(camera_x=120, camera_y=220, manipulator_x=10, manipulator_y=5)
#
#     # Perform calculations
#     alpha, beta = mapper.calculate_rotation()
#     scale = mapper.calculate_scale()
#     t1, t2 = mapper.calculate_translation(scale, alpha, beta)
#
#     # Calculate transformed points (camera to manipulator or vice versa)
#     inverted_points = mapper.invert_transformation(scale, alpha, beta, t1, t2)
#
#     # Determine camera axis map components
#     manipulator_points = mapper.manipulator_points
#     camera_axis_origin = manipulator_points[0]
#     camera_axis_y = (
#         (manipulator_points[1][0] - manipulator_points[0][0]) * scale,
#         (manipulator_points[1][1] - manipulator_points[0][1]) * scale,
#     )
#     camera_axis_x = (
#         (manipulator_points[2][0] - manipulator_points[1][0]) * scale,
#         (manipulator_points[2][1] - manipulator_points[1][1]) * scale,
#     )
#
#     # Print results for validation
#     print(f"Rotation Angles: Alpha={alpha:.2f}°, Beta={beta:.2f}°")
#     print(f"Scale Factor: {scale:.2f} pixels/mm")
#     print(f"Translation: T1={t1:.2f}, T2={t2:.2f}")
#
#     # Visualize the derived camera axis map
#     # visualize_camera_axis_map(manipulator_points, camera_axis_origin, camera_axis_x, camera_axis_y)
#     visualize_full_axis_maps(mapper.camera_points, mapper.manipulator_points, inverted_points)
#
#
# if __name__ == '__main__':
#     # Test with structured points, prompting for real or default values
#     print("Testing with structured points...")
#     test_axis_mapping(structured_mode=True, move_y=-10, move_x=10)

def test_axis_mapping(move_y=-10, move_x=10):
    """
    Test the AxisMapper:
    - Generate 1 random point (first point).
    - Derive the second and third points based on manipulator movements.
    """
    # Initialize the axis mapper
    mapper = AxisMapper(structured_mode=True)

    # Generate the first point (random or fixed for demo)
    first_camera_x, first_camera_y = 20, 30
    first_manipulator_x, first_manipulator_y = 0, 0

    # Derive the second point
    second_camera_x = first_camera_x
    second_camera_y = first_camera_y + move_y
    second_manipulator_x = first_manipulator_x
    second_manipulator_y = first_manipulator_y + move_y

    # Derive the third point
    third_camera_x = second_camera_x + move_x
    third_camera_y = second_camera_y
    third_manipulator_x = second_manipulator_x + move_x
    third_manipulator_y = second_manipulator_y

    # Add points to the mapper
    mapper.add_points(first_camera_x, first_camera_y, first_manipulator_x, first_manipulator_y)
    mapper.add_points(second_camera_x, second_camera_y, second_manipulator_x, second_manipulator_y)
    mapper.add_points(third_camera_x, third_camera_y, third_manipulator_x, third_manipulator_y)

    # Perform calculations
    alpha, beta = mapper.calculate_rotation()
    scale = mapper.calculate_scale()
    t1, t2 = mapper.calculate_translation(scale, alpha, beta)

    # Transform points
    transformed_points = mapper.transform_points(scale, alpha, beta, t1, t2)

    # Visualize combined axis map
    visualize_combined_axis_map(mapper.manipulator_points, mapper.camera_points, transformed_points)


if __name__ == '__main__':
    # Run the test
    test_axis_mapping(move_y=-10, move_x=10)
