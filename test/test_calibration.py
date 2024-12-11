import numpy as np

from src.camera_calibration.camera_calibration import (visualize_verification, calculate_alpha, calculate_beta,
                                                       calculate_scale,
                                                       calculate_affine_transformation,
                                                       calculate_camera_movement_offset)
from src.data.data import manipulation_points, camera_points


# Test the Affine transformation function based on formula
def test_transformation():
    transformation_matrix = calculate_affine_transformation(camera_points, manipulation_points)

    # Compute the inverse matrix
    inverse_matrix = np.linalg.inv(transformation_matrix)

    for m_point in manipulation_points:
        m_point_hom = np.array([m_point[0], m_point[1], 1])
        c_point_hom = inverse_matrix @ m_point_hom
        c_point = (c_point_hom[0] / c_point_hom[2], c_point_hom[1] / c_point_hom[2])
        print(f"Manipulation Point {m_point} maps back to Camera Point {c_point}")


if __name__ == "__main__":
    alpha_angle = calculate_alpha(camera_points[0], camera_points[1])
    beta_angle = calculate_beta(camera_points[0], camera_points[1])
    pixel_mm_ratio = calculate_scale(camera_points[0], camera_points[1], manipulation_points[1], manipulation_points[2])

    print(f"Alpha (degrees): {alpha_angle}")
    print(f"Beta (degrees): {beta_angle}")
    print(f"Pixel-to-mm scale: {pixel_mm_ratio}")

    result = calculate_camera_movement_offset(camera_points[0], manipulation_points[1], manipulation_points[2],
                                              camera_points[1], camera_points[2])
    print(f"Camera Movement Offset (T1, T2): {result}")

    # Visualize the combined system
    visualize_verification()
    # test the affine transformation
    test_transformation()
