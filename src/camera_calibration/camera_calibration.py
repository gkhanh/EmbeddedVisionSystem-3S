import math

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic


def calculate_alpha(firstPoint, secondPoint):
    """
    Calculate Alpha, the angle in degrees the camera's Y-axis deviates.

    Parameters:
    x1, y1, x2, y2: Coordinates of the first two points.

    Returns:
    float: Alpha in degrees.
    """
    x1, y1 = firstPoint
    x2, y2 = secondPoint
    return math.degrees(math.atan((x2 - x1) / (y1 - y2)))


def calculate_beta(firstPoint, secondPoint):
    """
    Calculate Beta, the angle in degrees the camera's X-axis deviates.

    Parameters:
    x2, y2, x3, y3: Coordinates of the second and third points.

    Returns:
    float: Beta in degrees.
    """
    x2, y2 = firstPoint
    x3, y3 = secondPoint
    return math.degrees(math.atan((x3 - x2) / (y3 - y2)))


def calculate_scale(cameraPoint1, cameraPoint2, real_point1, real_point2):
    """
    Calculate the scale (pixel/mm) between two points.

    Parameters:
    pixel_point1, pixel_point2: Tuples representing pixel coordinates of the two points (x, y).
    real_point1, real_point2: Tuples representing real-world coordinates of the two points (X, Y) in mm.

    Returns:
    float: Scale in pixels per mm.
    """
    # Unpack the pixel and real-world points
    x1, y1 = cameraPoint1
    x2, y2 = cameraPoint2
    X1, Y1 = real_point1
    X2, Y2 = real_point2

    # Pixel distance (Euclidean distance)
    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # Real-world distance (Euclidean distance)
    real_distance = math.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
    # Scale calculation
    scale = pixel_distance / real_distance

    return scale


def calculate_movement(x2, y2, pixel_per_mm, Alpha_deg, Beta_deg):
    """
    Calculates the amount of X and Y movement in mm to reach the camera axis origin.

    Parameters:
    x2 (float): Value for x2 in pixels.
    y2 (float): Value for y2 in pixels.
    pixel_per_mm (float): Conversion factor from pixels to mm.
    Alpha_deg (float): Angle Alpha in degrees.
    Beta_deg (float): Angle Beta in degrees.

    Returns:
    tuple: (T1, T2) where T1 is the X movement and T2 is the Y movement in mm.
    """
    # Convert angles from degrees to radians
    Alpha = math.radians(Alpha_deg)
    Beta = math.radians(Beta_deg)

    # Calculate t1 and t2
    t1 = x2 * pixel_per_mm
    t2 = y2 * pixel_per_mm

    # Debug print to check value t1 and t2
    ic(t1, t2)

    # Calculate T1 and T2
    T1 = t1 * math.sin(Beta) + t2 * math.sin(Alpha)
    T2 = t1 * math.cos(Beta) + t2 * math.cos(Alpha)

    return T1, T2


# @icAll
def calculate_camera_movement_offset(cameraPoint1, real_point1, real_point2, cameraPoint2, cameraPoint3):
    """
    Calculate the camera movement offsets in the X and Y directions.

    Parameters:
    cameraPoint1: Tuple representing the first camera point (x, y) in pixels.
    real_point1: Tuple representing the first real-world point (X, Y) in mm.
    real_point2: Tuple representing the second real-world point (X, Y) in mm.
    cameraPoint2: Tuple representing the second camera point (x, y) in pixels.
    cameraPoint3: Tuple representing the third camera point (x, y) in pixels.

    Returns:
    tuple: (cameraXOffset, cameraYOffset) representing the movement offsets in mm.
    """

    alpha = calculate_alpha(cameraPoint1, cameraPoint2)
    beta = calculate_beta(cameraPoint2, cameraPoint3)
    scalePixelInMilimeter = calculate_scale(cameraPoint1, cameraPoint2, real_point1, real_point2)

    cameraXOffset, cameraYOffset = calculate_movement(cameraPoint2[0], cameraPoint2[1], scalePixelInMilimeter, alpha,
                                                      beta)

    return cameraXOffset, cameraYOffset


def calculate_affine_transformation(camera_points, manipulation_points):
    """
    Calculate the affine transformation parameters that map camera points to manipulation points.

    Parameters:
    camera_points: List of tuples, camera points (x, y)
    manipulation_points: List of tuples, corresponding manipulation points (X, Y)

    Returns:
    numpy array: Affine transformation matrix [[a, b, c], [d, e, f]]
    """
    # Prepare matrices for least squares solution
    A = []
    b = []
    for (x, y), (X, Y) in zip(camera_points, manipulation_points):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        b.append(X)
        b.append(Y)

    A = np.array(A)
    b = np.array(b)

    # Solve for the parameters a, b, c, d, e, f
    params, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    a, b, c, d, e, f = params
    return np.array([[a, b, c],
                     [d, e, f],
                     [0, 0, 1]])  # Homogeneous coordinates


def visualize_verification():
    manipulation_points = [
        (0, 0),  # M0
        (3, 9),  # M1
        (3, 8),  # M2
        (4, 8),  # M3
    ]

    camera_points = [
        (2, 3),  # C1
        (3, 2),  # C2
        (4, 3),  # C3
    ]

    transformation_matrix = calculate_affine_transformation(camera_points, manipulation_points[1:4])
    c, f = transformation_matrix[0, 2], transformation_matrix[1, 2]
    camera_origin_manip = (c, f)
    a, b = transformation_matrix[0, 0], transformation_matrix[0, 1]
    d, e = transformation_matrix[1, 0], transformation_matrix[1, 1]

    # Normalize direction vectors
    camera_x_dir = np.array([a, d])
    camera_y_dir = np.array([b, e])
    camera_x_dir_unit = camera_x_dir / np.linalg.norm(camera_x_dir)
    camera_y_dir_unit = camera_y_dir / np.linalg.norm(camera_y_dir)

    # Set arrow scales
    arrow_scale = 4  # Adjust this value for arrow length
    arrow_scale_manip = 10  # Scale for manipulation axes

    plt.scatter([p[0] for p in manipulation_points], [p[1] for p in manipulation_points], color='red',
                label='Manipulation Points')
    plt.scatter(camera_origin_manip[0], camera_origin_manip[1], color='blue', label='Camera Origin')

    # Plot camera X-axis arrow
    plt.arrow(float(camera_origin_manip[0]), float(camera_origin_manip[1]),
              float(camera_x_dir_unit[0] * arrow_scale), float(camera_x_dir_unit[1] * arrow_scale),
              color='blue', label='Camera X-axis', head_width=0.2, head_length=0.3)

    # Plot camera Y-axis arrow
    plt.arrow(float(camera_origin_manip[0]), float(camera_origin_manip[1]),
              float(camera_y_dir_unit[0] * arrow_scale), float(camera_y_dir_unit[1] * arrow_scale),
              color='green', label='Camera Y-axis', head_width=0.2, head_length=0.3)

    # Plot manipulation X-axis arrow
    plt.arrow(manipulation_points[0][0], manipulation_points[0][1], arrow_scale_manip, 0,
              color='red', label='Manipulation X-axis', head_width=0.2, head_length=0.3)

    # Plot manipulation Y-axis arrow
    plt.arrow(manipulation_points[0][0], manipulation_points[0][1], 0, arrow_scale_manip,
              color='black', label='Manipulation Y-axis', head_width=0.2, head_length=0.3)

    plt.xlabel('Manipulation X-axis')
    plt.ylabel('Manipulation Y-axis')
    plt.title('Verification of Coordinate Transformation')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-2, 8)
    plt.ylim(0, 12)
    plt.show()

