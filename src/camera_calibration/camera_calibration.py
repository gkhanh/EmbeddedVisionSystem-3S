import math

import matplotlib.pyplot as plt
import numpy as np


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
    return math.degrees(math.atan2((y2 - y1), (x2 - x1)))


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
    return math.degrees(math.atan2((y3 - y2), (x3 - x2)))


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


def line_from_points(p1, p2):
    """
    Construct a line equation (A, B, C) for the line passing through points p1 and p2.
    Line form: A*x + B*y + C = 0

    Parameters:
        p1 (tuple): (x1, y1)
        p2 (tuple): (x2, y2)

    Returns:
        tuple: (A, B, C) representing the line
    """

    (x1, y1), (x2, y2) = p1, p2
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)
    return A, B, C


def line_intersection(L1, L2):
    """
    Find the intersection of two lines given by (A1, B1, C1) and (A2, B2, C2).

    Parameters:
        L1 (tuple): (A1, B1, C1)
        L2 (tuple): (A2, B2, C2)

    Returns:
        tuple or None: (x, y) intersection point if exists, otherwise None for parallel lines.
    """
    # Solve intersection of two lines: A1x + B1y + C1 = 0 and A2x + B2y + C2 = 0
    A1, B1, C1 = L1
    A2, B2, C2 = L2
    denominator = A1 * B2 - A2 * B1
    if denominator == 0:
        # Lines are parallel or coincident; handle carefully
        return None
    x = (B2 * (-C1) - B1 * (-C2)) / denominator
    y = (A1 * (-C2) - A2 * (-C1)) / denominator
    return x, y


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
    print(f"t1: {t1}, t2: {t2}")

    # Calculate T1 and T2
    T1 = t1 * math.cos(Beta) + t2 * math.cos(Alpha)
    T2 = t1 * math.sin(Beta) + t2 * math.sin(Alpha)

    return T1, T2


def calculate_affine_transformation(camera_point_coordinates, manipulation_points_coordinates):
    """
    Calculate the affine transformation parameters that map camera points to manipulation points.

    Parameters:
    camera_point_coordinates: List of tuples, camera points (x, y)
    manipulation_points_coordinates: List of tuples, corresponding manipulation points (X, Y)

    Returns:
    numpy array: Affine transformation matrix [[a, b, c], [d, e, f]]
    """
    # Prepare matrices for least squares solution
    A = []
    b = []
    for (x, y), (X, Y) in zip(camera_point_coordinates, manipulation_points_coordinates):
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


def camera_to_global(transformation_matrix, camera_point):
    """
    Convert a camera point from pixel coordinates to global (manipulation) coordinates.
    This uses the affine transformation matrix to map from camera frame to manipulation frame.

    Parameters:
        transformation_matrix (np.ndarray): 3x3 affine transformation matrix
        camera_point (tuple): (cx, cy) camera coordinates in pixels

    Returns:
        tuple: (X, Y) coordinates in the manipulation (global) frame
    """
    cx, cy = camera_point
    cp_h = np.array([cx, cy, 1])
    gp_h = transformation_matrix @ cp_h
    return gp_h[0] / gp_h[2], gp_h[1] / gp_h[2]


def signed_angle(u, v):
    # u and v are 2D vectors
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    angle_rad = math.atan2(cross, dot)
    return math.degrees(angle_rad)


def calculate_camera_movement_offset(camera_points, manipulation_points):
    # Calculate the affine transformation
    transformation_matrix = calculate_affine_transformation(camera_points, manipulation_points)

    # Transform camera points to global (manipulation) coordinates
    global_points = [camera_to_global(transformation_matrix, cp) for cp in camera_points]

    # Let's name them for clarity
    G1, G2, G3 = global_points  # Corresponding to C1, C2, C3

    # Assume:
    # Line G1->G2 defines direction of camera Y-axis
    # Line G1->G3 defines direction of camera X-axis

    # Find intersection of these two lines to locate camera origin
    L_y = line_from_points(G1, G2)
    L_x = line_from_points(G1, G3)

    camera_origin = line_intersection(L_y, L_x)
    if camera_origin is None:
        # If lines don't intersect, fall back or handle error
        camera_origin = G1  # fallback, though this shouldn't happen if axes are well-defined

    # Scale computation: Use original logic with the first two manipulation points
    # Assume cameraPoints[0]->cameraPoints[1] corresponds to manipulationPoints[1]->manipulationPoints[2] as before
    pixel_per_mm = calculate_scale(camera_points[0], camera_points[1], manipulation_points[1], manipulation_points[2])

    # Rotation computation
    Alpha, Beta = calculate_rotation_with_actual_axes(transformation_matrix)

    # Find movement offset (T1, T2) for cameraPoints[1]
    # Since the camera coordinate (x2, y2) are originally in pixels, we consider that C2 is supposed to lie along the axes.
    # Therefore, translate camera_points[1] into camera-based coordinates assuming camera_origin in global coordinates is (0,0)
    # However, since we are using pixel coords directly in `calculate_movement`, consider (x2, y2) from camera_points[1].
    x2, y2 = camera_points[1]

    T1, T2 = calculate_movement(x2, y2, pixel_per_mm, Alpha, Beta)

    return pixel_per_mm, camera_origin, T1, T2


def calculate_rotation_with_actual_axes(transformation_matrix):
    # Extract linear part
    a, b, _ = transformation_matrix[0]
    d, e, _ = transformation_matrix[1]

    # Camera axes directions:
    camera_x_dir = np.array([a, d])  # In manipulation frame
    camera_y_dir = np.array([b, e])  # In manipulation frame

    # Normalize
    camera_x_dir = camera_x_dir / np.linalg.norm(camera_x_dir)
    camera_y_dir = camera_y_dir / np.linalg.norm(camera_y_dir)

    # Define manipulation axes
    manip_x_dir = np.array([1.0, 0.0])
    manip_y_dir = np.array([0.0, 1.0])

    # Use signed_angle to get Alpha and Beta directly
    # Alpha: angle between camera Y-axis and manipulation Y-axis
    Alpha = signed_angle(camera_y_dir, manip_y_dir)

    # Beta: angle between camera X-axis and manipulation X-axis
    Beta = signed_angle(camera_x_dir, manip_x_dir)

    return Alpha, Beta


def visualize_verification(camera_points, manipulation_points):
    # Create a larger figure, for example 10 inches wide by 8 inches tall
    plt.figure(figsize=(11, 11))

    transformation_matrix = calculate_affine_transformation(camera_points, manipulation_points[1:4])
    c, f = transformation_matrix[0, 2], transformation_matrix[1, 2]
    camera_origin_manip = (c, f)
    a, b = transformation_matrix[0, 0], transformation_matrix[0, 1]
    d, e = transformation_matrix[1, 0], transformation_matrix[1, 1]

    # Normalize direction vectors
    camera_x_dir = np.array([a, d])
    camera_y_dir = np.array([b, e])
    print("Direction vectors:", camera_x_dir, camera_y_dir)

    # Check for zero-length vectors
    if np.allclose(camera_x_dir, 0) or np.allclose(camera_y_dir, 0):
        print("Error: Computed direction vectors are zero-length or invalid.")
        return

    camera_x_dir_unit = camera_x_dir / np.linalg.norm(camera_x_dir)
    camera_y_dir_unit = camera_y_dir / np.linalg.norm(camera_y_dir)

    # Set arrow scales
    arrow_scale = 10  # Adjust this value for arrow length
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
    plt.xlim(-10, 20)
    plt.ylim(-15, 20)
    plt.show()
