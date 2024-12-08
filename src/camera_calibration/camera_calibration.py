import matplotlib.pyplot as plt
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


import math


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

    # Calculate T1 and T2
    T1 = t1 * math.sin(Beta) + t2 * math.sin(Alpha)
    T2 = t1 * math.cos(Beta) + t2 * math.cos(Alpha)

    return T1, T2


# @icAll
def calculateCameraMovementOffset(cameraPoint1, real_point1, real_point2, cameraPoint2=None, cameraPoint3=None):
    # ic(cameraPoint1, cameraPoint2, cameraPoint3, real_point1, real_point2)
    cameraPoint2, cameraPoint3 = extrapolateManipulatorPosition(
        cameraPoint1, cameraPoint2
    ) if cameraPoint3 is None else (cameraPoint2, cameraPoint3)

    alpha = calculate_alpha(cameraPoint1, cameraPoint2)
    beta = calculate_beta(cameraPoint2, cameraPoint3)
    scalePixelInMilimeter = calculate_scale(cameraPoint1, cameraPoint2, real_point1, real_point2)
    ic(alpha, beta, scalePixelInMilimeter)
    cameraXOffset, cameraYOffset = calculate_movement(cameraPoint2[0], cameraPoint2[1], scalePixelInMilimeter, alpha,
                                                      beta)

    return cameraXOffset, cameraYOffset


def extrapolateManipulatorPosition(firstPoint, secondPoint=None, defaultInitialOffset=2.0):
    x1, y1 = firstPoint
    secondPoint = (x1 + defaultInitialOffset, y1 - defaultInitialOffset) if secondPoint is None else secondPoint
    x2, y2 = secondPoint
    # Calculate thirdPoint by rotating firstPoint around secondPoint by +90 degrees
    x3 = x2 + (y1 - y2)
    y3 = y2 - (x1 - x2)

    thirdPoint = (x3, y3)

    return secondPoint, thirdPoint


def visualize_points(set_3_points):
    """
    Visualizes three points on a Cartesian plane, creates arrows from the base point (B)
    to the other points (A and C) in the set of 3 points, with different colors for the lines and vectors.

    Parameters:
    set_3_points (list of tuple): List of 3 points (A, B, C) where B is the base point.
    """
    # Unzip the points into separate x and y coordinates for the set of 3 points
    x1, y1 = zip(*set_3_points)

    # Create the plot
    plt.figure(figsize=(6, 6))

    # Plot the set of 3 points and connect them with lines (no fill)
    plt.plot(x1[:2], y1[:2], marker='o', label='Y', color='blue', linestyle='-', markersize=8)  # AB in blue
    plt.plot(x1[1:], y1[1:], marker='o', label='X', color='red', linestyle='-', markersize=8)  # BC in red

    # Plot arrows from the base point (B) to the other points (A and C) in set_3_points
    B = set_3_points[1]  # Base point B is the second point in the set of 3 points (index 1)
    A, C = set_3_points[0], set_3_points[2]  # A is the first point, C is the third point

    # Plot vector AB (from B to A) in blue
    plt.arrow(B[0], B[1], A[0] - B[0], A[1] - B[1],
              head_width=0.2, head_length=0.3, fc='blue', ec='blue')

    # Plot vector BC (from B to C) in red
    plt.arrow(B[0], B[1], C[0] - B[0], C[1] - B[1],
              head_width=0.2, head_length=0.3, fc='red', ec='red')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Visualization of the Camera Calibration System')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.axis('equal')
    plt.show()
