import numpy as np
import matplotlib.pyplot as plt


class AxisMapper:
    def __init__(self, structured_mode=False):
        self.camera_points = []  # Points in the camera's axis map
        self.manipulator_points = []  # Corresponding points in the manipulator's axis map
        self.structured_mode = structured_mode  # Toggle for predefined pattern

    def add_points(self, camera_x, camera_y, manipulator_x, manipulator_y, move_y=None, move_x=None):
        """
        Add corresponding points from the camera's and manipulator's coordinate systems.
        In structured mode, derive the second and third points based on the specified or fetched distances.
        """
        self.camera_points.append((camera_x, camera_y))
        self.manipulator_points.append((manipulator_x, manipulator_y))

    def invert_transformation(self, scale, alpha, beta, t1, t2):
        """
        Invert the transformation from manipulator axis to camera axis.
        """
        inverted_points = []
        for x, y in self.manipulator_points:
            # Reverse translation
            x_translated = x - t1
            y_translated = y - t2

            # Reverse rotation
            x_rotated = x_translated * np.cos(np.radians(-beta)) - y_translated * np.sin(np.radians(-alpha))
            y_rotated = x_translated * np.sin(np.radians(-beta)) + y_translated * np.cos(np.radians(-alpha))

            # Reverse scaling
            x_camera = x_rotated / scale
            y_camera = y_rotated / scale

            inverted_points.append((x_camera, y_camera))
        return inverted_points

    # Calculate rotation and scaleCalculate the rotation angles alpha (Y-axis) and beta (X-axis) in degrees.
    def calculate_rotation(self):

        if len(self.camera_points) < 3:
            raise ValueError("Insufficient points to calculate rotation. Need exactly 3 points.")

            # Use only the first three points for calculations
        c1, c2, c3 = self.camera_points[:3]
        m1, m2, m3 = self.manipulator_points[:3]

        # Calculate alpha (Y-axis rotation)
        alpha = np.arctan2(c2[0] - c1[0], c2[1] - c1[1]) - np.arctan2(m2[0] - m1[0], m2[1] - m1[1])
        alpha = np.degrees(alpha)

        # Calculate beta (X-axis rotation)
        beta = np.arctan2(c3[0] - c2[0], c3[1] - c2[1]) - np.arctan2(m3[0] - m2[0], m3[1] - m2[1])
        beta = np.degrees(beta)

        return alpha, beta

    # Calculate the scale factor between the camera and manipulator coordinate systems.
    def calculate_scale(self):
        c1, c2 = self.camera_points[:2]
        m1, m2 = self.manipulator_points[:2]

        # Distance in camera's coordinate system
        camera_distance = np.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

        # Distance in manipulator's coordinate system
        manipulator_distance = np.sqrt((m2[0] - m1[0]) ** 2 + (m2[1] - m1[1]) ** 2)

        # Scale factor (mm per pixel)
        scale = manipulator_distance / camera_distance
        return scale

    # Calculate the translation vector to map camera origin to manipulator origin.
    def calculate_translation(self, scale, alpha, beta):
        c1 = self.camera_points[0]
        m1 = self.manipulator_points[0]

        # Scale and rotate the first camera point
        x_scaled = c1[0] * scale
        y_scaled = c1[1] * scale

        # Apply rotation
        x_rotated = x_scaled * np.cos(np.radians(beta)) - y_scaled * np.sin(np.radians(alpha))
        y_rotated = x_scaled * np.sin(np.radians(beta)) + y_scaled * np.cos(np.radians(alpha))

        # Calculate translation
        t1 = m1[0] - x_rotated
        t2 = m1[1] - y_rotated
        return t1, t2

    # Transform all camera points into the manipulator's coordinate system.
    def transform_points(self, scale, alpha, beta, t1, t2):
        transformed_points = []
        for x, y in self.camera_points:
            # Scale and rotate
            x_scaled = x * scale
            y_scaled = y * scale
            x_rotated = x_scaled * np.cos(np.radians(beta)) - y_scaled * np.sin(np.radians(alpha))
            y_rotated = x_scaled * np.sin(np.radians(beta)) + y_scaled * np.cos(np.radians(alpha))

            # Translate
            x_transformed = x_rotated + t1
            y_transformed = y_rotated + t2
            transformed_points.append((x_transformed, y_transformed))
        return transformed_points
