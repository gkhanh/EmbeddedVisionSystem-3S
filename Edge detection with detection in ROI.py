import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

Thickness_Lines_Edge = 3
Draw_Edge = True
Radius_Intersection = 10
Draw_Edge_Intersection = False
Thickness_Lines_ROI = 1
Draw_ROI = False
Thickness_Lines_Waveguide_Entrance = 3
Draw_Waveguide_Entrance = True


# Function to adjust contrast
def adjust_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Function to perform Canny edge detection and Hough Line transform
def detect_lines(image, min_stiffness_canny, max_stiffness_canny, threshold_Hough, minLineLenght_Hough, maxLineGap_Hough):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, min_stiffness_canny, max_stiffness_canny)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=threshold_Hough, minLineLength=minLineLenght_Hough, maxLineGap=maxLineGap_Hough)
    
    # Separate into horizontal and vertical lines
    horizontal_lines = [line[0] for line in lines if abs(line[0][3] - line[0][1]) < 20]  # Horizontal lines
    vertical_lines = [line[0] for line in lines if abs(line[0][2] - line[0][0]) < 20]    # Vertical lines

    return horizontal_lines, vertical_lines

# Function to fit RANSAC lines for both horizontal and vertical sets
def fit_ransac_lines(horizontal_lines, vertical_lines):
    slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical = None, None, None, None

    # Fit horizontal lines
    if len(horizontal_lines) > 0:
        horizontal_points = np.array([[x1, y1] for x1, y1, x2, y2 in horizontal_lines] + [[x2, y2] for x1, y1, x2, y2 in horizontal_lines])
        slope_horizontal, intercept_horizontal = fit_line_ransac(horizontal_points)

    # Fit vertical lines
    if len(vertical_lines) > 0:
        vertical_points = np.array([[x1, y1] for x1, y1, x2, y2 in vertical_lines] + [[x2, y2] for x1, y1, x2, y2 in vertical_lines])
        slope_vertical, intercept_vertical = fit_line_ransac(vertical_points)

    return slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical

# RANSAC line fitting function
def fit_line_ransac(points):
    if len(points) < 2:
        return None  # Not enough points to fit a line
    ransac = RANSACRegressor()
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac.fit(x, y)
    slope = ransac.estimator_.coef_[0]  # Slope (m)
    intercept = ransac.estimator_.intercept_  # Intercept (b)
    return slope, intercept

# Function to find the exact intersection of two lines given their slopes and intercepts
def find_exact_intersection(m1, b1, m2, b2):
    if m1 == m2:
        raise ValueError("The lines are parallel and have no intersection.")
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1
    return int(x_intersection), int(y_intersection)

# Function to create a rotated rectangular ROI under the horizontal line
def create_rotated_rectangle_roi(intersection, slope_horizontal, width=200, height=200, offset=811):
    # Starting point for the rectangle (offset based on the middle of the horizontal line)
    x_center = intersection[0] + offset
    y_center = int(slope_horizontal * (x_center - intersection[0]) + intersection[1])  # Adjust y based on slope

    # Calculate direction vector based on the slope of the horizontal line
    dx = width // 2
    dy = int(slope_horizontal * dx)

    # Instead of positioning the center, we will position the top of the rectangle under the line
    top_y = y_center + dy  # Positioning the top of the rectangle

    # Calculate the four corners of the rotated rectangle
    roi_corners = [
        (x_center - dx, top_y),  # Top-left corner
        (x_center + dx, top_y),  # Top-right corner
        (x_center + dx, top_y + height),  # Bottom-right corner
        (x_center - dx, top_y + height)   # Bottom-left corner
    ]
    
    return roi_corners

# Function to draw lines and the ROI on the image, including translating ROI lines to full image
def draw_lines_and_roi(image, slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical, intersection, roi_corners, vertical_lines_roi):
    construction_lines_image = image.copy()

    # Draw the horizontal construction line
    if slope_horizontal is not None and Draw_Edge:
        extended_horizontal_line = line_from_slope_intercept(slope_horizontal, intercept_horizontal, image.shape, axis='horizontal')
        x1, y1, x2, y2 = extended_horizontal_line
        cv2.line(construction_lines_image, (x1, y1), (x2, y2), (255, 0, 0), Thickness_Lines_Edge)  # Blue horizontal line

    # Draw the vertical construction line
    if slope_vertical is not None and Draw_Edge:
        extended_vertical_line = line_from_slope_intercept(slope_vertical, intercept_vertical, image.shape, axis='vertical')
        x1, y1, x2, y2 = extended_vertical_line
        cv2.line(construction_lines_image, (x1, y1), (x2, y2), (255, 0, 0), Thickness_Lines_Edge)  # Blue vertical line

    # Draw the intersection point
    if intersection is not None and Draw_Edge_Intersection:
        cv2.circle(construction_lines_image, intersection, Radius_Intersection, (0, 255, 0), -1)  # Green small dot

    # Draw the rotated ROI
    if roi_corners and Draw_ROI:
        for i in range(4):
            cv2.line(construction_lines_image, roi_corners[i], roi_corners[(i+1) % 4], (0, 255, 255), Thickness_Lines_ROI)  # Yellow 
            
    # Draw the vertical centerline from the ROI onto the full image
    if centerline_x is not None and roi_corners and Draw_Waveguide_Entrance:
        # Calculate the top-left corner of the ROI in the original image
        x_min = min([x for x, y in roi_corners])

        # Translate the centerline x-coordinate back to the full image
        centerline_x_translated = centerline_x + x_min

        # Draw the centerline on the original image
        cv2.line(construction_lines_image, (centerline_x_translated, 0), (centerline_x_translated, image.shape[0]), (0, 255, 0), Thickness_Lines_Waveguide_Entrance)  # Green centerline

    # Show the final image with ROI and vertical lines from ROI drawn on the full image
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(construction_lines_image, cv2.COLOR_BGR2RGB))
    plt.title("Full Image with ROI and Vertical Lines from ROI")
    plt.axis('off')
    plt.show()
    
# Function to convert slope and intercept to line endpoints
def line_from_slope_intercept(slope, intercept, image_shape, axis='horizontal'):
    height, width = image_shape[:2]
    if axis == 'horizontal':
        x1, x2 = 0, width
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        return np.array([x1, y1, x2, y2], dtype=np.int32)
    else:  # For vertical line, x is constant
        y1, y2 = 0, height
        x1 = int((y1 - intercept) / slope) if slope != 0 else intercept
        x2 = int((y2 - intercept) / slope) if slope != 0 else intercept
        return np.array([x1, y1, x2, y2], dtype=np.int32)
    
# Function to extract the ROI from the image based on the rectangle's corner points
def extract_roi(image, roi_corners):
    # Find bounding box around the rotated rectangle
    x_min = min([x for x, y in roi_corners])
    x_max = max([x for x, y in roi_corners])
    y_min = min([y for x, y in roi_corners])
    y_max = max([y for x, y in roi_corners])

    # Ensure ROI is within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    # Extract the ROI as a cropped section of the image
    roi_image = image[y_min:y_max, x_min:x_max]
    return roi_image

# Function to detect lines in the ROI
def detect_lines_in_roi(roi_image):
    
    # Adjust contrast for better edge detection
    roi_adjusted_image = adjust_contrast(roi_image, 1, 1)

    # Detect lines inside the ROI
    horizontal_lines, vertical_lines = detect_lines(roi_adjusted_image, 40, 50, 50, 50, 50)
    
    # Fit RANSAC lines in the ROI
    slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical = fit_ransac_lines(horizontal_lines, vertical_lines)

    return horizontal_lines, vertical_lines, slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical

# Function to fit and draw construction lines and the midpoint line
def draw_construction_lines_roi(roi_image, vertical_lines):
    # Copy the ROI for drawing
    roi_with_construction_lines = roi_image.copy()

    # To store the x positions of the two vertical construction lines
    vertical_x_positions = []

    # Loop through detected vertical lines and fit a RANSAC line to each
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        points = np.array([[x1, y1], [x2, y2]])

        # Handle vertical lines separately
        if x2 - x1 == 0:  # If the line is vertical
            vertical_x_positions.append(x1)  # Store the x-coordinate
        else:
            # Fit RANSAC line if it's not perfectly vertical
            slope, intercept = fit_line_ransac(points)

            # Calculate the endpoints for the extended construction line
            height = roi_with_construction_lines.shape[0]
            x1_extended = int((0 - intercept) / slope)  # Intersection with top of the ROI
            x2_extended = int((height - intercept) / slope)  # Intersection with bottom of the ROI

            # Store the x-coordinate of the construction line
            vertical_x_positions.append((x1_extended + x2_extended) // 2)

    # Calculate the midpoint line between the two vertical lines
    if len(vertical_x_positions) >= 2:
        mid_x = sum(vertical_x_positions) // len(vertical_x_positions)  # Average of x-positions

        # Draw the midpoint construction line
        cv2.line(roi_with_construction_lines, (mid_x, 0), (mid_x, roi_with_construction_lines.shape[0]), (255, 0, 0), 1)  # Blue midpoint line

        # Show the ROI with construction lines and the midpoint line
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(roi_with_construction_lines, cv2.COLOR_BGR2RGB))
        plt.title("Construction Lines and Midpoint Line in ROI")
        plt.axis('off')
        plt.show()
        
        return mid_x  # Return the centerline's x-coordinate for use in the full image
    return None

# Load the image
image_path = "./Pictures/foto6.png"  # Replace with your image path
image = cv2.imread(image_path)

# Adjust contrast for better edge detection
adjusted_image = adjust_contrast(image, 2, 1)

# Detect lines
horizontal_lines, vertical_lines = detect_lines(adjusted_image, 50, 60, 50, 50, 10)

# Fit RANSAC lines
slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical = fit_ransac_lines(horizontal_lines, vertical_lines)

# Get intersection point
if slope_horizontal is not None and slope_vertical is not None:
    intersection = find_exact_intersection(slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical)
    print(f"Intersection Point: {intersection}")
else:
    intersection = None

# Create the rotated ROI
if intersection is not None and slope_horizontal is not None:
    roi_corners = create_rotated_rectangle_roi(intersection, slope_horizontal)
    print(f"ROI Corners: {roi_corners}")
else:
    roi_corners = None

# If ROI is created, extract it and detect lines within it
if roi_corners:
    # Extract the ROI from the image
    roi_image = extract_roi(image, roi_corners)

    # Detect lines in the ROI
    horizontal_lines_roi, vertical_lines_roi, slope_horizontal_roi, intercept_horizontal_roi, slope_vertical_roi, intercept_vertical_roi = detect_lines_in_roi(roi_image)

    # Draw the detected construction lines in the ROI
    centerline_x = draw_construction_lines_roi(roi_image, vertical_lines_roi)
    
    # Draw the detected lines and the ROI on the original image
    draw_lines_and_roi(image, slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical, intersection, roi_corners, vertical_lines_roi)