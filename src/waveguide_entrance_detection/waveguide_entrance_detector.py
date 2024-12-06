import cv2
import numpy as np
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# ---------- 1. Prepare Image ----------
def prepare_image(image, scale_factor=2):
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image_gu = upscale_image(image_g, scale_factor)  # Upscale grayscale image
    return image_gu

def upscale_image(image, scale_factor=2, interpolation_method=cv2.INTER_CUBIC):
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation_method)

# ---------- 2. Detect edges ----------

def detect_edges(image_gu):

    edges = cv2.Canny(image_gu, 20, 40)

    lines = cv2.HoughLinesP(edges, rho=0.2, theta=np.pi/720, threshold=50, minLineLength=50, maxLineGap=10)

    horizontal_lines, vertical_lines = devide_lines(lines)

    filtered_horizontal_lines = filter_lines(horizontal_lines, axis='horizontal')
    filtered_vertical_lines = filter_lines(vertical_lines, axis='vertical')

    horizontal_midpoints = get_midpoints(filtered_horizontal_lines)
    vertical_midpoints = get_midpoints(filtered_vertical_lines)

    horizontal_edge_chip = fit_line(horizontal_midpoints, axis='horizontal')
    vertical_edge_chip = fit_line(vertical_midpoints, axis='vertical')

    return horizontal_edge_chip, vertical_edge_chip

def devide_lines(lines):

    horizontal_lines = [line[0] for line in lines if abs(line[0][3] - line[0][1]) < 20]
    vertical_lines = [line[0] for line in lines if abs(line[0][2] - line[0][0]) < 20]

    return horizontal_lines, vertical_lines

def filter_lines(lines, axis='horizontal', threshold=10):
    """Filter lines within a certain pixel range."""
    if not lines:
        return []

    lines_sorted = sorted(lines, key=lambda line: line[1 if axis == 'horizontal' else 0])
    filtered_lines = [lines_sorted[0]]

    for line in lines_sorted[1:]:
        if abs(line[1 if axis == 'horizontal' else 0] - filtered_lines[-1][1 if axis == 'horizontal' else 0]) <= threshold:
            filtered_lines.append(line)

    return filtered_lines

def get_midpoints(lines):

    midpoints = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in lines]

    return midpoints

def fit_line(points, axis='horizontal'):
    """Fit a line using RANSAC."""
    if not points:
        return None

    x = np.array([p[0] for p in points]).reshape(-1, 1) if axis == 'horizontal' else np.array([p[1] for p in points]).reshape(-1, 1)
    y = np.array([p[1] for p in points]) if axis == 'horizontal' else np.array([p[0] for p in points])

    ransac = RANSACRegressor()
    ransac.fit(x, y)

    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    return slope, intercept

# ---------- 3. Extract ROI ----------

def get_ROI(horizontal_edge_chip, vertical_edge_chip, image_gu):

    intersection_edge_point = calculate_intersection(horizontal_edge_chip, vertical_edge_chip)

    ROI_corners = calculate_rotated_ROI(intersection_edge_point, horizontal_edge_chip)

    ROI_image, ROI_bbox = extract_ROI(ROI_corners, image_gu)

    return ROI_image, ROI_bbox, ROI_corners, intersection_edge_point

def calculate_intersection(horizontal_chip_edge, vertical_chip_edge):

    slope1 = horizontal_chip_edge[0]
    intercept1 = horizontal_chip_edge[1]
    slope2 = vertical_chip_edge[0]
    intercept2 = vertical_chip_edge[1]

    x, y = symbols('x y')

    line1 = Eq(y, slope1 * x + intercept1)
    line2 = Eq(x, slope2 * y + intercept2)

    solution = solve((line1, line2), (x, y))

    if not solution:
        return None  # No intersection

    return (int(round(solution[x])), int(round(solution[y])))


def calculate_rotated_ROI(intersection, horizontal_chip_edge, roi_width=400, above_distance=100, below_distance=400, horizontal_offset=1622):
    # Center point of the ROI
    x_center = intersection[0] + horizontal_offset
    y_center = intersection[1]  # y position of the horizontal line (intersection point)

    # Calculate angle of rotation based on the slope of the horizontal line
    angle = np.arctan(horizontal_chip_edge[0])  # Angle in radians

    # Calculate half-width (horizontal extension) and total height (vertical extension based on above and below distances)
    half_width = roi_width / 2

    # Calculate the four corners of the rotated ROI, starting from the center
    # We'll rotate around the center by the calculated angle
    top_left = (
        int(x_center - half_width * np.cos(angle) + above_distance * np.sin(angle)),
        int(y_center - half_width * np.sin(angle) - above_distance * np.cos(angle))
    )
    top_right = (
        int(x_center + half_width * np.cos(angle) + above_distance * np.sin(angle)),
        int(y_center + half_width * np.sin(angle) - above_distance * np.cos(angle))
    )
    bottom_left = (
        int(x_center - half_width * np.cos(angle) - below_distance * np.sin(angle)),
        int(y_center - half_width * np.sin(angle) + below_distance * np.cos(angle))
    )
    bottom_right = (
        int(x_center + half_width * np.cos(angle) - below_distance * np.sin(angle)),
        int(y_center + half_width * np.sin(angle) + below_distance * np.cos(angle))
    )

    return [top_left, top_right, bottom_right, bottom_left]

def extract_ROI(ROI_corners, image_gu):

    roi_points = np.array(ROI_corners, dtype=np.int32)

    x, y, w, h = cv2.boundingRect(roi_points)

    ROI_image = image_gu[y:y+h, x:x+w]

    return ROI_image, [x, y, w, h]

# ---------- 4. Detect waveguide lines ----------

def detect_waveguide_lines(ROI_image):

    EQ_ROI_image = cv2.equalizeHist(ROI_image)

    _, EQBW_ROI_image = cv2.threshold(EQ_ROI_image, 250, 255, cv2.THRESH_BINARY)

    edges_in_ROI = cv2.Canny(EQBW_ROI_image, 40, 50)

    lines_in_ROI = cv2.HoughLinesP(edges_in_ROI, rho=0.2, theta=np.pi/720, threshold=50, minLineLength=50, maxLineGap=10)

    horizontal_lines, vertical_lines = devide_lines(lines_in_ROI)

    vertical_midpoints = get_midpoints(vertical_lines)

    left_vertical_midpoints, right_vertical_midpoints = group_midpoints(vertical_midpoints, max_distance=2)

    left_vertical_line = fit_line(left_vertical_midpoints, axis='vertical')
    right_vertical_line = fit_line(right_vertical_midpoints, axis='vertical')

    return left_vertical_line, right_vertical_line

def group_midpoints(midpoints, max_distance=2):
    """Group midpoints into clusters based on horizontal proximity."""

    if not midpoints:
        return []

    # Sort midpoints by their x-coordinate
    midpoints_sorted = sorted(midpoints, key=lambda point: point[0])

    groups = []
    current_group = [midpoints_sorted[0]]

    # Iterate through the midpoints and group them by proximity
    for i in range(1, len(midpoints_sorted)):
        x_prev, y_prev = midpoints_sorted[i - 1]
        x_curr, y_curr = midpoints_sorted[i]

        # If the horizontal distance is less than or equal to max_distance, group the points
        if abs(x_curr - x_prev) <= max_distance:
            current_group.append(midpoints_sorted[i])
        else:
            # If the distance is larger, close the current group and start a new one
            groups.append(current_group)
            current_group = [midpoints_sorted[i]]

    # Add the last group
    groups.append(current_group)

    if len(groups) == 2:
        left_midpoints = groups[0]
        right_midpoints = groups[1]
    else:
        print('Waveguide entrance detection has ' + str(len(groups)) + ' groups')

    return left_midpoints, right_midpoints

# ---------- 5. Get waveguide entrance ----------

def get_waveguide_entrance(horizontal_chip_edge, left_vertical_line, right_vertical_line, ROI_bbox):

    left_OS_vertical_line = convert_line_to_original_scale(left_vertical_line, ROI_bbox)
    right_OS_vertical_line = convert_line_to_original_scale(right_vertical_line, ROI_bbox)

    left_point = calculate_intersection(horizontal_chip_edge, left_OS_vertical_line )
    right_point = calculate_intersection(horizontal_chip_edge, right_OS_vertical_line)

    waveguide_entrance_coordinates = calculate_middle(left_point, right_point)

    return left_OS_vertical_line, right_OS_vertical_line, waveguide_entrance_coordinates

def convert_line_to_original_scale(line, roi_bbox):

    slope = line[0]
    intercept = line[1]

    x_offset, y_offset, _, _ = roi_bbox

    adjusted_intercept = intercept + x_offset - (slope * y_offset)

    return slope, adjusted_intercept

def calculate_middle(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    midpoint_x = (x1 + x2) // 2
    midpoint_y = (y1 + y2) // 2

    print(midpoint_x, midpoint_y)

    return (midpoint_x, midpoint_y)

# ---------- 6. visualization ----------

def visualize_image(image, title="Image"):
    """Visualize the image using matplotlib."""
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for visualization
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_lines_hor_or_ver(image, lines, color=(255, 0, 0), thickness=2):
    image_with_lines = image.copy()

    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color, thickness)

    visualize_image(image_with_lines)

def draw_lines_hor_and_ver(image, horizontal_lines, vertical_lines, color_horizontal=(255, 0, 0), color_vertical=(0, 255, 0), thickness=2):

    image_with_lines = image.copy()

    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color_horizontal, thickness)

    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color_vertical, thickness)

    visualize_image(image_with_lines)

def draw_midpoints_hor_or_ver(image, vertical_midpoints, color_horizontal=(255, 0, 0), color_vertical=(0, 255, 0), radius=5, thickness=-1):

    image_with_midpoints = image.copy()

    for x_mid, y_mid in vertical_midpoints:
        cv2.circle(image_with_midpoints, (int(x_mid), int(y_mid)), radius, color_vertical, thickness)

    visualize_image(image_with_midpoints)

def draw_midpoints_hor_and_ver(image, horizontal_midpoints, vertical_midpoints, color_horizontal=(255, 0, 0), color_vertical=(0, 255, 0), radius=5, thickness=-1):

    image_with_midpoints = image.copy()

    for x_mid, y_mid in horizontal_midpoints:
        cv2.circle(image_with_midpoints, (int(x_mid), int(y_mid)), radius, color_horizontal, thickness)

    for x_mid, y_mid in vertical_midpoints:
        cv2.circle(image_with_midpoints, (int(x_mid), int(y_mid)), radius, color_vertical, thickness)

    visualize_image(image_with_midpoints)

def draw_RANSAC_edges(image, horizontal_edge_chip, vertical_edge_chip):

    slope_horizontal = horizontal_edge_chip[0]
    intercept_horizontal = horizontal_edge_chip[1]

    slope_vertical = vertical_edge_chip[0]
    intercept_vertical = vertical_edge_chip[1]

    height, width = image.shape[:2]

    x1, x2 = 0, width
    y1 = int(slope_horizontal * x1 + intercept_horizontal)
    y2 = int(slope_horizontal * x2 + intercept_horizontal)
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    y1, y2 = 0, height
    x1 = int(slope_vertical * y1 + intercept_vertical)
    x2 = int(slope_vertical * y2 + intercept_vertical)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image

def draw_rotated_roi(image, ROI_corners):

    image_with_ROI = image.copy()
    pts = np.array(ROI_corners, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image_with_ROI, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    return image_with_ROI

def draw_waveguide_entrance_lines(image, left_vertical_line, right_vertical_line):

    slope_vertical_left = left_vertical_line[0]
    intercept_vertical_left = left_vertical_line[1]

    slope_vertical_right = right_vertical_line[0]
    intercept_vertical_right = right_vertical_line[1]

    height, width = image.shape[:2]

    y1, y2 = 0, height
    x1 = int(slope_vertical_left * y1 + intercept_vertical_left)
    x2 = int(slope_vertical_left * y2 + intercept_vertical_left)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green vertical line

    y1, y2 = 0, height
    x1 = int(slope_vertical_right * y1 + intercept_vertical_right)
    x2 = int(slope_vertical_right * y2 + intercept_vertical_right)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green vertical line

    visualize_image(image)

def draw_all_on_image(image, horizontal_edge_chip, vertical_edge_chip, intersection_point, ROI_corners, vertical_lines_in_original_coords, waveguide_entrance_coordinates):

    image_with_drawings = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw lines
    image_with_drawings = draw_RANSAC_edges(image_with_drawings, horizontal_edge_chip, vertical_edge_chip)


    # Draw intersection point
    cv2.circle(image_with_drawings, intersection_point, 5, (0, 255, 0), -1)

    # Draw ROI
    image_with_drawings = draw_rotated_roi(image_with_drawings, ROI_corners)

    for slope, intercept in vertical_lines_in_original_coords:
        y1, y2 = 0, image.shape[0]  # From top to bottom
        x1 = int(slope * y1 + intercept)
        x2 = int(slope * y2 + intercept)
        cv2.line(image_with_drawings, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red lines for vertical lines in ROI

    cv2.circle(image_with_drawings, waveguide_entrance_coordinates, 2, (0, 255, 0), -1)  # Green circle for intersection

    return image_with_drawings