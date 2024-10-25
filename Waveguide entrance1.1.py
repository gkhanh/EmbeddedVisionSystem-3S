import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# ---------- 1. Prepare Image ----------
def prepare_image(image_path, scale_factor=2):
    """Load, grayscale, and upscale the image."""
    image = cv2.imread(image_path)  # Load image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    upscaled_image = upscale_image(gray, scale_factor)  # Upscale grayscale image
    return upscaled_image

def upscale_image(image, scale_factor=2, interpolation_method=cv2.INTER_CUBIC):
    """Upscale the image by a factor using specified interpolation."""
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation_method)

# ---------- 2. Detect Edges ----------

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

def get_lines(image, edges):
    """Detect and return the horizontal and vertical lines in the image."""
    
    lines = cv2.HoughLinesP(edges, rho=0.2, theta=np.pi/720, threshold=50, minLineLength=50, maxLineGap=10)
    #draw_all_lines(image, lines)
    
    # Separate horizontal and vertical lines
    horizontal_lines = [line[0] for line in lines if abs(line[0][3] - line[0][1]) < 20]
    vertical_lines = [line[0] for line in lines if abs(line[0][2] - line[0][0]) < 20]
    #draw_lines(image, horizontal_lines, vertical_lines)
    
    # Filter and fit lines
    filtered_horizontal_lines = filter_lines(horizontal_lines, axis='horizontal')
    filtered_vertical_lines = filter_lines(vertical_lines, axis='vertical')
    #draw_lines(image, filtered_horizontal_lines, filtered_vertical_lines)
    
    if filtered_horizontal_lines and filtered_vertical_lines:
        # Fit lines using RANSAC
        horizontal_midpoints = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in filtered_horizontal_lines]
        vertical_midpoints = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in filtered_vertical_lines]
        #draw_midpoints_hor_ver(image, horizontal_midpoints, vertical_midpoints)
        slope_horizontal, intercept_horizontal = fit_line(horizontal_midpoints, axis='horizontal')
        slope_vertical, intercept_vertical = fit_line(vertical_midpoints, axis='vertical')
        
        return slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical
    else:
        return None, None, None, None

# ---------- 3. Calculate ROI ----------

# Your original intersection function (unchanged)
def calculate_intersection(m1, b1, m2, b2):
    """Calculate the intersection point between two lines."""
    x_intersection = (m2 * b1 + b2) / (1 - m2 * m1)
    y_intersection = m1 * x_intersection + b1
    return int(x_intersection), int(y_intersection)

def calculate_rotated_roi(intersection, slope_horizontal, roi_width=200, above_distance=100, below_distance=100, horizontal_offset=811):
    """Calculate the rotated ROI coordinates based on intersection point and make it parallel to the horizontal line."""
    # Center point of the ROI
    x_center = intersection[0] + horizontal_offset
    y_center = intersection[1]  # y position of the horizontal line (intersection point)
    
    # Calculate angle of rotation based on the slope of the horizontal line
    angle = np.arctan(slope_horizontal)  # Angle in radians

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

# ---------- 4. Drawing Combined in One Image ----------
def draw_all_on_image(image, slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical, intersection_point, roi_corners):
    """Draw RANSAC lines, intersection point, and ROI on the image."""
    image_with_drawings = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
    
    # Draw lines
    image_with_drawings = draw_RANSAC_edges(image_with_drawings, slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical)

    # Draw intersection point
    cv2.circle(image_with_drawings, intersection_point, 5, (0, 255, 0), -1)  # Green circle for intersection

    # Draw ROI
    image_with_drawings = draw_rotated_roi(image_with_drawings, roi_corners)
    
    return image_with_drawings

def draw_RANSAC_edges(image, slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical):
    """Draw RANSAC lines"""
    height, width = image.shape[:2]
    
    # Horizontal line coordinates (from left to right)
    x1, x2 = 0, width
    y1 = int(slope_horizontal * x1 + intercept_horizontal)
    y2 = int(slope_horizontal * x2 + intercept_horizontal)
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue horizontal line

    # Vertical line coordinates (from top to bottom)
    y1, y2 = 0, height
    x1 = int(slope_vertical * y1 + intercept_vertical)
    x2 = int(slope_vertical * y2 + intercept_vertical)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green vertical line
    return image

def draw_RANSAC_ROI_lines(image, slope_vertical1, intercept_vertical1, slope_vertical2, intercept_vertical2):
    """Draw RANSAC lines"""
    height, width = image.shape[:2]

    # Vertical line coordinates (from top to bottom)
    y1, y2 = 0, height
    x1 = int(slope_vertical1 * y1 + intercept_vertical1)
    x2 = int(slope_vertical1 * y2 + intercept_vertical1)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green vertical line
    
    y1, y2 = 0, height
    x1 = int(slope_vertical2 * y1 + intercept_vertical2)
    x2 = int(slope_vertical2 * y2 + intercept_vertical2)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green vertical line
    
    visualize_image(image)

def draw_all_lines(image, lines, color=(255, 0, 0), thickness=2):
    image_with_lines = image.copy()
    
    # Draw each line from the Hough Transform output
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color, thickness)
    
    visualize_image(image_with_lines)

def draw_lines(image, horizontal_lines, vertical_lines, color_horizontal=(255, 0, 0), color_vertical=(0, 255, 0), thickness=2):
    """Draw the horizontal and vertical lines on the image."""
    image_with_lines = image.copy()
    
    # Draw horizontal lines in blue (or specified color)
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color_horizontal, thickness)
    
    # Draw vertical lines in green (or specified color)
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color_vertical, thickness)
    
    # Visualize the image with the drawn lines
    visualize_image(image_with_lines)
    
def draw_midpoints(image, vertical_midpoints, color_horizontal=(255, 0, 0), color_vertical=(0, 255, 0), radius=5, thickness=-1):
    """Draw the midpoints of the horizontal and vertical lines on the image."""
    image_with_midpoints = image.copy()
    
    # Draw vertical midpoints in green (or specified color)
    for x_mid, y_mid in vertical_midpoints:
        cv2.circle(image_with_midpoints, (int(x_mid), int(y_mid)), radius, color_vertical, thickness)
    
    visualize_image(image_with_midpoints)

def draw_midpoints_hor_ver(image, horizontal_midpoints, vertical_midpoints, color_horizontal=(255, 0, 0), color_vertical=(0, 255, 0), radius=5, thickness=-1):
    """Draw the midpoints of the horizontal and vertical lines on the image."""
    image_with_midpoints = image.copy()
    
    # Draw horizontal midpoints in blue (or specified color)
    for x_mid, y_mid in horizontal_midpoints:
        cv2.circle(image_with_midpoints, (int(x_mid), int(y_mid)), radius, color_horizontal, thickness)
    
    # Draw vertical midpoints in green (or specified color)
    for x_mid, y_mid in vertical_midpoints:
        cv2.circle(image_with_midpoints, (int(x_mid), int(y_mid)), radius, color_vertical, thickness)
    
    visualize_image(image_with_midpoints)

def draw_rotated_roi(image, roi_corners):
    """Draw the rotated ROI using the calculated corner points."""
    image_with_roi = image.copy()
    pts = np.array(roi_corners, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image_with_roi, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  # Red polygon for ROI
    return image_with_roi

# ----------detect in ROI----------

def change_contrast(image, alpha=0.2, beta=50):
    """Reduce contrast by scaling intensity and adding an offset."""
    changed_contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return changed_contrast_image

def detect_in_roi(image, roi_corners, scale_factor=1):
    """Perform edge detection and Hough transform within the specified ROI."""
    # Create a mask from the ROI polygon
    roi_points = np.array(roi_corners, dtype=np.int32)
    
    # Extract the bounding box of the ROI
    x, y, w, h = cv2.boundingRect(roi_points)
    
    # Crop the ROI from the image using the bounding box
    or_roi_image = image[y:y+h, x:x+w]
    
    roi_image = cv2.equalizeHist(or_roi_image)  # Apply histogram equalization
    
    plt.imshow(roi_image)
    
    _, thresholded = cv2.threshold(roi_image, 250, 255, cv2.THRESH_BINARY)
    
    plt.imshow(thresholded)
    
    # Detect edges in the ROI
    edges_in_roi = cv2.Canny(thresholded, 40, 50)
    
    # Perform Hough Line Transform within the ROI
    lines_in_roi = cv2.HoughLinesP(edges_in_roi, rho=0.2, theta=np.pi/720, threshold=50, minLineLength=50, maxLineGap=10)
    
    vertical_lines = [line[0] for line in lines_in_roi if abs(line[0][2] - line[0][0]) < 20]

    vertical_midpoints = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in vertical_lines]
    grouped_midpoints = group_midpoints_by_proximity(vertical_midpoints, max_distance=2)
    if len(grouped_midpoints) == 2:
        left_midpoints = grouped_midpoints[0]
        right_midpoints = grouped_midpoints[1]
    else:
        print('Waveguide entrance detection has ' + str(len(grouped_midpoints)) + ' groups')
    
    slope_vertical1, intercept_vertical1 = fit_line(left_midpoints, axis='vertical')
    slope_vertical2, intercept_vertical2 = fit_line(right_midpoints, axis='vertical')
    
    # Draw detected lines in the ROI for visualization
    if lines_in_roi is not None:
        or_roi_image = cv2.cvtColor(or_roi_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing
        draw_RANSAC_ROI_lines(or_roi_image, slope_vertical1, intercept_vertical1, slope_vertical2, intercept_vertical2)
        
        
def group_midpoints_by_proximity(midpoints, max_distance=5):
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
    
    return groups

# ---------- 5. Visualization Function ----------
def visualize_image(image, title="Image"):
    """Visualize the image using matplotlib."""
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for visualization
    plt.title(title)
    plt.axis('off')
    plt.show()

# ---------- Example Usage ----------
image_path = "C:/Users/patri/Documents/Proberen en testen/nieuwe fotos/foto5.png"
horizontal_distance = 1622  # Adjustable distance from intersection

# 1. Prepare Image
upscaled_image = prepare_image(image_path, scale_factor=2)

# 2. Detect Edges
edges = cv2.Canny(upscaled_image, 20, 40)

# 3. Get Horizontal and Vertical Lines
slope_h, intercept_h, slope_v, intercept_v = get_lines(upscaled_image, edges)

# 4. Calculate Intersection and ROI
if slope_h is not None and slope_v is not None:
    intersection_point = calculate_intersection(slope_h, intercept_h, slope_v, intercept_v)
    
    # Define custom above and below distances
    roi_width = 400
    above_distance = 100  # Distance above the horizontal line
    below_distance = 400   # Distance below the horizontal line
    
    roi_corners = calculate_rotated_roi(intersection_point, slope_h, roi_width=roi_width, above_distance=above_distance, below_distance=below_distance, horizontal_offset=horizontal_distance)

    # Draw everything on the image
    image_with_all = draw_all_on_image(upscaled_image, slope_h, intercept_h, slope_v, intercept_v, intersection_point, roi_corners)
    
    image_with_detection_in_roi = detect_in_roi(upscaled_image, roi_corners)
    
    # Visualize the final image
    visualize_image(image_with_all, title="Combined Visualization: Lines, Intersection, and ROI")


































