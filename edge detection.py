import cv2
import numpy as np
import matplotlib.pyplot as plt

Draw_detected_lines = True
thickness_lines = 1
thickness_intersection = 1
thickness_mid_points = 3

# Adjust contrast function (if needed)
def adjust_contrast(image, alpha=2, beta=1):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Function to filter lines within a 10-pixel range
def filter_lines_within_range(lines, axis='horizontal', threshold=10):
    if len(lines) == 0:
        return []
    
    if axis == 'horizontal':
        lines_sorted = sorted(lines, key=lambda line: line[1])  # Sort by y-coordinate for horizontal lines
    else:
        lines_sorted = sorted(lines, key=lambda line: line[0])  # Sort by x-coordinate for vertical lines
    
    # Start by taking the first line
    filtered_lines = [lines_sorted[0]]
    
    # Iterate through the sorted lines and filter them if they are within the threshold
    for line in lines_sorted[1:]:
        if axis == 'horizontal':
            if abs(line[1] - filtered_lines[-1][1]) <= threshold:
                filtered_lines.append(line)
        else:
            if abs(line[0] - filtered_lines[-1][0]) <= threshold:
                filtered_lines.append(line)
    
    return filtered_lines

# Function to fit a line using linear regression
def fit_line_linear_regression(points, axis='horizontal'):
    if len(points) == 0:
        return None
    if axis == 'horizontal':
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
    else:
        x = np.array([p[1] for p in points])
        y = np.array([p[0] for p in points])
    
    # Perform linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

# Convert the slope and intercept into a line (for horizontal or vertical)
def get_line_from_regression(slope, intercept, image_shape, axis='horizontal'):
    height, width = image_shape[:2]
    if axis == 'horizontal':
        x1, x2 = 0, width
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        return np.array([x1, y1, x2, y2], dtype=np.int32)
    else:
        y1, y2 = 0, height
        x1 = int(slope * y1 + intercept)
        x2 = int(slope * y2 + intercept)
        return np.array([x1, y1, x2, y2], dtype=np.int32)

# Function to calculate intersection between the two lines
def calculate_intersection(m1, b1, m2, b2):
    # Intersection of y = m1 * x + b1 and x = m2 * y + b2
    # Solve for x first:
    x_intersection = (m2 * b1 + b2) / (1 - m2 * m1)
    
    # Solve for y using the horizontal line equation
    y_intersection = m1 * x_intersection + b1
    return int(x_intersection), int(y_intersection)


# Load the image
# image_path = "C:/Users/patri/Documents/Proberen en testen/nieuwe fotos/foto6.png"  # Replace with your image path
image_path = './Pictures/foto2.png' # Replace with your image path
image = cv2.imread(image_path)

# Set the scaling factor
scale_factor = 2  # This will double the size of the image

# Get the original dimensions
height, width = image.shape[:2]

# Calculate new dimensions
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

# Upscale the image using different interpolation methods
# Choose the interpolation method: INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Convert the image to grayscale
gray = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

# Use Canny Edge Detection to identify edges in the image
edges = cv2.Canny(gray, 20, 50)

# Use Hough Line Transform with tuned parameters
lines = cv2.HoughLinesP(edges, rho=0.5, theta=np.pi/360, threshold=50, minLineLength=50, maxLineGap=10)

# Separate lines into horizontal and vertical categories
horizontal_lines = [line[0] for line in lines if abs(line[0][3] - line[0][1]) < 20]  # Horizontal lines
vertical_lines = [line[0] for line in lines if abs(line[0][2] - line[0][0]) < 20]    # Vertical lines

# Filter lines within 10 pixels of the average for both horizontal and vertical lines
filtered_horizontal_lines = filter_lines_within_range(horizontal_lines, axis='horizontal', threshold=10)
filtered_vertical_lines = filter_lines_within_range(vertical_lines, axis='vertical', threshold=10)

# Perform linear regression to fit the lines
if len(filtered_horizontal_lines) > 0:
    horizontal_points = [(x1, y1) for x1, y1, x2, y2 in filtered_horizontal_lines]
    slope_horizontal, intercept_horizontal = fit_line_linear_regression(horizontal_points, axis='horizontal')
    middle_horizontal_line = get_line_from_regression(slope_horizontal, intercept_horizontal, upscaled_image.shape, axis='horizontal')
else:
    middle_horizontal_line = None

if len(filtered_vertical_lines) > 0:
    vertical_points = [(x1, y1) for x1, y1, x2, y2 in filtered_vertical_lines]
    slope_vertical, intercept_vertical = fit_line_linear_regression(vertical_points, axis='vertical')
    middle_vertical_line = get_line_from_regression(slope_vertical, intercept_vertical, upscaled_image.shape, axis='vertical')
else:
    middle_vertical_line = None

# Draw midpoints and lines if requested
if Draw_detected_lines:
    # Draw horizontal midpoints as blue circles
    for x1, y1, x2, y2 in filtered_horizontal_lines:
        midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(upscaled_image, midpoint, thickness_mid_points, (255, 0, 0), -1)  # Blue circle for horizontal midpoints
    
    # Draw vertical midpoints as green circles
    for x1, y1, x2, y2 in filtered_vertical_lines:
        midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(upscaled_image, midpoint, thickness_mid_points, (0, 255, 0), -1)  # Green circle for vertical midpoints

# Create an image to show the final construction lines
construction_lines_image = upscaled_image.copy()

# Draw the horizontal and vertical lines based on linear regression
if middle_horizontal_line is not None:
    x1, y1, x2, y2 = middle_horizontal_line
    cv2.line(construction_lines_image, (x1, int(y1)), (x2, int(y2)), (255, 0, 0), thickness_lines)  # Blue line for horizontal

if middle_vertical_line is not None:
    x1, y1, x2, y2 = middle_vertical_line
    cv2.line(construction_lines_image, (int(x1), y1), (int(x2), y2), (0, 255, 0), thickness_lines)  # Green line for vertical

# Find and mark the intersection point
if middle_horizontal_line is not None and middle_vertical_line is not None:
    # Calculate the intersection of the two lines
    intersection = calculate_intersection(slope_horizontal, intercept_horizontal, slope_vertical, intercept_vertical)
    if intersection is not None:
        print(intersection)
        cv2.circle(construction_lines_image, intersection, thickness_intersection, (0, 0, 255), -1)  # Red circle for intersection

# Show the final image with construction lines and intersection point
plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(construction_lines_image, cv2.COLOR_BGR2RGB))
plt.title("Intersection Point (Red) with Construction Lines")
plt.axis('off')
plt.show()
