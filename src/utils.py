import numpy as np
from sklearn.linear_model import RANSACRegressor


def filter_lines_within_boundary(lines, axis='horizontal', min_length=150, margin=10):
    filtered_lines = []
    if lines is None:  # Check for NoneType to avoid errors
        return filtered_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if axis == 'horizontal' and abs(y1 - y2) < margin and length >= min_length:
            filtered_lines.append((x1, y1, x2, y2))
        elif axis == 'vertical' and abs(x1 - x2) < margin and length >= min_length:
            filtered_lines.append((x1, y1, x2, y2))

    print(f"Filtered {len(filtered_lines)} {axis} lines")
    return filtered_lines


def fit_line_linear_regression(points, axis='horizontal'):
    if len(points) < 2:  # Ensure enough points for regression
        return None, None

    if axis == 'horizontal':
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
    else:
        x = np.array([p[1] for p in points])
        y = np.array([p[0] for p in points])

    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b


def get_line_from_regression(slope, intercept, image_shape, axis='horizontal', offset=0):
    if slope is None or intercept is None:  # Check for invalid regression
        return None

    height, width = image_shape[:2]
    if axis == 'horizontal':
        x1, x2 = 0, width
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        return np.array([x1, y1, x2, y2], dtype=np.int32)
    else:
        y1, y2 = 0, height
        x1 = int(slope * y1 + intercept) + offset
        x2 = int(slope * y2 + intercept) + offset
        return np.array([x1, y1, x2, y2], dtype=np.int32)

def fit_line_ransac(points, axis='horizontal'):
    if len(points) < 2:
        return None  # Not enough points to fit a line
    ransac = RANSACRegressor()
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac.fit(x, y)
    slope = ransac.estimator_.coef_[0]  # Slope (m)
    intercept = ransac.estimator_.intercept_  # Intercept (b)
    return slope, intercept