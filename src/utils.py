import numpy as np

def filter_lines_within_boundary(lines, axis='horizontal', min_length=150, margin=10):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if axis == 'horizontal' and abs(y1 - y2) < margin and length >= min_length:
            filtered_lines.append((x1, y1, x2, y2))
        elif axis == 'vertical' and abs(x1 - x2) < margin and length >= min_length:
            filtered_lines.append((x1, y1, x2, y2))
    return filtered_lines

def fit_line_linear_regression(points, axis='horizontal'):
    if len(points) == 0:
        return None
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
