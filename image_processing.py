import cv2
import os
import numpy as np
from sklearn.linear_model import RANSACRegressor

def perform_edge_detection(input_path, output_path, low_threshold=50, high_threshold=150):
    """
    Load an image, perform edge detection, and save the result.

    Args:
        input_path: Path to input image
        output_path: Path to save edge-detected image
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
    """
    # Read the image
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(f"Could not load image from {input_path}")

    print(f"Loaded image: {input_path}")
    print(f"Image shape: {image.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Save the result
    cv2.imwrite(output_path, edges)
    print(f"Edge detection completed. Saved to: {output_path}")

    return edges


def find_two_best_fit_lines(edges):
    """
    Find two lines of best fit from edge-detected image using linear regression.

    Args:
        edges: Binary edge image from Canny detection

    Returns:
        line1, line2: RANSAC regressor objects for two lines
    """
    # Get coordinates of all edge pixels (white pixels = 255)
    y_coords, x_coords = np.where(edges > 0)

    if len(x_coords) < 10:
        return None, None

    # Reshape for sklearn
    X = x_coords.reshape(-1, 1)
    y = y_coords

    # Find first line using RANSAC for robustness
    ransac1 = RANSACRegressor(random_state=42)
    ransac1.fit(X, y)

    # Get inlier mask and remove those points
    inlier_mask = ransac1.inlier_mask_
    outlier_mask = ~inlier_mask

    # Find second line from remaining points
    if np.sum(outlier_mask) > 10:
        X_remaining = X[outlier_mask]
        y_remaining = y[outlier_mask]

        ransac2 = RANSACRegressor(random_state=42)
        ransac2.fit(X_remaining, y_remaining)

        return ransac1, ransac2

    return ransac1, None


def draw_lines_on_edges(edges, line1, line2):
    """
    Draw two lines on edge image.

    Args:
        edges: Binary edge image
        line1, line2: RANSAC regressor objects

    Returns:
        Image with lines drawn
    """
    # Convert to color image for drawing colored lines
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    width = edges.shape[1]
    x_line = np.array([[0], [width]]).reshape(-1, 1)

    # Draw first line in red
    if line1 is not None:
        y_line1 = line1.predict(x_line)
        cv2.line(output, (0, int(y_line1[0])), (width, int(y_line1[1])), (0, 0, 255), 2)

    # Draw second line in green
    if line2 is not None:
        y_line2 = line2.predict(x_line)
        cv2.line(output, (0, int(y_line2[0])), (width, int(y_line2[1])), (0, 255, 0), 2)

    return output


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Input image path
    input_image = "data/Set_01_1820.png"

    # Use specific threshold for line detection
    low_threshold = 30
    high_threshold = 20

    # Perform edge detection
    output_edges = "output/Set_01_1820_edges_low30_high20.png"
    edges = perform_edge_detection(input_image, output_edges, low_threshold, high_threshold)

    # Find two best-fit lines
    print("Finding two best-fit lines...")
    line1, line2 = find_two_best_fit_lines(edges)

    # Draw lines on edge image
    edges_with_lines = draw_lines_on_edges(edges, line1, line2)

    # Save result
    output_with_lines = "output/Set_01_1820_edges_with_lines.png"
    cv2.imwrite(output_with_lines, edges_with_lines)

    print(f"\nLine detection complete!")
    print(f"Edge image: {output_edges}")
    print(f"Image with lines: {output_with_lines}")
