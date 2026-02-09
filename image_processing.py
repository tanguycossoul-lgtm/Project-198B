import cv2
import os
import numpy as np
import glob
import csv

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


def enhance_laser(image):
    """
    Enhance laser visibility for detection.

    Args:
        image: Input BGR image

    Returns:
        Enhanced grayscale image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return blurred


def detect_laser_segments(enhanced_image, min_length=50):
    """
    Detect laser line segments using Probabilistic Hough Transform.

    Args:
        enhanced_image: Preprocessed grayscale image
        min_length: Minimum line length to detect

    Returns:
        List of line segments [(x1, y1, x2, y2), ...]
    """
    # Apply Canny edge detection
    edges = cv2.Canny(enhanced_image, 50, 150)

    # Detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=10
    )

    if lines is None:
        return []

    # Convert to list of tuples
    segments = [(line[0], line[1], line[2], line[3])
                for line in lines[:, 0]]

    return segments


def calculate_segment_angle(x1, y1, x2, y2):
    """
    Calculate angle of line segment from horizontal (in degrees).

    Args:
        x1, y1: Start point
        x2, y2: End point

    Returns:
        Angle in degrees (-180 to 180)
    """
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def order_segments_from_entry(segments, entry_point):
    """
    Order laser segments starting from entry point.

    Args:
        segments: List of line segments [(x1,y1,x2,y2), ...]
        entry_point: (x, y) tuple of laser entry point

    Returns:
        Ordered list of segments tracing the laser path
    """
    if not segments:
        return []

    # Find segment closest to entry point
    def distance_to_segment(seg, point):
        x1, y1, x2, y2 = seg
        # Distance to both endpoints
        d1 = np.sqrt((x1 - point[0])**2 + (y1 - point[1])**2)
        d2 = np.sqrt((x2 - point[0])**2 + (y2 - point[1])**2)
        return min(d1, d2)

    # Sort by distance from entry point
    sorted_segments = sorted(segments,
                            key=lambda s: distance_to_segment(s, entry_point))

    return sorted_segments


def process_frame(image_path, entry_point, frame_number):
    """
    Process a single frame to detect laser and calculate angles.

    Args:
        image_path: Path to image file
        entry_point: (x, y) entry point coordinates
        frame_number: Frame number for tracking

    Returns:
        Dictionary with frame data (angles, segments, etc.)
    """
    # Load image
    image = cv2.imread(image_path)

    # Enhance laser
    enhanced = enhance_laser(image)

    # Detect segments
    segments = detect_laser_segments(enhanced)

    # Order segments from entry point
    ordered_segments = order_segments_from_entry(segments, entry_point)

    # Calculate angles for each segment
    angles = []
    for seg in ordered_segments:
        angle = calculate_segment_angle(*seg)
        angles.append(angle)

    # Prepare frame data
    frame_data = {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'entry_angle': angles[0] if angles else None,
        'all_angles': angles,
        'segments': ordered_segments,
        'num_segments': len(ordered_segments)
    }

    return frame_data


def visualize_laser_segments(image, segments, entry_point, output_path, show_angles=True):
    """
    Create annotated image with detected laser segments overlaid.

    Args:
        image: Original BGR image
        segments: List of detected line segments [(x1, y1, x2, y2), ...]
        entry_point: (x, y) entry point coordinates
        output_path: Path to save annotated image
        show_angles: Whether to annotate angles on segments
    """
    # Create a copy for annotation
    annotated = image.copy()

    # Draw entry point
    cv2.circle(annotated, entry_point, 10, (255, 0, 255), -1)  # Magenta circle
    cv2.putText(annotated, "Entry", (entry_point[0] + 15, entry_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Draw each segment with different colors
    for idx, seg in enumerate(segments):
        x1, y1, x2, y2 = seg

        # Color gradient: first segment (closest to entry) in green, others fade to blue
        if idx == 0:
            color = (0, 255, 0)  # Green for first segment (entry)
            thickness = 3
        elif idx < 5:
            color = (0, 255, 255)  # Cyan for next segments
            thickness = 2
        else:
            color = (255, 128, 0)  # Blue for remaining segments
            thickness = 2

        # Draw the line segment
        cv2.line(annotated, (x1, y1), (x2, y2), color, thickness)

        # Optionally add angle annotation
        if show_angles and idx < 10:  # Only annotate first 10 to avoid clutter
            angle = calculate_segment_angle(x1, y1, x2, y2)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(annotated, f"{angle:.1f}°", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Add segment count
    cv2.putText(annotated, f"Segments: {len(segments)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save the annotated image
    cv2.imwrite(output_path, annotated)


def process_image_sequence(data_dir, entry_point, output_csv, save_visualizations=False):
    """
    Process all frames in sequence and export angle data.

    Args:
        data_dir: Directory containing image sequence
        entry_point: (x, y) laser entry point
        output_csv: Path to output CSV file
        save_visualizations: If True, save annotated images showing detected segments
    """
    # Find all frames
    image_files = sorted(glob.glob(f"{data_dir}/Set_01_*.png"))

    all_frame_data = []

    # Create visualization directory if needed
    if save_visualizations:
        vis_dir = "output/visualizations"
        os.makedirs(vis_dir, exist_ok=True)

    # Process each frame
    for idx, image_path in enumerate(image_files):
        frame_num = int(image_path.split('_')[-1].split('.')[0])
        print(f"Processing frame {frame_num}...")

        frame_data = process_frame(image_path, entry_point, frame_num)
        all_frame_data.append(frame_data)

        # Save visualization if requested
        if save_visualizations:
            image = cv2.imread(image_path)
            vis_output = f"{vis_dir}/frame_{frame_num}_annotated.png"
            visualize_laser_segments(image, frame_data['segments'],
                                    entry_point, vis_output, show_angles=True)
            print(f"  Visualization saved to: {vis_output}")

    # Export to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Frame', 'Filename', 'Entry_Angle',
                        'Num_Segments', 'All_Angles'])

        # Data rows
        for data in all_frame_data:
            writer.writerow([
                data['frame_number'],
                data['filename'],
                f"{data['entry_angle']:.2f}" if data['entry_angle'] else 'N/A',
                data['num_segments'],
                ', '.join([f"{a:.2f}" for a in data['all_angles']])
            ])

    print(f"\nProcessed {len(all_frame_data)} frames")
    print(f"Data saved to: {output_csv}")

    return all_frame_data


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
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Configuration
    data_directory = "data"
    entry_point = (100, 500)  # User specifies (x, y) - UPDATE THIS
    output_csv = "output/laser_angles_over_time.csv"
    enable_visualizations = True  # Set to False to skip visualization export

    print("Laser Path Tracking System")
    print("=" * 50)
    print(f"Entry point: {entry_point}")
    print(f"Processing frames from: {data_directory}")
    print(f"Visualizations: {'Enabled' if enable_visualizations else 'Disabled'}")
    print()

    # Process all frames
    results = process_image_sequence(data_directory, entry_point, output_csv,
                                    save_visualizations=enable_visualizations)

    print("\nTracking complete!")
    print(f"Results saved to: {output_csv}")
    if enable_visualizations:
        print(f"Annotated images saved to: output/visualizations/")
