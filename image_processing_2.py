import csv
import cv2
import os
import glob
import shutil
import numpy as np


# =============================================================================
# Shared Utilities
# =============================================================================

def order_corners(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape (4, 2)

    Returns:
        Ordered array of shape (4, 2)
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    # Sort by y first to get top pair and bottom pair
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    # Within each pair, sort by x
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def angle_from_corners(corners):
    """
    Compute rotation angle of a rectangle from its ordered corners.
    Uses the top edge (TL -> TR) to measure deviation from horizontal.

    Args:
        corners: Ordered array (4, 2) — TL, TR, BR, BL

    Returns:
        Angle in degrees (0 = horizontal, positive = counter-clockwise)
    """
    tl, tr = corners[0], corners[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    return np.degrees(np.arctan2(dy, dx))


def draw_rectangle_overlay(image, detections, frame_number=None):
    """
    Draw detected rectangle(s) corners and edges on an image.

    Args:
        image: BGR image
        detections: List of dicts with 'corners' (4,2) and 'angle' (float)
        frame_number: Optional frame number to stamp

    Returns:
        Annotated BGR image
    """
    annotated = image.copy()
    rect_colors = [(0, 255, 0), (0, 255, 255)]  # Green, Yellow for rect 1, 2

    for rect_idx, det in enumerate(detections):
        corners = det['corners']
        angle = det['angle']
        rc = rect_colors[rect_idx % len(rect_colors)]

        pts = np.int32(corners).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=rc, thickness=2)

        # Draw corner circles
        corner_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        for i, (x, y) in enumerate(np.int32(corners)):
            cc = corner_colors[i]
            cv2.circle(annotated, (x, y), 6, cc, -1)
            cv2.putText(annotated, f"R{rect_idx}:{i}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, cc, 1)

        # Angle text
        y_offset = 60 + rect_idx * 30
        angle_text = f"Rect {rect_idx}: {angle:.2f} deg"
        cv2.putText(annotated, angle_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc, 2)

    # Frame number
    if frame_number is not None:
        text = f"Frame: {frame_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.84
        thickness = 2
        (_, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(annotated, text, (10, text_h + 10), font, font_scale,
                    (0, 255, 0), thickness, cv2.LINE_AA)

    return annotated


# =============================================================================
# Detection: findContours + approxPolyDP
# =============================================================================

def detect_findcontours(image, bright_threshold=200, morph_kernel_size=7,
                        approx_epsilon=0.05, morph_iterations=3,
                        min_area=500, max_rects=2):
    """
    Detect up to max_rects white rectangles using contour approximation.

    1. Grayscale -> blur -> binary threshold (pixels below threshold -> white)
    2. Morphological close to fill gaps from partial obstruction
    3. findContours -> approxPolyDP to find 4-vertex polygons
    4. Fallback: minAreaRect on largest contours

    Args:
        image: BGR input image
        bright_threshold: Pixels below this value become white in the binary image
        morph_kernel_size: Kernel size for morphological closing
        approx_epsilon: approxPolyDP tolerance as fraction of perimeter
        morph_iterations: Number of morphological close iterations
        min_area: Minimum contour area to consider
        max_rects: Maximum number of rectangles to return (1 or 2)

    Returns:
        (results, binary_raw) where results is a list of dicts with
        'corners' and 'angle', and binary_raw is the thresholded image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold: pixels below threshold -> white, above -> black
    _, binary_raw = cv2.threshold(blurred, bright_threshold, 255, cv2.THRESH_BINARY_INV)
    binary = binary_raw.copy()

    # Morphological close to fill gaps from partial obstruction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (morph_kernel_size, morph_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                              iterations=morph_iterations)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect quadrilaterals found via approxPolyDP
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_epsilon * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append((area, approx))

    # Sort by area descending
    quads.sort(key=lambda x: x[0], reverse=True)

    results = []
    for area, quad in quads[:max_rects]:
        corners = order_corners(quad.reshape(4, 2))
        results.append({'corners': corners, 'angle': angle_from_corners(corners)})

    # Fallback: use minAreaRect on largest contours if no quads found
    if not results and contours:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours_sorted:
            if cv2.contourArea(cnt) < min_area:
                break
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            corners = order_corners(box)
            results.append({'corners': corners, 'angle': angle_from_corners(corners)})
            if len(results) >= max_rects:
                break

    return results, binary_raw


# =============================================================================
# Processing Pipeline
# =============================================================================

def process_frame_2(image_path, frame_number, crop_tl, crop_br,
                    vis_dir=None, threshold=20, leaflet_sel="top"):
    """
    Process a single frame with findContours detection.

    Args:
        image_path: Path to image file
        frame_number: Frame number
        crop_tl: Top-left crop coordinate
        crop_br: Bottom-right crop coordinate
        vis_dir: Directory to save overlay images
        threshold: Binary threshold value
        leaflet_sel: "top" or "bot" leaflet selection

    Returns:
        Dict with frame_number, filename, detections (list of dicts)
    """
    image = cv2.imread(image_path)
    x1, y1 = crop_tl
    x2, y2 = crop_br
    image = image[y1:y2, x1:x2]

    detections, binary_vis = detect_findcontours(image, bright_threshold=threshold)

    # Save threshold visualization
    if vis_dir:
        cv2.imwrite(f"{vis_dir}/frame_{frame_number}_threshold.png", binary_vis)

    # Save overlay
    if vis_dir and detections:
        overlay = draw_rectangle_overlay(image, detections,
                                         frame_number=frame_number)
        cv2.imwrite(f"{vis_dir}/frame_{frame_number}_overlay.png", overlay)

    return {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'detections': detections,
    }


def process_sequence_2(data_dir, output_dir, crop_tl, crop_br,
                       frame_range, threshold=20, leaflet_sel="top"):
    """
    Process a sequence of frames with findContours detection.

    Args:
        data_dir: Input directory with Set_01_*.png files
        output_dir: Root output directory
        crop_tl: Top-left crop coordinate
        crop_br: Bottom-right crop coordinate
        frame_range: (start, end) inclusive
        threshold: Binary threshold value
        leaflet_sel: "top" or "bot" leaflet selection
    """
    vis_dir = f"{output_dir}/{leaflet_sel}/visualisations"
    os.makedirs(vis_dir, exist_ok=True)

    image_files = sorted(glob.glob(f"{data_dir}/Set_01_*.png"))
    start, end = frame_range
    image_files = [f for f in image_files
                   if start <= int(f.split('_')[-1].split('.')[0]) <= end]

    all_data = []
    for image_path in image_files:
        frame_num = int(image_path.split('_')[-1].split('.')[0])
        print(f"  Processing frame {frame_num}...")

        frame_data = process_frame_2(image_path, frame_num,
                                     crop_tl, crop_br, vis_dir=vis_dir,
                                     threshold=threshold,
                                     leaflet_sel=leaflet_sel)
        all_data.append(frame_data)

    # Write CSV
    csv_path = f"{output_dir}/{leaflet_sel}/rectangles.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Rect', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'Angle'])
        for d in all_data:
            if d['detections']:
                for rect_idx, det in enumerate(d['detections']):
                    c = det['corners'].flatten()
                    writer.writerow([d['frame_number'], rect_idx,
                                     f"{c[0]:.1f}", f"{c[1]:.1f}",
                                     f"{c[2]:.1f}", f"{c[3]:.1f}",
                                     f"{c[4]:.1f}", f"{c[5]:.1f}",
                                     f"{c[6]:.1f}", f"{c[7]:.1f}",
                                     f"{det['angle']:.2f}"])
            else:
                writer.writerow([d['frame_number'], 0] + ['N/A'] * 9)

    detected = sum(1 for d in all_data if d['detections'])
    total_rects = sum(len(d['detections']) for d in all_data)
    print(f"  {detected}/{len(all_data)} frames detected ({total_rects} rects total). CSV: {csv_path}")

    return all_data


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Clear and recreate output directory
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Configuration
    data_directory = "data"
    frame_range = (1, 2000)         # Inclusive range of frame numbers to process
    crop_leaflet_top_tl = (1450, 380)  # Top-left (x, y) crop — top leaflet
    crop_leaflet_top_br = (1635, 660)  # Bottom-right (x, y) crop — top leaflet
    crop_leaflet_bot_tl = (1450, 660)  # Top-left (x, y) crop — bottom leaflet
    crop_leaflet_bot_br = (1635, 940)  # Bottom-right (x, y) crop — bottom leaflet
    threshold           = 20           # Pixels below this -> white, above -> black

    print("White Rectangle Detection — findContours")
    print("=" * 50)
    print(f"Processing frames from: {data_directory}")
    print(f"Crop region (top): {crop_leaflet_top_tl} to {crop_leaflet_top_br}")
    print(f"Crop region (bot): {crop_leaflet_bot_tl} to {crop_leaflet_bot_br}")
    print(f"Frame range: {frame_range[0]} to {frame_range[1]}")
    print(f"Threshold: {threshold}")
    print()

    print("--- Leaflet: top ---")
    process_sequence_2(data_directory, output_dir,
                       crop_tl=crop_leaflet_top_tl, crop_br=crop_leaflet_top_br,
                       frame_range=frame_range,
                       threshold=threshold, leaflet_sel="top")

    print()
    print("--- Leaflet: bot ---")
    process_sequence_2(data_directory, output_dir,
                       crop_tl=crop_leaflet_bot_tl, crop_br=crop_leaflet_bot_br,
                       frame_range=frame_range,
                       threshold=threshold, leaflet_sel="bot")

    print("\nProcessing complete!")
