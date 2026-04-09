import cv2
import os
import numpy as np


def preprocess_image(image, threshold):
    """
    Convert BGR image to binary (grayscale -> blur -> threshold).
    Pixels below threshold -> white, above -> black.

    Returns:
        binary: thresholded image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def count_white_pixels(binary):
    """Return the number of white (255) pixels in a binary image."""
    return int(np.sum(binary == 255))


def fit_circle(points):
    """
    Fit a circle to a set of 2D points using algebraic least-squares.
    x² + y² + Dx + Ey + F = 0

    Args:
        points: np.ndarray of shape (N, 2)

    Returns:
        (cx, cy, r) or (None, None, None) if fewer than 3 points
    """
    if len(points) < 3:
        return None, None, None

    A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    b_vec = -(points[:, 0] ** 2 + points[:, 1] ** 2)
    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    D, E, F = result
    cx = -D / 2
    cy = -E / 2
    r = np.sqrt(cx ** 2 + cy ** 2 - F)
    return cx, cy, r


def compute_leaflet_angles(all_data, image_path_last, crop_tl, output_dir):
    """
    Fit a circle to the leftmost-pixel trajectory, fill in the angle field
    for all leaflet frames, and return the list of (frame_number, angle, method)
    tuples for all frames.

    Args:
        all_data: List of frame dicts (modified in-place)
        image_path_last: Path to the last input image
        crop_tl: Top-left crop coordinate
        output_dir: Directory to save the trajectory image

    Returns:
        List of (frame_number, angle, method) tuples
    """
    cx_fit, cy_fit, _ = fit_circle_trajectory(all_data, image_path_last, crop_tl, output_dir)

    for d in all_data:
        if d['method'] == 'leaflet' and d['leftmost'] is not None and cx_fit is not None:
            lft = d['leftmost']
            d['angle'] = np.degrees(np.arctan2(lft[1] - cy_fit, lft[0] - cx_fit)) % 360

    return [(d['frame_number'], d['angle'], d['method']) for d in all_data]


def fit_circle_trajectory(all_data, image_path_last, crop_tl, output_dir):
    """
    Fit a circle to the leftmost-pixel trajectory, draw it on the last input
    image alongside all centroid and leftmost points, and save the result.

    Args:
        all_data: List of frame dicts (must contain 'leftmost' and 'centroid')
        image_path_last: Path to the last input image (used as background)
        crop_tl: Top-left crop coordinate (offset for drawing on full image)
        output_dir: Directory to save the trajectory image

    Returns:
        (cx_fit, cy_fit, r_fit) or (None, None, None) if fit failed
    """
    lft_points = np.array([d['leftmost'] for d in all_data if d['leftmost'] is not None],
                           dtype=np.float64)
    cen_points = np.array([d['centroid'] for d in all_data if d['centroid'] is not None],
                           dtype=np.float64)

    cx_fit, cy_fit, r_fit = fit_circle(lft_points) if len(lft_points) >= 3 else (None, None, None)

    if cx_fit is not None:
        last_frame_num = all_data[-1]['frame_number']
        traj = cv2.imread(image_path_last).copy()
        ox, oy = crop_tl
        for pt in lft_points:
            cv2.circle(traj, (int(pt[0]) + ox, int(pt[1]) + oy), 2, (0, 255, 255), -1)
        for pt in cen_points:
            cv2.circle(traj, (int(pt[0]) + ox, int(pt[1]) + oy), 2, (0, 255, 0), -1)
        cv2.circle(traj, (int(cx_fit) + ox, int(cy_fit) + oy), int(r_fit), (0, 165, 255), 2)
        cv2.drawMarker(traj, (int(cx_fit) + ox, int(cy_fit) + oy), (0, 165, 255),
                       cv2.MARKER_CROSS, markerSize=16, thickness=2)
        traj_path = f"{output_dir}/frame_{last_frame_num}_circle_fit_trajectory.png"
        cv2.imwrite(traj_path, traj)
        print(f"  Circle fit: center=({cx_fit:.1f}, {cy_fit:.1f}), r={r_fit:.1f}. Saved: {traj_path}")
    else:
        print(f"  Not enough leftmost points for circle fit ({len(lft_points)} points).")

    return cx_fit, cy_fit, r_fit


def method2_leaflet(image_path, frame_number, crop_tl, crop_br,
                    output_dir=None, threshold=20,
                    dbg_overlay=True, dbg_threshold=True):
    """
    Process a single frame: threshold and compute centroid + leftmost white pixel.

    Args:
        image_path: Path to image file
        frame_number: Frame number
        crop_tl: Top-left crop coordinate
        crop_br: Bottom-right crop coordinate
        output_dir: Directory to save overlay images
        threshold: Binary threshold value
        dbg_overlay: If True, save centroid overlay image
        dbg_threshold: If True, save threshold visualization image

    Returns:
        Dict with frame_number, filename, centroid, leftmost,
        angle=None, method='leaflet'
    """
    image = cv2.imread(image_path)
    x1, y1 = crop_tl
    x2, y2 = crop_br
    image = image[y1:y2, x1:x2]

    binary_vis = preprocess_image(image, threshold)
    # Compute white-pixel centroid from thresholded image
    M = cv2.moments(binary_vis)
    if M['m00'] > 0:
        centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    else:
        centroid = None

    # Leftmost white pixel
    cols = np.where(binary_vis == 255)[1]
    if cols.size > 0:
        min_col = cols.min()
        rows_at_min_col = np.where(binary_vis[:, min_col] == 255)[0]
        leftmost = (int(min_col), int(rows_at_min_col[len(rows_at_min_col) // 2]))
    else:
        leftmost = None

    # Save threshold visualization
    if output_dir and dbg_threshold:
        cv2.imwrite(f"{output_dir}/frame_{frame_number}_threshold.png", binary_vis)

    # Save overlay with centroid and leftmost markers
    if output_dir and dbg_overlay:
        overlay = image.copy()
        if centroid:
            cv2.drawMarker(overlay, centroid, (0, 255, 0),
                           cv2.MARKER_CROSS, markerSize=20, thickness=2)
        if leftmost:
            cv2.drawMarker(overlay, leftmost, (0, 255, 255),
                           cv2.MARKER_CROSS, markerSize=20, thickness=2)
        text = f"Frame: {frame_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (_, text_h), _ = cv2.getTextSize(text, font, 0.84, 2)
        cv2.putText(overlay, text, (10, text_h + 10), font, 0.84,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(f"{output_dir}/frame_{frame_number}_overlay.png", overlay)

    return {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'centroid': centroid,
        'leftmost': leftmost,
        'angle': None,
        'method': 'leaflet',
    }
