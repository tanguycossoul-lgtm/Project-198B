import csv
import cv2
import os
import glob
import shutil
import numpy as np
from scipy.optimize import curve_fit


# =============================================================================
# Method 1 — Laser Detection (copied from image_processing.py)
# =============================================================================

def _enhance_laser(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def _apply_luminosity_threshold(image, lower_threshold=100):
    result = image.copy()
    result[result < lower_threshold] = 0
    return result


def _gaussian(x, amplitude, mu, sigma, offset):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset


def find_laser_peaks(clahe_image, strip_height=5, step=10):
    """
    Sweep the image top-to-bottom in horizontal strips and fit a Gaussian
    to each strip's intensity profile to locate the laser peak.

    Args:
        clahe_image: CLAHE-enhanced grayscale image
        strip_height: Height of each horizontal strip in pixels
        step: Vertical increment between strips in pixels

    Returns:
        List of (x, y) tuples representing the Gaussian peak positions
    """
    height, width = clahe_image.shape
    x_axis = np.arange(width)
    peaks = []

    for y in range(0, height - strip_height, step):
        strip = clahe_image[y:y + strip_height, :]
        profile = np.mean(strip, axis=0).astype(np.float64)

        peak_idx = np.argmax(profile)
        a0 = float(profile[peak_idx])
        mu0 = float(peak_idx)
        sigma0 = 20.0
        offset0 = float(np.median(profile))

        try:
            popt, _ = curve_fit(
                _gaussian, x_axis, profile,
                p0=[a0, mu0, sigma0, offset0],
                maxfev=2000
            )
            mu = popt[1]
            if 0 <= mu < width:
                y_center = y + strip_height // 2
                peaks.append((int(round(mu)), y_center))
        except (RuntimeError, ValueError):
            continue

    return peaks


def _fit_laser_lines(peaks, image_height):
    mid_y = image_height // 2
    top_peaks = [(x, y) for x, y in peaks if y < mid_y]
    bottom_peaks = [(x, y) for x, y in peaks if y >= mid_y]

    result = {}
    for label, group in [('top', top_peaks), ('bottom', bottom_peaks)]:
        if len(group) < 3:
            result[label] = {'params': None, 'inliers': group}
            continue

        xs = np.array([p[0] for p in group], dtype=np.float64)
        ys = np.array([p[1] for p in group], dtype=np.float64)

        inlier_mask = np.ones(len(xs), dtype=bool)
        for _ in range(2):
            slope, intercept = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
            residuals = np.abs(xs - (slope * ys + intercept))
            inlier_mask = residuals <= np.std(residuals[inlier_mask])
            if np.sum(inlier_mask) < 2:
                break

        if np.sum(inlier_mask) >= 2:
            slope, intercept = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
            inliers = [(int(xs[i]), int(ys[i])) for i in range(len(group)) if inlier_mask[i]]
        else:
            inliers = list(group)

        result[label] = {'params': (slope, intercept), 'inliers': inliers}

    return result


def _calculate_fit_angle(fit_result):
    top_params = fit_result['top']['params']
    bottom_params = fit_result['bottom']['params']

    if top_params is None or bottom_params is None:
        return None

    slope_top = top_params[0]
    slope_bottom = bottom_params[0]

    dot = -slope_top * slope_bottom - 1
    mag_top = np.sqrt(slope_top**2 + 1)
    mag_bottom = np.sqrt(slope_bottom**2 + 1)

    cos_angle = np.clip(dot / (mag_top * mag_bottom), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def method1_laser(image_path, frame_number, crop_tl, crop_br,
                  vis_dir=None, luminosity_threshold=100,
                  strip_height=5, peak_step=10, save_overlay=True):
    """
    Process a single frame using laser Gaussian peak detection.

    Returns:
        Dict with frame_number, filename, centroid=None, leftmost=None,
        white_pixel_count, laser_angle
    """
    image = cv2.imread(image_path)
    x1, y1 = crop_tl
    x2, y2 = crop_br
    cropped = image[y1:y2, x1:x2]

    enhanced = _enhance_laser(cropped)
    thresholded = _apply_luminosity_threshold(enhanced, luminosity_threshold)
    peaks = find_laser_peaks(thresholded, strip_height=strip_height, step=peak_step)

    image_height = cropped.shape[0]
    fit_result = _fit_laser_lines(peaks, image_height)
    laser_angle = _calculate_fit_angle(fit_result)

    if vis_dir and save_overlay:
        annotated = cropped.copy()
        colors = {'top': (0, 0, 255), 'bottom': (255, 0, 0)}
        mid_y = image_height // 2
        y_ranges = {'top': (0, mid_y), 'bottom': (mid_y, image_height)}
        for label in ['top', 'bottom']:
            color = colors[label]
            data = fit_result[label]
            for (x, y) in data['inliers']:
                cv2.circle(annotated, (x, y), 5, color, -1)
            if data['params'] is not None:
                slope, intercept = data['params']
                y_start, y_end = y_ranges[label]
                cv2.line(annotated,
                         (int(slope * y_start + intercept), y_start),
                         (int(slope * y_end + intercept), y_end),
                         color, 2)
        text = f"Frame: {frame_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (_, text_h), _ = cv2.getTextSize(text, font, 0.84, 2)
        cv2.putText(annotated, text, (10, text_h + 10), font, 0.84,
                    (0, 255, 0), 2, cv2.LINE_AA)
        if laser_angle is not None:
            cv2.putText(annotated, f"Angle: {laser_angle:.2f} deg", (10, text_h + 40),
                        font, 0.84, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(f"{vis_dir}/frame_{frame_number}_overlay.png", annotated)

    return {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'centroid': None,
        'leftmost': None,
        'white_pixel_count': 0,
        'laser_angle': laser_angle,
    }


# =============================================================================
# Processing Pipeline
# =============================================================================

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


def method2_leaflet(image_path, frame_number, crop_tl, crop_br,
                    vis_dir=None, threshold=20,
                    save_overlay=True, save_threshold=True):
    """
    Process a single frame: threshold and compute centroid + leftmost white pixel.

    Args:
        image_path: Path to image file
        frame_number: Frame number
        crop_tl: Top-left crop coordinate
        crop_br: Bottom-right crop coordinate
        vis_dir: Directory to save overlay images
        threshold: Binary threshold value
        save_overlay: If True, save centroid overlay image
        save_threshold: If True, save threshold visualization image

    Returns:
        Dict with frame_number, filename, centroid, leftmost, white_pixel_count
    """
    image = cv2.imread(image_path)
    x1, y1 = crop_tl
    x2, y2 = crop_br
    image = image[y1:y2, x1:x2]

    binary_vis = preprocess_image(image, threshold)
    white_pixel_count = count_white_pixels(binary_vis)

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
    if vis_dir and save_threshold:
        cv2.imwrite(f"{vis_dir}/frame_{frame_number}_threshold.png", binary_vis)

    # Save overlay with centroid and leftmost markers
    if vis_dir and save_overlay:
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
        cv2.imwrite(f"{vis_dir}/frame_{frame_number}_overlay.png", overlay)

    return {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'centroid': centroid,
        'leftmost': leftmost,
        'white_pixel_count': white_pixel_count,
        'laser_angle': None,
    }


def process_sequence_2(data_dir, output_dir, crop_tl, crop_br,
                       frame_range, threshold=20, leaflet_sel="top",
                       save_overlay=True, save_threshold=True,
                       method2_leaflet_threshold=400):
    """
    Process a sequence of frames.

    Args:
        data_dir: Input directory with Set_01_*.png files
        output_dir: Root output directory
        crop_tl: Top-left crop coordinate
        crop_br: Bottom-right crop coordinate
        frame_range: (start, end) inclusive
        threshold: Binary threshold value
        leaflet_sel: "top" or "bot" leaflet selection
        save_overlay: If True, save centroid overlay images
        save_threshold: If True, save threshold visualization images
        method2_leaflet_threshold: Minimum white pixel count to run method2_leaflet
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

        # Pre-check white pixel count before full processing
        image = cv2.imread(image_path)
        x1, y1 = crop_tl
        x2, y2 = crop_br
        binary = preprocess_image(image[y1:y2, x1:x2], threshold)
        wpc = count_white_pixels(binary)

        if wpc < method2_leaflet_threshold:
            print(f"  method1_laser frame {frame_num} (white_pixel_count={wpc} < {method2_leaflet_threshold})")
            frame_data = method1_laser(image_path, frame_num,
                                       crop_tl, crop_br, vis_dir=vis_dir,
                                       save_overlay=save_overlay)
            frame_data['white_pixel_count'] = wpc
            all_data.append(frame_data)
            continue

        print(f"  method2_leaflet frame {frame_num}...")
        frame_data = method2_leaflet(image_path, frame_num,
                                     crop_tl, crop_br, vis_dir=vis_dir,
                                     threshold=threshold,
                                     save_overlay=save_overlay, save_threshold=save_threshold)
        all_data.append(frame_data)

    # Best-fit circle through leftmost-pixel trajectory
    lft_points = np.array([d['leftmost'] for d in all_data if d['leftmost'] is not None],
                           dtype=np.float64)
    cen_points = np.array([d['centroid'] for d in all_data if d['centroid'] is not None],
                           dtype=np.float64)
    cx_fit = cy_fit = r_fit = None
    if len(lft_points) >= 3:
        # Algebraic least-squares: x² + y² + Dx + Ey + F = 0
        A = np.column_stack([lft_points[:, 0], lft_points[:, 1], np.ones(len(lft_points))])
        b_vec = -(lft_points[:, 0] ** 2 + lft_points[:, 1] ** 2)
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        D, E, F = result
        cx_fit = -D / 2
        cy_fit = -E / 2
        r_fit = np.sqrt(cx_fit ** 2 + cy_fit ** 2 - F)

        # Draw on the full (uncropped) last input image
        last_frame_num = all_data[-1]['frame_number']
        traj = cv2.imread(image_files[-1]).copy()
        ox, oy = crop_tl  # offset: points are in cropped space
        # Draw all leftmost points (cyan)
        for pt in lft_points:
            cv2.circle(traj, (int(pt[0]) + ox, int(pt[1]) + oy), 2, (0, 255, 255), -1)
        # Draw all centroid points (green)
        for pt in cen_points:
            cv2.circle(traj, (int(pt[0]) + ox, int(pt[1]) + oy), 2, (0, 255, 0), -1)
        # Draw fitted circle and center
        cv2.circle(traj, (int(cx_fit) + ox, int(cy_fit) + oy), int(r_fit), (0, 165, 255), 2)
        cv2.drawMarker(traj, (int(cx_fit) + ox, int(cy_fit) + oy), (0, 165, 255),
                       cv2.MARKER_CROSS, markerSize=16, thickness=2)
        traj_path = f"{vis_dir}/frame_{last_frame_num}_circle_fit_trajectory.png"
        cv2.imwrite(traj_path, traj)
        print(f"  Circle fit: center=({cx_fit:.1f}, {cy_fit:.1f}), r={r_fit:.1f}. Saved: {traj_path}")
    else:
        print(f"  Not enough leftmost points for circle fit ({len(lft_points)} points).")

    # Write CSV (after circle fit so angles can be included)
    csv_path = f"{output_dir}/{leaflet_sel}/results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Centroid_x', 'Centroid_y', 'Leftmost_x', 'Leftmost_y',
                         'Centroid_angle_from_center_deg', 'Leftmost_angle_from_center_deg',
                         'White_pixel_count', 'Laser_angle'])
        for d in all_data:
            cen = d['centroid']
            lft = d['leftmost']
            cx_s = f"{cen[0]}" if cen else 'N/A'
            cy_s = f"{cen[1]}" if cen else 'N/A'
            lx_s = f"{lft[0]}" if lft else 'N/A'
            ly_s = f"{lft[1]}" if lft else 'N/A'
            if cen and cx_fit is not None:
                cen_angle_s = f"{np.degrees(np.arctan2(cen[1] - cy_fit, cen[0] - cx_fit)) % 360:.2f}"
            else:
                cen_angle_s = 'N/A'
            if lft and cx_fit is not None:
                lft_angle_s = f"{np.degrees(np.arctan2(lft[1] - cy_fit, lft[0] - cx_fit)) % 360:.2f}"
            else:
                lft_angle_s = 'N/A'
            laser_s = f"{d['laser_angle']:.2f}" if d['laser_angle'] is not None else 'N/A'
            writer.writerow([d['frame_number'], cx_s, cy_s, lx_s, ly_s, cen_angle_s, lft_angle_s,
                             d['white_pixel_count'], laser_s])

    print(f"  {len(all_data)} frames processed. CSV: {csv_path}")

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
    frame_range         = (0, 2000)              # Inclusive range of frame numbers to process
    crop_leaflet_top_tl = (1450, 380)       # Top-left (x, y) crop — top leaflet
    crop_leaflet_top_br = (1635, 660)       # Bottom-right (x, y) crop — top leaflet
    crop_leaflet_bot_tl = (1450, 660)       # Top-left (x, y) crop — bottom leaflet
    crop_leaflet_bot_br = (1615, 1240)      # Bottom-right (x, y) crop — bottom leaflet
    threshold                    = 20       # Pixels below this -> white, above -> black
    method2_leaflet_threshold    = 400      # Min white pixel count to process a frame
    save_overlay                 = True     # Save centroid overlay images
    save_threshold               = False    # Save threshold visualization images

    print("Leaflet Angle Measurement")
    print("=" * 50)
    print(f"Processing frames from: {data_directory}")
    print(f"Crop region (top): {crop_leaflet_top_tl} to {crop_leaflet_top_br}")
    print(f"Crop region (bot): {crop_leaflet_bot_tl} to {crop_leaflet_bot_br}")
    print(f"Frame range: {frame_range[0]} to {frame_range[1]}")
    print(f"Threshold: {threshold}")
    print(f"Method2 leaflet threshold: {method2_leaflet_threshold}")
    print()

#    print("--- Leaflet: top ---")
#    process_sequence_2(data_directory, output_dir,
#                       crop_tl=crop_leaflet_top_tl, crop_br=crop_leaflet_top_br,
#                       frame_range=frame_range, threshold=threshold,
#                       leaflet_sel="top",
#                       save_overlay=save_overlay, save_threshold=save_threshold,
#                       method2_leaflet_threshold=method2_leaflet_threshold)
#    print()

    print("--- Leaflet: bot ---")
    process_sequence_2(data_directory, output_dir,
                       crop_tl=crop_leaflet_bot_tl, crop_br=crop_leaflet_bot_br,
                       frame_range=frame_range, threshold=threshold,
                       leaflet_sel="bot",
                       save_overlay=save_overlay, save_threshold=save_threshold,
                       method2_leaflet_threshold=method2_leaflet_threshold)

    print("\nProcessing complete!")
