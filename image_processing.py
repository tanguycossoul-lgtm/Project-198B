import csv
import cv2
import os
import glob
import numpy as np
from scipy.optimize import curve_fit


def enhance_laser(image):
    """
    Enhance laser visibility for detection.

    Args:
        image: Input BGR image

    Returns:
        Tuple of (enhanced, intermediates) where intermediates is a dict
        containing 'gray' and 'clahe' images
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    intermediates = {
        'gray': gray,
        'clahe': enhanced,
    }

    return enhanced, intermediates


def _gaussian(x, amplitude, mu, sigma, offset):
    """Gaussian function for curve fitting."""
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

        # Initial guesses
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
            # Only keep if the peak is within the image bounds
            if 0 <= mu < width:
                y_center = y + strip_height // 2
                peaks.append((int(round(mu)), y_center))
        except (RuntimeError, ValueError):
            continue

    return peaks


def fit_laser_lines(peaks, image_height):
    """
    Split peaks into top and bottom halves, reject outliers, and fit a line
    to each half. Fits x = slope*y + intercept since we sweep vertically.

    Args:
        peaks: List of (x, y) tuples
        image_height: Height of the image (used to split halves)

    Returns:
        Dictionary with 'top' and 'bottom' entries, each containing:
            'params': (slope, intercept) or None if fit failed
            'inliers': list of (x, y) inlier peaks
    """
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

        # Iterative outlier rejection (2 passes, 1*std threshold)
        inlier_mask = np.ones(len(xs), dtype=bool)
        for _ in range(2):
            slope, intercept = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
            residuals = np.abs(xs - (slope * ys + intercept))
            inlier_mask = residuals <= np.std(residuals[inlier_mask])
            if np.sum(inlier_mask) < 2:
                break

        # Final refit on surviving inliers
        if np.sum(inlier_mask) >= 2:
            slope, intercept = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
            inliers = [(int(xs[i]), int(ys[i])) for i in range(len(group)) if inlier_mask[i]]
        else:
            inliers = list(group)

        result[label] = {'params': (slope, intercept), 'inliers': inliers}

    return result


def calculate_fit_angle(fit_result):
    """
    Calculate the angle between the top and bottom linear fit segments.

    Since fits are x = slope*y + intercept, the direction vector of each
    line is (slope, 1). The angle between them is computed via the dot product.

    Args:
        fit_result: Dictionary from fit_laser_lines with 'top' and 'bottom' entries

    Returns:
        Angle in degrees between the two fit lines, or None if either fit is missing
    """
    top_params = fit_result['top']['params']
    bottom_params = fit_result['bottom']['params']

    if top_params is None or bottom_params is None:
        return None

    slope_top = top_params[0]
    slope_bottom = bottom_params[0]

    # Direction vectors: (slope, 1) for each line in (x, y) space
    dot = slope_top * slope_bottom + 1
    mag_top = np.sqrt(slope_top**2 + 1)
    mag_bottom = np.sqrt(slope_bottom**2 + 1)

    cos_angle = np.clip(dot / (mag_top * mag_bottom), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))

    return angle_deg


def stamp_frame_number(image, frame_number):
    """
    Draw frame number in top-left corner of an image (red).
    Uses same font as the angle overlay (scale 1.2, thickness 3).
    Works on both BGR and grayscale images.
    """
    is_gray = len(image.shape) == 2
    if is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    text = f"Frame: {frame_number}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.84
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(image, text, (10, text_h + 10), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return image


def visualize_fits(image, fit_result, image_height, output_path, frame_number=None):
    """
    Draw linear fits and inlier peak points on top of the original image.

    Args:
        image: Original BGR image
        fit_result: Dictionary from fit_laser_lines
        image_height: Height of the image
        output_path: Path to save annotated image
        frame_number: If set, stamp frame number on the image
    """
    annotated = image.copy()
    mid_y = image_height // 2

    colors = {'top': (0, 0, 255), 'bottom': (255, 0, 0)}  # Red / Blue
    y_ranges = {'top': (0, mid_y), 'bottom': (mid_y, image_height)}

    for label in ['top', 'bottom']:
        color = colors[label]
        data = fit_result[label]

        # Draw inlier points
        for (x, y) in data['inliers']:
            cv2.circle(annotated, (x, y), 5, color, -1)

        # Draw fitted line
        if data['params'] is not None:
            slope, intercept = data['params']
            y_start, y_end = y_ranges[label]
            x_start = int(slope * y_start + intercept)
            x_end = int(slope * y_end + intercept)
            cv2.line(annotated, (x_start, y_start), (x_end, y_end), color, 2)

    # Calculate and display the angle between the two fit lines
    angle = calculate_fit_angle(fit_result)
    if frame_number is not None:
        annotated = stamp_frame_number(annotated, frame_number)

    if angle is not None:
        angle_text = f"Angle: {angle:.2f} deg"
        cv2.putText(annotated, angle_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.84, (0, 255, 0), 2)
    else:
        cv2.putText(annotated, "Angle: N/A", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.84, (0, 0, 255), 2)

    cv2.imwrite(output_path, annotated)


def process_frame(image_path, frame_number, crop_tl=(500, 10), crop_br=(1450, 1450),
                   vis_dir=None, intermediates_dir=None, peak_step=10):
    """
    Process a single frame to enhance and edge-detect the laser.

    Args:
        image_path: Path to image file
        frame_number: Frame number for tracking
        crop_tl: Top-left (x, y) coordinate for cropping
        crop_br: Bottom-right (x, y) coordinate for cropping
        vis_dir: If set, save overlay images directly to this directory
        intermediates_dir: If set, save intermediate step images to this directory

    Returns:
        Dictionary with frame data
    """
    # Load image and crop to region of interest
    image = cv2.imread(image_path)
    x1, y1 = crop_tl
    x2, y2 = crop_br
    image = image[y1:y2, x1:x2]

    # Save intermediate: original loaded image
    if intermediates_dir:
        cv2.imwrite(f"{intermediates_dir}/frame_{frame_number}_1_original.png",
                     stamp_frame_number(image.copy(), frame_number))

    # Enhance laser
    _, enhance_steps = enhance_laser(image)

    # Save intermediate: grayscale and CLAHE images
    if intermediates_dir:
        cv2.imwrite(f"{intermediates_dir}/frame_{frame_number}_2_grayscale.png",
                     stamp_frame_number(enhance_steps['gray'].copy(), frame_number))
        cv2.imwrite(f"{intermediates_dir}/frame_{frame_number}_3_clahe.png",
                     stamp_frame_number(enhance_steps['clahe'].copy(), frame_number))

    # Find laser peaks via Gaussian fitting on horizontal strips
    peaks = find_laser_peaks(enhance_steps['clahe'], step=peak_step)

    # Fit linear models on top and bottom halves, rejecting outliers
    image_height = image.shape[0]
    fit_result = fit_laser_lines(peaks, image_height)

    # Save overlay: fitted lines and inlier points on original image
    if vis_dir:
        visualize_fits(image, fit_result, image_height,
                       f"{vis_dir}/frame_{frame_number}_overlay.png",
                       frame_number=frame_number)

    # Calculate angle between the two fit lines
    angle = calculate_fit_angle(fit_result)

    # Prepare frame data
    frame_data = {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'fit_result': fit_result,
        'angle': angle,
    }

    return frame_data


def process_image_sequence(data_dir, output_dir, crop_tl=(500, 10), crop_br=(1450, 1450),
                           frame_range=(1, 50), save_intermediates=False, peak_step=10):
    """
    Process all frames in sequence.

    Args:
        data_dir: Directory containing image sequence
        output_dir: Output directory
        crop_tl: Top-left (x, y) coordinate for cropping
        crop_br: Bottom-right (x, y) coordinate for cropping
        frame_range: (start, end) inclusive range of frame numbers to process
        save_intermediates: If True, save intermediate images at each step
    """
    # Find all frames and filter by frame_range
    image_files = sorted(glob.glob(f"{data_dir}/Set_01_*.png"))
    start, end = frame_range
    image_files = [f for f in image_files
                   if start <= int(f.split('_')[-1].split('.')[0]) <= end]

    # Create visualisations directories if needed
    vis_dir = f"{output_dir}/visualisations"
    intermediates_dir = f"{vis_dir}/intermediates"
    if save_intermediates:
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(intermediates_dir, exist_ok=True)

    # Process each frame
    all_frame_data = []
    for _, image_path in enumerate(image_files):
        frame_num = int(image_path.split('_')[-1].split('.')[0])
        print(f"Processing frame {frame_num}...")

        frame_data = process_frame(image_path, frame_num,
                      crop_tl=crop_tl, crop_br=crop_br,
                      vis_dir=vis_dir if save_intermediates else None,
                      intermediates_dir=intermediates_dir if save_intermediates else None,
                      peak_step=peak_step)
        all_frame_data.append(frame_data)

    # Export angles to CSV
    output_csv = f"{output_dir}/laser_angles.csv"
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Angle'])
        for data in all_frame_data:
            angle_str = f"{data['angle']:.2f}" if data['angle'] is not None else 'N/A'
            writer.writerow([data['frame_number'], angle_str])

    print(f"\nProcessed {len(all_frame_data)} frames (range {start}-{end})")
    print(f"Angles saved to: {output_csv}")


if __name__ == "__main__":
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Configuration
    data_directory = "data"
    crop_tl = (500, 5)      # Top-left (x, y) crop coordinate
    crop_br = (1450, 1450)   # Bottom-right (x, y) crop coordinate
    frame_range = (760, 780)    # Inclusive range of frame numbers to process
    save_intermediates = True
    peak_step = 10               # Vertical step between horizontal strips

    print("Laser Path Tracking System")
    print("=" * 50)
    print(f"Processing frames from: {data_directory}")
    print(f"Crop region: {crop_tl} to {crop_br}")
    print(f"Frame range: {frame_range[0]} to {frame_range[1]}")
    print(f"Intermediates: {'Enabled' if save_intermediates else 'Disabled'}")
    print()

    # Process all frames
    process_image_sequence(data_directory, output_dir,
                           crop_tl=crop_tl, crop_br=crop_br,
                           frame_range=frame_range,
                           save_intermediates=save_intermediates,
                           peak_step=peak_step)

    print("\nProcessing complete!")
    if save_intermediates:
        print(f"Overlays saved to: {output_dir}/visualisations/")
        print(f"Intermediates saved to: {output_dir}/visualisations/intermediates/")

        # Convert overlay images to video
        from convert_png_to_video import convert_pngs_to_video
        overlay_video = f"{output_dir}/overlay_video_{frame_range[0]}-{frame_range[1]}.mp4"
        convert_pngs_to_video(
            f"{output_dir}/visualisations", overlay_video,
            frame_range=frame_range,
            glob_pattern="frame_*_overlay.png",
            frame_regex=r'frame_(\d+)_overlay\.png$',
            stamp_frames=False
        )
