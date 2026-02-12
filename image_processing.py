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


def find_laser_peaks(clahe_image, strip_height=5, step=30):
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

        # First fit
        slope, intercept = np.polyfit(ys, xs, 1)
        residuals = np.abs(xs - (slope * ys + intercept))

        # Reject outliers beyond 2 * std of residuals
        threshold = 2 * np.std(residuals)
        inlier_mask = residuals <= threshold

        # Refit on inliers
        if np.sum(inlier_mask) >= 2:
            slope, intercept = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
            inliers = [(int(xs[i]), int(ys[i])) for i in range(len(group)) if inlier_mask[i]]
        else:
            inliers = list(group)

        result[label] = {'params': (slope, intercept), 'inliers': inliers}

    return result


def visualize_fits(image, fit_result, image_height, output_path):
    """
    Draw linear fits and inlier peak points on top of the original image.

    Args:
        image: Original BGR image
        fit_result: Dictionary from fit_laser_lines
        image_height: Height of the image
        output_path: Path to save annotated image
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

    cv2.imwrite(output_path, annotated)


def process_frame(image_path, frame_number, intermediates_dir=None):
    """
    Process a single frame to enhance and edge-detect the laser.

    Args:
        image_path: Path to image file
        frame_number: Frame number for tracking
        intermediates_dir: If set, save intermediate images to this directory

    Returns:
        Dictionary with frame data
    """
    # Load image
    image = cv2.imread(image_path)

    # Save intermediate: original loaded image
    if intermediates_dir:
        cv2.imwrite(f"{intermediates_dir}/frame_{frame_number}_1_original.png", image)

    # Enhance laser
    _, enhance_steps = enhance_laser(image)

    # Save intermediate: grayscale and CLAHE images
    if intermediates_dir:
        cv2.imwrite(f"{intermediates_dir}/frame_{frame_number}_2_grayscale.png", enhance_steps['gray'])
        cv2.imwrite(f"{intermediates_dir}/frame_{frame_number}_3_clahe.png", enhance_steps['clahe'])

    # Find laser peaks via Gaussian fitting on horizontal strips
    peaks = find_laser_peaks(enhance_steps['clahe'])

    # Fit linear models on top and bottom halves, rejecting outliers
    image_height = image.shape[0]
    fit_result = fit_laser_lines(peaks, image_height)

    # Save intermediate: fitted lines and inlier points on original image
    if intermediates_dir:
        visualize_fits(image, fit_result, image_height,
                       f"{intermediates_dir}/frame_{frame_number}_4_fits.png")

    # Prepare frame data
    frame_data = {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'fit_result': fit_result,
    }

    return frame_data


def process_image_sequence(data_dir, output_dir, save_intermediates=False):
    """
    Process all frames in sequence.

    Args:
        data_dir: Directory containing image sequence
        output_dir: Output directory
        save_intermediates: If True, save intermediate images at each step
    """
    # Find all frames
    image_files = sorted(glob.glob(f"{data_dir}/Set_01_*.png"))

    # Create intermediates directory if needed
    intermediates_dir = f"{output_dir}/intermediates"
    if save_intermediates:
        os.makedirs(intermediates_dir, exist_ok=True)

    # Process each frame
    for _, image_path in enumerate(image_files):
        frame_num = int(image_path.split('_')[-1].split('.')[0])
        print(f"Processing frame {frame_num}...")

        process_frame(image_path, frame_num,
                      intermediates_dir=intermediates_dir if save_intermediates else None)

    print(f"\nProcessed {len(image_files)} frames")


if __name__ == "__main__":
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Configuration
    data_directory = "data"
    save_intermediates = True

    print("Laser Path Tracking System")
    print("=" * 50)
    print(f"Processing frames from: {data_directory}")
    print(f"Intermediates: {'Enabled' if save_intermediates else 'Disabled'}")
    print()

    # Process all frames
    process_image_sequence(data_directory, output_dir,
                           save_intermediates=save_intermediates)

    print("\nProcessing complete!")
    if save_intermediates:
        print(f"Intermediate images saved to: {output_dir}/intermediates/")
