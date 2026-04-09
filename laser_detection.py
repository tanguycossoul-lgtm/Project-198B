import cv2
import os
import numpy as np
from scipy.optimize import curve_fit


def _enhance_laser(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


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
                  output_dir=None, luminosity_threshold=100,
                  strip_height=5, peak_step=10, dbg_overlay=True):
    """
    Process a single frame using laser Gaussian peak detection.

    Args:
        image_path: Path to image file
        frame_number: Frame number
        crop_tl: Top-left crop coordinate
        crop_br: Bottom-right crop coordinate
        output_dir: Directory to save overlay images
        luminosity_threshold: Pixels below this are zeroed after CLAHE
        strip_height: Height of each horizontal strip for peak detection
        peak_step: Vertical step between strips
        dbg_overlay: If True, save annotated overlay image

    Returns:
        Dict with frame_number, filename, centroid=None, leftmost=None,
        white_pixel_count=0, laser_angle
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

    if output_dir and dbg_overlay:
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
        cv2.imwrite(f"{output_dir}/frame_{frame_number}_overlay.png", annotated)

    return {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'centroid': None,
        'leftmost': None,
        'angle': laser_angle,
        'method': 'laser',
    }
