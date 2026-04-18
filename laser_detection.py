import cv2
import os
import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, amplitude, mu, sigma, offset):
    return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset


def _enhance_laser(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _apply_luminosity_threshold(image, lower_threshold=100):
    _, result = cv2.threshold(image, lower_threshold, 255, cv2.THRESH_BINARY)
    return result


def find_laser_peaks(image, strip_width=20, step=10, min_peak=100):
    """
    Sweep the image left-to-right in vertical strips. For each strip whose
    peak intensity exceeds min_peak, fit a Gaussian within a ±200px band
    around the argmax to locate the laser peak at sub-pixel precision.

    Args:
        image:       CLAHE-enhanced, thresholded grayscale image
        strip_width: Width of each vertical strip in pixels
        step:        Horizontal increment between strips in pixels
        min_peak:    Min strip peak intensity to attempt a Gaussian fit

    Returns:
        List of (x, y) tuples from Gaussian fit
    """
    height, width = image.shape
    y_axis = np.arange(height)
    peaks = []

    for x in range(0, width - strip_width, step):
        strip = image[:, x:x + strip_width]
        profile = np.mean(strip, axis=1).astype(np.float64)

        peak_idx = int(np.argmax(profile))
        if profile[peak_idx] < min_peak:
            continue
        x_center = x + strip_width // 2

        band = 200
        y0 = max(0, peak_idx - band)
        y1 = min(height, peak_idx + band)
        profile_band = profile[y0:y1]
        y_band = y_axis[y0:y1]

        try:
            popt, _ = curve_fit(
                _gaussian, y_band, profile_band,
                p0=[float(profile_band.max()), float(peak_idx),
                    float(band) / 2, float(profile_band.min())],
                bounds=([0, y0, 1, 0], [np.inf, y1, band, np.inf]),
                maxfev=5000
            )
            mu = popt[1]
            if 0 <= mu < height:
                peaks.append((x_center, int(round(mu))))
        except (RuntimeError, ValueError):
            pass

    return peaks


def _fit_laser_line(peaks):
    if len(peaks) < 3:
        return {'params': None, 'inliers': peaks}

    xs = np.array([p[0] for p in peaks], dtype=np.float64)
    ys = np.array([p[1] for p in peaks], dtype=np.float64)

    inlier_mask = np.ones(len(xs), dtype=bool)
    for _ in range(2):
        slope, intercept = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
        residuals = np.abs(xs - (slope * ys + intercept))
        inlier_mask = residuals <= np.std(residuals[inlier_mask])
        if np.sum(inlier_mask) < 2:
            break

    if np.sum(inlier_mask) >= 2:
        slope, intercept = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
        inliers = [(int(xs[i]), int(ys[i])) for i in range(len(peaks)) if inlier_mask[i]]
    else:
        inliers = list(peaks)

    return {'params': (slope, intercept), 'inliers': inliers}


def _calculate_fit_angle(fit_result):
    if fit_result['params'] is None:
        return None
    slope = fit_result['params'][0]
    return np.degrees(np.arctan2(np.abs(slope), -np.sign(slope)))


def detect_primary_laser_angle(image_path, crop_tl, crop_br, img_thresh,
                               strip_width, peak_step, min_peak):
    """
    Detect the primary (incident) laser angle from a single image.
    Raises RuntimeError if detection fails (fewer than 3 peaks found).

    Returns:
        Angle in degrees (same convention as _calculate_fit_angle)
    """
    image = cv2.imread(image_path)
    x1, y1 = crop_tl
    x2, y2 = crop_br
    cropped     = image[y1:y2, x1:x2]
    enhanced    = _enhance_laser(cropped)
    thresholded = _apply_luminosity_threshold(enhanced, img_thresh)
    peaks       = find_laser_peaks(thresholded, strip_width=strip_width,
                                   step=peak_step, min_peak=min_peak)
    fit   = _fit_laser_line(peaks)
    angle = _calculate_fit_angle(fit)
    if angle is None:
        raise RuntimeError(
            f"Primary laser angle detection failed on {image_path} "
            f"({len(peaks)} peaks found, need >=3 for line fit)"
        )
    return angle


def method1_laser(image_path, image, frame_number, crop_tl, crop_br, img_thresh,
                  strip_width, peak_step, min_peak, primary_laser_angle,
                  dbg_dir, dbg_overlay):
    """
    Process a single frame using laser Gaussian peak detection.
    Computes the leaflet angle from mirror reflection geometry

    Args:
        image_path:           Path to image file
        frame_number:         Frame number
        crop_tl:              Top-left crop coordinate
        crop_br:              Bottom-right crop coordinate
        img_thresh:           Pixels below this are zeroed after CLAHE
        strip_width:          Width of each vertical strip for peak detection
        peak_step:            Horizontal step between strips
        min_peak:             Min strip peak intensity to consider a peak valid
        primary_laser_angle:  Incident laser angle in degrees (from detect_primary_laser_angle)
        dbg_dir:              Directory to save overlay images
        dbg_overlay:          If True, save annotated overlay image

    Returns:
        Dict with frame_number, filename, centroid=None, leftmost=None,
        angle (leaflet angle), method='laser'
    """
    x1, y1 = crop_tl
    x2, y2 = crop_br
    cropped = image[y1:y2, x1:x2]

    enhanced    = _enhance_laser(cropped)
    thresholded = _apply_luminosity_threshold(enhanced, img_thresh)
    peaks       = find_laser_peaks(thresholded, strip_width=strip_width,
                                   step=peak_step, min_peak=min_peak)

    image_height    = cropped.shape[0]
    fit_result      = _fit_laser_line(peaks)
    reflected_angle = _calculate_fit_angle(fit_result)

    if reflected_angle is not None:
        leaflet_angle = ((reflected_angle + primary_laser_angle) / 2) - 90
    else:
        leaflet_angle = None

    if dbg_dir and dbg_overlay:
        vis = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        for (x, y) in fit_result['inliers']:
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        if fit_result['params'] is not None:
            slope, intercept = fit_result['params']
            cv2.line(vis,
                     (int(intercept), 0),
                     (int(slope * image_height + intercept), image_height),
                     (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (_, text_h), _ = cv2.getTextSize("X", font, 0.7, 2)
        refl_str   = f"{reflected_angle:.2f} deg" if reflected_angle is not None else "N/A"
        leafl_str  = f"{leaflet_angle:.2f} deg"   if leaflet_angle  is not None else "N/A"
        for i, text in enumerate([
            f"Frame: {frame_number}",
            f"Peaks: {len(peaks)}",
            f"Primary  angle:  {primary_laser_angle:.2f} deg",
            f"Reflected angle: {refl_str}",
            f"Leaflet  angle:  {leafl_str}",
        ]):
            cv2.putText(vis, text, (10, text_h + 10 + i * (text_h + 6)),
                        font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(f"{dbg_dir}/dbg_{frame_number}_laser_overlay.png", vis)

    return {
        'frame_number': frame_number,
        'filename': os.path.basename(image_path),
        'centroid': None,
        'leftmost': None,
        'angle': leaflet_angle,
        'method': 'laser',
    }
