import csv
import cv2
import os
import glob
import shutil
from smooth_results import smooth_results

from laser_detection import detect_primary_laser_angle, method1_laser
from leaflet_detection import preprocess_image, count_white_pixels, method2_leaflet, compute_leaflet_angles


def reset_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)



def write_results_csv(results, results_smooth, smooth_cutoff, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Angle', f'Angle ({smooth_cutoff})', 'Method', 'Angle (laser)', 'Angle (leaflet)'])
        for (frame_number, angle, method), (_, angle_smooth, _) in zip(results, results_smooth):
            angle_s        = f"{angle:.2f}"        if angle        is not None else 'N/A'
            angle_smooth_s = f"{angle_smooth:.2f}" if angle_smooth is not None else 'N/A'
            laser_angle    = angle_s if method == 'laser'   else ''
            leaflet_angle  = angle_s if method == 'leaflet' else ''
            writer.writerow([frame_number, angle_s, angle_smooth_s, method, laser_angle, leaflet_angle])


def process_sequence(data_dir, dbg_dir,
                     frame_range,
                     method_select_thresh,
                     img_laser_tl, img_laser_br,
                     img_laser_thresh, laser_strip_width, laser_peak_step, laser_min_peak,
                     primary_laser_angle,
                     img_leafl_tl, img_leafl_br,
                     img_leafl_thresh,
                     leaflet_calib_frame_range=None,
                     dbg_overlay=True, dbg_threshold=True):
    """
    Process a sequence of frames, dispatching to method1_laser or method2_leaflet
    based on the white pixel count threshold.

    Args:
        data_dir: Input directory with Set_01_*.png files
        dbg_dir: Directory to save debug images
        frame_range: (start, end) inclusive
        method_select_thresh: Min white pixel count to use method2_leaflet;
                              frames below this use method1_laser
        img_laser_tl: Top-left crop coordinate for method1_laser
        img_laser_br: Bottom-right crop coordinate for method1_laser
        img_laser_thresh: Intensity threshold for method1_laser
        laser_strip_width: Width of each vertical strip for centroid detection
        laser_peak_step: Vertical step between strips
        laser_min_peak: Min strip peak intensity to consider a peak valid
        img_leafl_tl: Top-left crop coordinate for method2_leaflet
        img_leafl_br: Bottom-right crop coordinate for method2_leaflet
        img_leafl_thresh: Binary threshold value for leaflet detection
        dbg_overlay: If True, save overlay images
        dbg_threshold: If True, save threshold visualization images
    """
    image_files = sorted(glob.glob(f"{data_dir}/Set_01_*.png"))
    start, end = frame_range
    image_files = [f for f in image_files
                   if start <= int(f.split('_')[-1].split('.')[0]) <= end]

    all_data             = []
    crossing_laser_angle = None
    crossing_frame_num   = None
    first_crossing_done  = False
    prev_method          = None
    last_leaflet_path    = None
    last_leaflet_frame   = None

    for img_path in image_files:
        frame_num = int(img_path.split('_')[-1].split('.')[0])

        # Pre-check white pixel count (using leaflet crop) to select detection method
        image = cv2.imread(img_path)
        x1, y1 = img_leafl_tl
        x2, y2 = img_leafl_br
        binary = preprocess_image(image[y1:y2, x1:x2], img_leafl_thresh)
        wpc    = count_white_pixels(binary)

        if wpc < method_select_thresh:
            print(f"  method1_laser  frame {frame_num} (white_pixel_count={wpc})")
            frame_data = method1_laser(img_path, image, frame_num,
                                       img_laser_tl, img_laser_br,
                                       img_laser_thresh,
                                       laser_strip_width, laser_peak_step, laser_min_peak,
                                       primary_laser_angle,
                                       dbg_dir, dbg_overlay)
            # leaflet→laser transition: calibrate on the last leaflet frame
            if not first_crossing_done and prev_method == 'leaflet':
                first_crossing_done = True
                crossing_frame_num  = last_leaflet_frame
                cal_image = cv2.imread(last_leaflet_path)
                laser_data = method1_laser(last_leaflet_path, cal_image, last_leaflet_frame,
                                           img_laser_tl, img_laser_br,
                                           img_laser_thresh,
                                           laser_strip_width, laser_peak_step, laser_min_peak,
                                           primary_laser_angle,
                                           dbg_dir, False)
                crossing_laser_angle = laser_data['angle']
            prev_method = 'laser'
        else:
            print(f"  method2_leaflet frame {frame_num} (white_pixel_count={wpc})")
            frame_data = method2_leaflet(img_path, image, frame_num,
                                         img_leafl_tl, img_leafl_br,
                                         img_thresh=img_leafl_thresh,
                                         dbg_dir=dbg_dir,
                                         dbg_overlay=dbg_overlay,
                                         dbg_threshold=dbg_threshold)
            # laser→leaflet transition: calibrate on this frame
            if not first_crossing_done and prev_method == 'laser':
                first_crossing_done = True
                crossing_frame_num  = frame_num
                laser_data = method1_laser(img_path, image, frame_num,
                                           img_laser_tl, img_laser_br,
                                           img_laser_thresh,
                                           laser_strip_width, laser_peak_step, laser_min_peak,
                                           primary_laser_angle,
                                           dbg_dir, False)
                crossing_laser_angle = laser_data['angle']
            last_leaflet_path  = img_path
            last_leaflet_frame = frame_num
            prev_method = 'leaflet'
        all_data.append(frame_data)

    results = compute_leaflet_angles(all_data, image_files[-1], img_leafl_tl, dbg_dir,
                                     leaflet_calib_frame_range)

    if crossing_laser_angle is not None and crossing_frame_num is not None:
        crossing_leaflet_angle = next(
            (angle for fn, angle, method in results
             if fn == crossing_frame_num and method == 'leaflet' and angle is not None),
            None
        )
        if crossing_leaflet_angle is not None:
            correction = crossing_leaflet_angle - crossing_laser_angle
            print(f"  Laser correction: {correction:.2f} deg (frame {crossing_frame_num})")
            results = [
                (fn, angle + correction if method == 'laser' and angle is not None else angle, method)
                for fn, angle, method in results
            ]

    print(f"  {len(all_data)} frames processed.")
    return results


if __name__ == "__main__":

    # Configuration
    data_dir    = "data"
    output_dir  = "output"
    dbg_dir     = "dbg"  # Subdirectory for debug images
    csv_path    = f"{output_dir}/results.csv"

    frame_range          = (0, 2000)   # Inclusive range of frame numbers to process
 #  frame_range          = (0, 200)   # Inclusive range of frame numbers to process
    method_select_thresh = 400          # Min white pixel count to use method2_leaflet
    dbg_overlay            = False        # Save overlay images
    dbg_threshold        = False        # Save threshold visualization images
    smooth_cutoff        = 0.1         # Low-pass filter cutoff (fraction of Nyquist, 0.0-1.0)

    # Method 1 config
    img_laser_prim_tl   = (1200, 5)      # Top-left (x, y) crop coordinate
    img_laser_prim_br   = (1800, 1000)  # Bottom-right (x, y) crop coordinate
    img_laser_tl        = (500, 5)      # Top-left (x, y) crop coordinate
    img_laser_br        = (1200, 1650)  # Bottom-right (x, y) crop coordinate
    img_laser_thresh    = 90            # Pixels below this -> white, above -> black
    laser_strip_width   = 10            # Width of each vertical strip for centroid detection
    laser_peak_step     = 10            # Vertical step between strips
    laser_min_peak      = 100           # Min strip peak intensity to consider a peak valid

    # Method 2 config
    img_leafl_bot_tl            = (1450, 660)      # Top-left (x, y) crop — bottom leaflet
    img_leafl_bot_br            = (1615, 1240)     # Bottom-right (x, y) crop — bottom leaflet
    img_leafl_thresh            = 20               # Pixels below this -> white, above -> black
    leaflet_calib_frame_range   = (0, 200)          # Frame range used to fit the circle

    reset_dirs(output_dir, dbg_dir)

    first_frame = sorted(glob.glob(f"{data_dir}/Set_01_*.png"))[0]
    primary_laser_angle = detect_primary_laser_angle(
        first_frame,
        img_laser_prim_tl, img_laser_prim_br,
        img_laser_thresh, laser_strip_width, laser_peak_step, laser_min_peak)
    print(f"Primary laser angle: {primary_laser_angle:.2f} deg")

    results = process_sequence(data_dir, dbg_dir,
                               frame_range=frame_range,
                               method_select_thresh=method_select_thresh,
                               img_laser_tl=img_laser_tl, img_laser_br=img_laser_br,
                               img_laser_thresh=img_laser_thresh,
                               laser_strip_width=laser_strip_width, laser_peak_step=laser_peak_step, laser_min_peak=laser_min_peak,
                               primary_laser_angle=primary_laser_angle,
                               img_leafl_tl=img_leafl_bot_tl, img_leafl_br=img_leafl_bot_br,
                               img_leafl_thresh=img_leafl_thresh,
                               leaflet_calib_frame_range=leaflet_calib_frame_range,
                               dbg_overlay=dbg_overlay, dbg_threshold=dbg_threshold)

    results_smooth = smooth_results(results, smooth_cutoff)
    write_results_csv(results, results_smooth, smooth_cutoff, csv_path)
    print("\nProcessing complete!")
