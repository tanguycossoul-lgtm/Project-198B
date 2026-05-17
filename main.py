import argparse
import csv
import cv2
import os
import glob
import shutil
import sys
from pipeline.smooth_results import smooth_results
from pipeline.config_loader import load_config

from pipeline.laser_detection import detect_primary_laser_angle, method1_laser
from pipeline.leaflet_detection import preprocess_image, count_white_pixels, method2_leaflet, compute_leaflet_angles


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
        data_dir: Input directory with *.png files
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
    image_files = sorted(glob.glob(f"{data_dir}/*.png"))
    start, end = frame_range
    image_files = [f for f in image_files
                   if start <= int(f.split('_')[-1].split('.')[0]) <= end]

    all_data = []

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
        else:
            print(f"  method2_leaflet frame {frame_num} (white_pixel_count={wpc})")
            frame_data = method2_leaflet(img_path, image, frame_num,
                                         img_leafl_tl, img_leafl_br,
                                         img_thresh=img_leafl_thresh,
                                         dbg_dir=dbg_dir,
                                         dbg_overlay=dbg_overlay,
                                         dbg_threshold=dbg_threshold)
        all_data.append(frame_data)

    results = compute_leaflet_angles(all_data, image_files[-1], img_leafl_tl, dbg_dir,
                                     leaflet_calib_frame_range)

    print(f"  {len(all_data)} frames processed.")
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, metavar="DATASET",
                        help="Dataset subfolder name inside data/")
    args = parser.parse_args()

    data_dir   = os.path.join("data", args.d)
    output_dir = "output"
    dbg_dir    = "dbg"
    csv_path   = f"{output_dir}/results.csv"
    dbg_overlay   = False
    dbg_threshold = False

    if not os.path.isdir(data_dir) or not glob.glob(f"{data_dir}/*.png"):
        print(f"Error: no PNG files found in '{data_dir}'", file=sys.stderr)
        sys.exit(1)

    calib_path = os.path.join(data_dir, "config_custom.json")
    if not os.path.exists(calib_path):
        calib_path = "config_default.json"
    print(f"Config: {calib_path}")
    cfg = load_config(calib_path)

    frame_range                = tuple(cfg["frame_range"])
    method_select_thresh       = cfg["method_select_thresh"]
    smooth_cutoff              = cfg["smooth_cutoff"]
    img_laser_prim_tl          = tuple(cfg["img_laser_prim_tl"])
    img_laser_prim_br          = tuple(cfg["img_laser_prim_br"])
    img_laser_tl               = tuple(cfg["img_laser_tl"])
    img_laser_br               = tuple(cfg["img_laser_br"])
    img_laser_thresh           = cfg["img_laser_thresh"]
    laser_strip_width          = cfg["laser_strip_width"]
    laser_peak_step            = cfg["laser_peak_step"]
    laser_min_peak             = cfg["laser_min_peak"]
    img_leafl_bot_tl           = tuple(cfg["img_leafl_tl"])
    img_leafl_bot_br           = tuple(cfg["img_leafl_br"])
    img_leafl_thresh           = cfg["img_leafl_thresh"]
    leaflet_calib_frame_range  = tuple(cfg["leaflet_calib_frame_range"])

    reset_dirs(output_dir, dbg_dir)

    first_frame = sorted(glob.glob(f"{data_dir}/*.png"))[0]
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
