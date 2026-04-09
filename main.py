import csv
import cv2
import os
import glob
import shutil

from laser_detection import method1_laser
from leaflet_detection import preprocess_image, count_white_pixels, method2_leaflet, compute_leaflet_angles


def process_sequence(data_dir, output_dir, tl, br,
                     frame_range, luminosity_threshold=20, leaflet_sel="top",
                     dbg_overlay=True, dbg_threshold=True,
                     method_selection_threshold=400):
    """
    Process a sequence of frames, dispatching to method1_laser or method2_leaflet
    based on the white pixel count threshold.

    Args:
        data_dir: Input directory with Set_01_*.png files
        output_dir: Root output directory
        tl: Top-left crop coordinate
        br: Bottom-right crop coordinate
        frame_range: (start, end) inclusive
        luminosity_threshold: Binary threshold value for leaflet detection
        leaflet_sel: "top" or "bot" leaflet selection
        dbg_overlay: If True, save overlay images
        dbg_threshold: If True, save threshold visualization images
        method_selection_threshold: Min white pixel count to use method2_leaflet;
                                    frames below this use method1_laser
    """
    image_files = sorted(glob.glob(f"{data_dir}/Set_01_*.png"))
    start, end = frame_range
    image_files = [f for f in image_files
                   if start <= int(f.split('_')[-1].split('.')[0]) <= end]

    all_data = []
    for image_path in image_files:
        frame_num = int(image_path.split('_')[-1].split('.')[0])

        # Pre-check white pixel count to select detection method
        image = cv2.imread(image_path)
        x1, y1 = tl
        x2, y2 = br
        binary = preprocess_image(image[y1:y2, x1:x2], luminosity_threshold)
        wpc    = count_white_pixels(binary)

        if wpc < method_selection_threshold:
            print(f"  method1_laser  frame {frame_num} (white_pixel_count={wpc})")
            frame_data = method1_laser(image_path, frame_num,
                                       tl, br, output_dir=output_dir,
                                       dbg_overlay=dbg_overlay)
        else:
            print(f"  method2_leaflet frame {frame_num} (white_pixel_count={wpc})")
            frame_data = method2_leaflet(image_path, frame_num,
                                         tl, br, output_dir=output_dir,
                                         threshold=luminosity_threshold,
                                         dbg_overlay=dbg_overlay,
                                         dbg_threshold=dbg_threshold)
        all_data.append(frame_data)

    results = compute_leaflet_angles(all_data, image_files[-1], tl, output_dir)

    # Write CSV
    csv_path = f"{output_dir}/results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Angle', 'Method'])
        for frame_number, angle, method in results:
            angle_s = f"{angle:.2f}" if angle is not None else 'N/A'
            writer.writerow([frame_number, angle_s, method])

    print(f"  {len(all_data)} frames processed. CSV: {csv_path}")
    return all_data


if __name__ == "__main__":

    # Configuration
    data_directory = "data"
    output_dir     = "output"
    frame_range                = (0, 2000)       # Inclusive range of frame numbers to process
    leaflet_bot_tl             = (1450, 660)     # Top-left (x, y) crop — bottom leaflet
    leaflet_bot_br             = (1615, 1240)    # Bottom-right (x, y) crop — bottom leaflet
    luminosity_threshold       = 20              # Pixels below this -> white, above -> black
    method_selection_threshold = 400             # Min white pixel count to use method2_leaflet
    dbg_overlay                = False           # Save overlay images
    dbg_threshold              = False           # Save threshold visualization images

    # Clear and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("Leaflet Angle Measurement")
    print("=" * 50)
    print(f"Processing frames from: {data_directory}")
    print(f"Frame range: {frame_range[0]} to {frame_range[1]}")
    print(f"Luminosity threshold: {luminosity_threshold}")
    print(f"Method selection threshold: {method_selection_threshold}")
    print()

    print("--- Leaflet: bottom ---")
    process_sequence(data_directory, output_dir,
                     tl=leaflet_bot_tl, br=leaflet_bot_br,
                     frame_range=frame_range, luminosity_threshold=luminosity_threshold,
                     leaflet_sel="bot",
                     dbg_overlay=dbg_overlay, dbg_threshold=dbg_threshold,
                     method_selection_threshold=method_selection_threshold)

    print("\nProcessing complete!")
