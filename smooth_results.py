"""
Smooth angle results from a results.csv file using a zero-phase
Butterworth low-pass filter. Can be run standalone to sweep cutoff values.

Usage:
    python smooth_results.py [csv_path] [--cutoff 0.1] [--sweep]
"""
import csv
import sys
import numpy as np
from scipy.signal import butter, filtfilt


def load_results_csv(csv_path):
    """
    Load a results.csv into a list of (frame_number, angle, method) tuples.
    Angle is float or None if 'N/A'.
    """
    results = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row['Frame'])
            method    = row['Method']
            # Support both old single-angle and new split-column formats
            raw = row.get('Angle (laser)') or row.get('Angle (leaflet)') or row.get('Angle', 'N/A')
            if not raw or raw == 'N/A':
                angle = None
            else:
                try:
                    angle = float(raw)
                except ValueError:
                    angle = None
            results.append((frame_num, angle, method))
    return results


def smooth_results(results, cutoff_freq):
    """
    Apply a zero-phase Butterworth low-pass filter to the angle column.
    None values are linearly interpolated before filtering, then restored.

    Args:
        results:      List of (frame_number, angle, method) tuples
        cutoff_freq:  Cutoff as fraction of Nyquist (0.0-1.0)

    Returns:
        New list of (frame_number, angle, method) with smoothed angles
    """
    angles    = [angle for _, angle, _ in results]
    none_mask = [a is None for a in angles]

    arr  = np.array([a if a is not None else np.nan for a in angles])
    nans = np.isnan(arr)
    if nans.all():
        return results
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

    b, a     = butter(2, cutoff_freq, btype='low')
    smoothed = filtfilt(b, a, arr)

    return [
        (frame_num, None if none_mask[i] else float(smoothed[i]), method)
        for i, (frame_num, _, method) in enumerate(results)
    ]


def write_smoothed_csv(results, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Method', 'Angle (laser)', 'Angle (leaflet)'])
        for frame_number, angle, method in results:
            angle_s       = f"{angle:.2f}" if angle is not None else 'N/A'
            laser_angle   = angle_s if method == 'laser'   else ''
            leaflet_angle = angle_s if method == 'leaflet' else ''
            writer.writerow([frame_number, method, laser_angle, leaflet_angle])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smooth results.csv angles")
    parser.add_argument("csv_path", nargs="?", default="output/results.csv")
    parser.add_argument("--cutoff", type=float, default=0.1,
                        help="Low-pass cutoff as fraction of Nyquist (0.0-1.0)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep a range of cutoff values and write one CSV per value")
    args = parser.parse_args()

    results = load_results_csv(args.csv_path)
    print(f"Loaded {len(results)} frames from {args.csv_path}")

    if args.sweep:
        cutoffs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        all_smoothed = {}
        for cutoff in cutoffs:
            smoothed = smooth_results(results, cutoff)
            all_smoothed[cutoff] = smoothed
            angles = [a for _, a, _ in smoothed if a is not None]
            print(f"  cutoff={cutoff:.2f}  range=[{min(angles):.2f}, {max(angles):.2f}]")

        out_path = args.csv_path.replace(".csv", "_sweep.csv")
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Method', 'Angle'] + [f'Angle({c:.2f})' for c in cutoffs])
            for i, (frame_num, angle_raw, method) in enumerate(results):
                angle_raw_s = f"{angle_raw:.2f}" if angle_raw is not None else 'N/A'
                row = [frame_num, method, angle_raw_s]
                for cutoff in cutoffs:
                    angle = all_smoothed[cutoff][i][1]
                    row.append(f"{angle:.2f}" if angle is not None else 'N/A')
                writer.writerow(row)
        print(f"  -> {out_path}")
    else:
        smoothed = smooth_results(results, args.cutoff)
        out_path = args.csv_path.replace(".csv", f"_smooth_{args.cutoff:.2f}.csv")
        write_smoothed_csv(smoothed, out_path)
        print(f"  cutoff={args.cutoff:.2f}  -> {out_path}")
