import cv2
import glob
import os
import re


def convert_pngs_to_video(data_dir, output_path, fps=25, frame_range=(1, 50),
                          glob_pattern=None, frame_regex=None, stamp_frames=True):
    """
    Convert a sequence of PNG images into an MP4 video playable in browsers.

    Args:
        data_dir: Directory containing PNG files
        output_path: Path for the output video file
        fps: Frames per second (default 25)
        frame_range: Tuple (start, end) inclusive frame numbers to include (default (1, 50))
        glob_pattern: Custom glob pattern (default: "Set_01_*.png")
        frame_regex: Custom regex to extract frame number (default: r'Set_01_(\d+)\.png$')
        stamp_frames: If True, draw frame number on each frame (default True)
    """
    if glob_pattern is None:
        glob_pattern = "Set_01_*.png"
    if frame_regex is None:
        frame_regex = r'Set_01_(\d+)\.png$'

    image_files = sorted(glob.glob(f"{data_dir}/{glob_pattern}"))

    if not image_files:
        print(f"No {glob_pattern} files found in {data_dir}")
        return

    # Filter by frame range
    start, end = frame_range
    filtered_files = []
    for path in image_files:
        match = re.search(frame_regex, os.path.basename(path))
        if match:
            frame_num = int(match.group(1))
            if start <= frame_num <= end:
                filtered_files.append((path, frame_num))

    if not filtered_files:
        print(f"No files found in frame range {frame_range}")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(filtered_files[0][0])
    height, width = first_frame.shape[:2]

    # Use mp4v codec for broad browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_path, frame_num in filtered_files:
        frame = cv2.imread(image_path)

        # Draw frame number on top-left corner
        if stamp_frames:
            text = f"Frame: {frame_num}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.84
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.putText(frame, text, (10, text_h + 10), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        writer.write(frame)
        print(f"Added {image_path} (frame {frame_num})")

    writer.release()
    print(f"\nVideo saved to: {output_path}")
    print(f"Frames: {len(filtered_files)}, FPS: {fps}, Resolution: {width}x{height}")


if __name__ == "__main__":
    # Configuration
    frame_range = (1, 2000)
    
    output_path = f"output/laser_video_{frame_range[0]}-{frame_range[1]}.mp4"
    convert_pngs_to_video("data", output_path, frame_range=frame_range)
