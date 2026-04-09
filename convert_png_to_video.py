import cv2
import glob
import os
import re


def convert_pngs_to_video(data_dir, output_path, fps=25, frame_range=(1, 50),
                          glob_pattern=None, frame_regex=None, stamp_frames=True,
                          repeat_count=1, repeat_delay=3):
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
        repeat_count: Number of times the frame range is repeated in the output video (default 1)
        repeat_delay: Delay in seconds between repeats (default 3)
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

    delay_frames = int(repeat_delay * fps)
    half_delay = delay_frames // 2

    for repeat in range(repeat_count):
        for image_path, frame_num in filtered_files:
            frame = cv2.imread(image_path)

            # Draw frame number on top-left corner
            if stamp_frames:
                text = f"Frame: {frame_num}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.84
                thickness = 2
                (_, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.putText(frame, text, (10, text_h + 10), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            writer.write(frame)
            print(f"[repeat {repeat + 1}/{repeat_count}] Added {image_path} (frame {frame_num})")

        # Add delay frames between repeats (not after the last one)
        if repeat < repeat_count - 1:
            def stamped(path, frame_num):
                img = cv2.imread(path)
                if stamp_frames:
                    text = f"Frame: {frame_num}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.84
                    thickness = 2
                    (_, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.putText(img, text, (10, text_h + 10), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                return img
            last_frame = stamped(filtered_files[-1][0], filtered_files[-1][1])
            first_frame = stamped(filtered_files[0][0], filtered_files[0][1])
            for _ in range(half_delay):
                writer.write(last_frame)
            for _ in range(delay_frames - half_delay):
                writer.write(first_frame)

    writer.release()
    print(f"\nVideo saved to: {output_path}")
    total_frames = len(filtered_files) * repeat_count + delay_frames * (repeat_count - 1)
    print(f"Repeats: {repeat_count}, Delay: {repeat_delay}s, FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")


if __name__ == "__main__":
    # Configuration
    frame_range = (765, 1147)
    repeat_count = 1
    repeat_delay = 3

    output_path = f"output/laser_video_{frame_range[0]}-{frame_range[1]}_{repeat_count}x.mp4"
    convert_pngs_to_video("data", output_path, frame_range=frame_range,
                          repeat_count=repeat_count, repeat_delay=repeat_delay)
