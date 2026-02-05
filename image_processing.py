import cv2
import os

def perform_edge_detection(input_path, output_path, low_threshold=50, high_threshold=150):
    """
    Load an image, perform edge detection, and save the result.

    Args:
        input_path: Path to input image
        output_path: Path to save edge-detected image
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
    """
    # Read the image
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(f"Could not load image from {input_path}")

    print(f"Loaded image: {input_path}")
    print(f"Image shape: {image.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Save the result
    cv2.imwrite(output_path, edges)
    print(f"Edge detection completed. Saved to: {output_path}")

    return edges


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Input image path
    input_image = "data/Set_01_1820.png"

    # Output image path
    output_image = "output/Set_01_1820_edges.png"

    # Perform edge detection
    edges = perform_edge_detection(input_image, output_image)

    print(f"\nEdge detection complete!")
    print(f"Input: {input_image}")
    print(f"Output: {output_image}")
