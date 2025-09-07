"""
detect_plates.py
----------------
License Plate Detection using YOLO.

Usage:
    python detect_plates.py --weights runs/detect/train5/weights/best.pt --source input.jpg

Output:
    - Annotated image saved as <image>_detected.jpg
    - Image also displayed in a window
"""

import os
import cv2
import argparse
from ultralytics import YOLO


def detect_plates(weights: str, image_path: str, conf: float = 0.25):
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load YOLO model
    model = YOLO(weights)

    # Run detection
    results = model.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
    img = cv2.imread(image_path)

    # Draw bounding boxes
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save annotated image
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_detected{ext}"
    cv2.imwrite(output_path, img)

    # Display image in a popup window
    cv2.imshow("Detected Plates", img)
    cv2.waitKey(0)   # Wait for any key press
    cv2.destroyAllWindows()

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="License Plate Detection with YOLO")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--source", "--image", dest="image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path = detect_plates(args.weights, args.image, conf=args.conf)
    print(f" Output saved to {out_path}")
