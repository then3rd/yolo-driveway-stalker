"""Object in zone detector."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import coloredlogs
import cv2
import numpy as np
import requests
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

from .constants import (
    CYAN,
    DEFAULT_POINTS_JSON,
    DETECTION_CLASSES,
    GREEN,
    MODEL_CACHE_DIR,
    MODEL_URL,
    RED,
)

logger = logging.getLogger(__name__)

COLOR_ZONE = CYAN
COLOR_PASS = GREEN
COLOR_FAIL = RED


def setup_logger(log_level):
    """Colored logger setup."""
    logging.basicConfig(level=log_level)
    coloredlogs.DEFAULT_LOG_FORMAT = "%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s"
    coloredlogs.install(level=log_level, isatty=True)


def download_model(model_url, model_path):
    """Download the YOLO model if it doesn't exist."""
    model_path = Path(model_path)
    if not model_path.exists():
        logger.info(f"Downloading model to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with model_path.open("wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))
        logger.info("Download complete!")
    else:
        logger.info(f"Model already exists at {model_path}")


def load_model():
    """Load model."""
    # Download model if not present
    url_parsed = urlparse(MODEL_URL)
    file_name = Path(url_parsed.path).name
    model_path = MODEL_CACHE_DIR / file_name
    download_model(MODEL_URL, model_path)
    # Load model
    model = YOLO(str(model_path))
    model.to("cpu")  # Explicitly set device to CPU
    return model


def parse_area(area_str):
    """Parse area string into coordinates."""
    if not area_str:
        return None
    return [int(x) for x in area_str.split(",")]


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray-casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x if p1y != p2y else p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def is_in_area(points, area, *, check_all_points=True):  # noqa: C901, PLR0911
    """Check if a polygon or bounding box is in a defined area."""
    if area is None:
        return False
    # If area is a polygon (list of points), use point-in-polygon algorithm
    if isinstance(area, list) and len(area) > 4 and all(isinstance(p, list) for p in area):
        # Area is a polygon
        # For each point in the input, check if it's inside the polygon
        box = 4
        if (
            isinstance(points, list)
            and len(points) == box
            and not all(isinstance(p, list) for p in points)
        ):
            # points is a bounding box [x1, y1, x2, y2]
            x1, y1, x2, y2 = points
            corners = [
                [x1, y1],  # top left
                [x2, y1],  # top right
                [x1, y2],  # bottom left
                [x2, y2],  # bottom right
            ]
            if check_all_points:
                # Check if all corners are inside the polygon
                return all(point_in_polygon(corner, area) for corner in corners)
            # Check if center is in the polygon
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return point_in_polygon([center_x, center_y], area)
        # points is a list of points (polygon)
        # Check if any/all points are in the area polygon
        if check_all_points:
            return all(point_in_polygon(point, area) for point in points)
        # Check if at least one point is inside
        return any(point_in_polygon(point, area) for point in points)
    # If area is a rectangle [x1, y1, x2, y2]
    if isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area):
        box = 4
        # Check if input is a polygon (array of points) or a bounding box
        if (
            isinstance(points, list)
            and len(points) == box
            and not all(isinstance(p, list) for p in points)
        ):
            # It's a bounding box [x1, y1, x2, y2]
            x1, y1, x2, y2 = points
            if check_all_points:
                # Check if all four corners of the box are in the area
                top_left = area[0] <= x1 <= area[2] and area[1] <= y1 <= area[3]
                top_right = area[0] <= x2 <= area[2] and area[1] <= y1 <= area[3]
                bottom_left = area[0] <= x1 <= area[2] and area[1] <= y2 <= area[3]
                bottom_right = area[0] <= x2 <= area[2] and area[1] <= y2 <= area[3]
                return top_left and top_right and bottom_left and bottom_right
            # Calculate center point of the box
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            # Check if center is in area
            return area[0] <= box_center_x <= area[2] and area[1] <= box_center_y <= area[3]
        # It's a polygon (array of points)
        # Check if all points are inside the rectangular area
        for point in points:
            x, y = point
            if not (area[0] <= x <= area[2] and area[1] <= y <= area[3]):
                return False
        return True

    logger.warning(
        f"Unrecognized area format: {type(area)} {len(area) if isinstance(area, list) else ''}"
    )
    return False


def process_image(image_path: Path, model, area, conf=0.4):
    """Process an image with model."""
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Perform detection
    results = model(image, conf=conf)
    # Create a copy of the image to draw on
    output_image = image.copy()
    # Add current timestamp
    add_timestamp_to_image(output_image)
    # Draw zone area
    if area:
        if isinstance(area, list) and len(area) > 4 and all(isinstance(p, list) for p in area):
            # It's a polygon - draw it
            pts = np.array(area, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img=output_image, pts=[pts], isClosed=True, color=COLOR_ZONE, thickness=3)
            # Add label at the first point
            cv2.putText(
                output_image,
                "Zone",
                (int(area[0][0]), int(area[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                COLOR_ZONE,
                3,
            )
        else:
            # It's a rectangle
            cv2.rectangle(
                output_image,
                (area[0], area[1]),
                (area[2], area[3]),
                COLOR_PASS,
                3,
            )
            cv2.putText(
                output_image,
                "Zone",
                (area[0], area[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                COLOR_PASS,
                3,
            )

    # Track counts
    zone_count = 0
    outside_count = 0

    # Process each detection result
    for r in results:
        boxes = r.boxes
        masks = r.masks

        # Process each detection
        for i, box in enumerate(boxes):
            # Check if the detection is a desired COCO class
            cls = int(box.cls.item())
            if cls in DETECTION_CLASSES:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = box.conf.item()
                # Handle segmentation masks if available
                if masks is not None:
                    try:
                        # Get the corresponding mask polygon points
                        mask_points = masks.xy[i]
                        # Convert mask points to integer for drawing
                        mask_points_int = mask_points.astype(int)
                        # Determine location based on mask points
                        in_zone = is_in_area(mask_points, area)
                        # Draw the polygon mask
                        cv2.polylines(
                            img=output_image,
                            pts=[mask_points_int],
                            isClosed=True,
                            color=COLOR_PASS if in_zone else COLOR_FAIL,
                            thickness=3,
                        )
                        # Optional: Fill the polygon with semi-transparent color
                        # cv2.fillPoly(output_image, [mask_points_int], (*color, 50))
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error processing mask for detection {i}: {e}")
                        # Fall back to bounding box if mask fails
                        in_zone = is_in_area([x1, y1, x2, y2], area)
                else:
                    # No masks available, use bounding box for location determination
                    logger.debug("Using bounding box for detection")
                    in_zone = is_in_area([x1, y1, x2, y2], area)
                    # Draw the bounding box
                    cv2.rectangle(
                        output_image, (x1, y1), (x2, y2), COLOR_PASS if in_zone else COLOR_FAIL, 2
                    )

                # Set color and label based on location
                if in_zone:
                    color = COLOR_PASS
                    label = "Object (inside)"
                    zone_count += 1
                else:
                    color = COLOR_FAIL
                    label = "Object (outside)"
                    outside_count += 1

                # Draw label
                cv2.putText(
                    output_image,
                    f"{label} {conf_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

    # Display counts
    cv2.putText(
        output_image,
        f"Objects in zone: {zone_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR_PASS,
        2,
    )
    cv2.putText(
        output_image,
        f"Objects outside zone: {outside_count}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR_FAIL,
        2,
    )

    return output_image, zone_count, outside_count


def resize_maintain_aspect_ratio(image, width=None, height=None):
    """Resize image based on width or height and selected interpolation method"""
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:  # resizing height
        if height is None:
            raise ValueError("Either width or height must be provided")
        r = height / float(h)
        dim = (int(w * r), height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def add_timestamp_to_image(image):
    """Add the current ISO timestamp to an image"""
    # Get current timestamp in ISO format
    timestamp = datetime.now(ZoneInfo("America/Denver")).strftime("%b %d, %Y - %I:%M:%S %p")
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)  # White color in BGR
    thickness = 2
    # Calculate position for top right with padding
    text_size = cv2.getTextSize(timestamp, font, font_scale, thickness)[0]
    padding = 10
    x = image.shape[1] - text_size[0] - padding
    y = text_size[1] + padding
    position = (x, y)
    # Draw text with a dark background for better visibility
    cv2.putText(image, timestamp, position, font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, timestamp, position, font, font_scale, color, thickness)


def load_polygon_from_file(file_path: Path):
    """Load polygon points from a JSON file."""
    with Path.open(file_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        logger.info(f"Loaded {len(data)} polygon points from {file_path}")
        return data
    logger.warning(f"No valid 'points' field found in {file_path}")
    return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Detect objects in zones")
    parser.add_argument("--image", type=str, required=True, help="path to input image")
    parser.add_argument(
        "--zone-area", type=str, default=None, help="Zone area coordinates as x1,y1,x2,y2"
    )
    parser.add_argument(
        "--points", type=str, default=None, help="Path to JSON file containing polygon points"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.4, help="Confidence threshold for detections"
    )
    parser.add_argument("--output", type=str, default="output.jpg", help="path for output image")
    parser.add_argument("--show", type=bool, help="Display output image")
    return parser.parse_args()


def main():
    """Primary routine"""
    # Parse command line arguments
    args = parse_arguments()
    setup_logger(log_level="DEBUG")
    # Load model
    logger.info("Loading model...")
    model = load_model()
    # Parse zone area - either from polygon file or from area coordinates
    area = None
    json_path = DEFAULT_POINTS_JSON
    if args.zone_area:
        area = parse_area(args.zone_area)
    else:
        if args.points:
            json_path = Path(args.points)
        area = load_polygon_from_file(json_path)
    # Process the image
    logger.info(f"Processing image: {args.image}")
    try:
        output_image, zone_count, outside_count = process_image(
            args.image,
            model,
            area,
            conf=args.confidence,
        )
        # Save the output image
        output_path = Path(args.output)
        cv2.imwrite(str(output_path), output_image)
        logger.info(f"Output saved to: {args.output}")
        # Display results
        logger.info(
            f"Cars detected: {zone_count} in zone, {outside_count} in outside"  # , {other_count} elsewhere"
        )
        if args.show:
            # Display the image
            cv2.imshow(
                "YOLO Car Detection", resize_maintain_aspect_ratio(output_image, height=1024)
            )
            logger.info("Press any key to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:  # noqa: BLE001
        logger.info(f"Error processing image: {e}")


def torch_test():
    """Torch Test"""
    import torch

    logger.info(f"CUDA available: {torch.cuda.is_available()}")  # Returns True if CUDA is available
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        logger.info(f"Current device ID: {current_device}")
        logger.info(f"Current device name: {torch.cuda.get_device_name(current_device)}")
        logger.info(f"Device memory address: {torch.cuda.device(current_device)}")
        logger.info(f"Total number of GPUs: {torch.cuda.device_count()}")


if __name__ == "__main__":
    # torch_test()
    main()
