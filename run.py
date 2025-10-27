"""Car in garage vs driveway detector."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import coloredlogs
import cv2
import requests
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)

CAR = 2  # 2 is car class in COCO dataset

logger = logging.getLogger(__name__)


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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Detect cars in driveways")
    parser.add_argument("--image", type=str, required=True, help="path to input image")
    parser.add_argument(
        "--garage-area", type=str, default=None, help="Garage area coordinates as x1,y1,x2,y2"
    )
    # parser.add_argument(
    #     "--driveway-area", type=str, default=None, help="Driveway area coordinates as x1,y1,x2,y2"
    # )
    parser.add_argument(
        "--confidence", type=float, default=0.4, help="Confidence threshold for detections"
    )
    parser.add_argument("--output", type=str, default="output.jpg", help="path for output image")
    parser.add_argument("--show", type=bool, help="Display output image")
    return parser.parse_args()


def load_model():
    """Load  model."""
    # Define model path and URL
    # model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
    model_path = Path.home() / ".cache" / "ultralytics" / "yolo11n-seg.pt"
    # model_path = Path.home() / ".cache" / "ultralytics" / "yolo11n.pt"

    # Download model if not present
    download_model(model_url, model_path)

    # Load model
    model = YOLO(str(model_path))
    model.to("cpu")  # Explicitly set device to CPU

    return model


def parse_area(area_str):
    """Parse area string into coordinates."""
    if not area_str:
        return None
    return [int(x) for x in area_str.split(",")]


def is_in_area(points, area, *, check_all_points=True):
    """Check if a polygon or bounding box is in a defined area."""
    if area is None:
        return False

    box = 4
    # Check if input is a polygon (array of points) or a bounding box
    if isinstance(points, list) and len(points) == box:
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
    # Check if all points are inside the area
    for point in points:
        x, y = point
        if not (area[0] <= x <= area[2] and area[1] <= y <= area[3]):
            return False
    return True


def process_image(image_path: Path, model, garage_area, conf=0.4):
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

    # Draw garage area
    if garage_area:
        cv2.rectangle(
            output_image,
            (garage_area[0], garage_area[1]),
            (garage_area[2], garage_area[3]),
            GREEN,
            3,
        )
        cv2.putText(
            output_image,
            "Garage",
            (garage_area[0], garage_area[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            GREEN,
            3,
        )

    # Track counts
    garage_count = 0
    driveway_count = 0

    # Process each detection result
    for r in results:
        boxes = r.boxes

        masks = r.masks

        if masks is None:
            logger.warning("No segmentation masks found in results. Using bounding boxes only.")
            continue

        # Process each detection
        for i, box in enumerate(boxes):
            # Check if the detection is a car (class 2 in COCO)
            cls = int(box.cls.item())
            if cls == CAR:
                # Get bounding box coordinates (for label placement)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = box.conf.item()

                # Get the corresponding mask polygon points
                try:
                    mask_points = masks.xy[i]

                    # Convert mask points to integer for drawing
                    mask_points_int = mask_points.astype(int)

                    # Determine location based on mask points
                    in_garage = is_in_area(mask_points, garage_area)

                    # Set color and label based on location
                    if in_garage:
                        color = GREEN
                        label = "Car (Garage)"
                        garage_count += 1
                    else:
                        color = RED
                        label = "Car (Driveway)"
                        driveway_count += 1

                    # Draw the polygon mask
                    cv2.polylines(
                        img=output_image,
                        pts=[mask_points_int],
                        isClosed=True,
                        color=color,
                        thickness=3,
                    )
                    # cv2.fillPoly(output_image, [mask_points_int], (*color, 50))

                    # Draw bounding box and label
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), CYAN, 2)
                    cv2.putText(
                        output_image,
                        f"{label} {conf_score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )
                except (IndexError, AttributeError) as e:
                    logger.warning(f"Error processing mask for detection {i}: {e}")
                    continue

    # Display counts
    cv2.putText(
        output_image,
        f"Cars in Garage: {garage_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        GREEN,
        2,
    )
    cv2.putText(
        output_image,
        f"Cars in Driveway: {driveway_count}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        RED,
        2,
    )

    return output_image, garage_count, driveway_count


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
    # Load the image

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


def main():
    """Primary routine"""
    # Parse command line arguments
    args = parse_arguments()

    setup_logger(log_level="DEBUG")

    # Load model
    logger.info("Loading model...")
    model = load_model()

    # Parse garage and driveway areas
    garage_area = parse_area(args.garage_area)
    # driveway_area = parse_area(args.driveway_area)

    # Process the image
    logger.info(f"Processing image: {args.image}")
    try:
        output_image, garage_count, driveway_count = process_image(
            args.image,
            model,
            garage_area,
            conf=args.confidence,
        )

        # Save the output image
        output_path = Path(args.output)
        cv2.imwrite(str(output_path), output_image)
        logger.info(f"Output saved to: {args.output}")

        # Display results
        logger.info(
            f"Cars detected: {garage_count} in garage, {driveway_count} in driveway"  # , {other_count} elsewhere"
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
