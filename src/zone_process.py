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
from pydantic import BaseModel, Field
from shapely.geometry import Point, Polygon
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


class PointModel(BaseModel):
    """Model representing a 2D point."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        """Convert to tuple for Shapely operations."""
        return (self.x, self.y)


class BoundingBoxModel(BaseModel):
    """Model representing a bounding box with two points."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_corners(self) -> list[tuple[float, float]]:
        """Get the four corners of the bounding box."""
        return [
            (self.x1, self.y1),  # top left
            (self.x2, self.y1),  # top right
            (self.x2, self.y2),  # bottom right
            (self.x1, self.y2),  # bottom left
        ]

    def as_shapely_polygon(self) -> Polygon:
        """Convert bounding box to Shapely Polygon."""
        return Polygon(self.as_corners())

    def get_center(self) -> Point:
        """Get the center point of the bounding box."""
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PolygonModel(BaseModel):
    """Model representing a polygon."""

    points: list[PointModel] = Field(min_length=3)

    def as_point_list(self) -> list[tuple[float, float]]:
        """Convert to list of tuples for Shapely operations."""
        return [point.as_tuple() for point in self.points]

    def as_shapely_polygon(self) -> Polygon:
        """Convert to Shapely Polygon."""
        return Polygon(self.as_point_list())

    def as_numpy_array(self) -> np.ndarray:
        """Convert to numpy array for OpenCV operations."""
        return np.array(self.as_point_list(), np.int32).reshape((-1, 1, 2))


class DetectionResult(BaseModel):
    """Model representing a detection result."""

    bbox: BoundingBoxModel
    confidence: float
    class_id: int
    mask_points: list[tuple[float, float]] | None = None
    in_zone: bool = False

    def as_mask_polygon(self) -> Polygon | None:
        """Convert mask points to Shapely Polygon if available."""
        if self.mask_points:
            return Polygon(self.mask_points)
        return None


def setup_logger(log_level: str):
    """Colored logger setup."""
    logging.basicConfig(level=log_level)
    coloredlogs.DEFAULT_LOG_FORMAT = "%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s"
    coloredlogs.install(level=log_level, isatty=True)


def download_model(model_url: str, model_path: Path):
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


def load_model() -> YOLO:
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


def parse_area(area_str: str) -> BoundingBoxModel | None:
    """Parse area string into a BoundingBoxModel."""
    if not area_str:
        return None

    coords = [int(x) for x in area_str.split(",")]
    if len(coords) != 4:
        raise ValueError("Area must be specified as x1,y1,x2,y2")
    return BoundingBoxModel(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


def is_in_area(  # noqa: C901, PLR0911
    geometry: list[tuple[float, float]] | BoundingBoxModel | Polygon,
    area: PolygonModel | BoundingBoxModel | Polygon,
    *,
    check_all_points: bool = True,
) -> bool:
    """Check if a geometry is in a defined area using Shapely."""
    if area is None:
        return False

    # Convert area to Shapely Polygon if needed
    area_polygon = area
    if isinstance(area, PolygonModel) or isinstance(area, BoundingBoxModel):
        area_polygon = area.as_shapely_polygon()
    elif not isinstance(area_polygon, Polygon):
        # Convert list of points to Polygon
        try:
            area_polygon = Polygon(area)
        except Exception as e:
            logger.warning(f"Failed to create polygon from area: {e}")
            return False

    # Convert input geometry to Shapely object if needed
    if isinstance(geometry, BoundingBoxModel):
        if check_all_points:
            # Check if all corners are inside the area
            return all(area_polygon.contains(Point(p)) for p in geometry.as_corners())
        # Check if center is in the area
        return area_polygon.contains(geometry.get_center())
    if isinstance(geometry, list):
        # If geometry is a list of points
        if len(geometry) == 4 and not all(isinstance(p, list) for p in geometry):
            # It's a bounding box [x1, y1, x2, y2]

            x1, y1, x2, y2 = geometry
            bbox = BoundingBoxModel(x1=x1, y1=y1, x2=x2, y2=y2)
            return is_in_area(bbox, area_polygon, check_all_points=check_all_points)

        # It's a list of points
        if check_all_points:
            # Check if all points are inside the area
            return all(area_polygon.contains(Point(p)) for p in geometry)
        # Check if any point is inside the area
        return any(area_polygon.contains(Point(p)) for p in geometry)
    if isinstance(geometry, Polygon):
        if check_all_points:
            # Check if the entire polygon is inside the area
            return area_polygon.contains(geometry)
        # Check if the polygons intersect
        return area_polygon.intersects(geometry)

    logger.warning(f"Unrecognized geometry format: {type(geometry)}")
    return False


def process_image(  # noqa: C901
    image_path: Path, model: YOLO, area: PolygonModel | BoundingBoxModel, conf: float = 0.4
) -> tuple[np.ndarray, int, int]:
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
        if isinstance(area, PolygonModel):
            # It's a polygon - draw it

            cv2.polylines(
                img=output_image,
                pts=[area.as_numpy_array()],
                isClosed=True,
                color=COLOR_ZONE,
                thickness=3,
            )
            # Add label at the first point
            first_point = area.points[0]
            cv2.putText(
                output_image,
                "Zone",
                (int(first_point.x), int(first_point.y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                COLOR_ZONE,
                3,
            )

        elif isinstance(area, BoundingBoxModel):
            # It's a rectangle
            cv2.rectangle(
                output_image,
                (int(area.x1), int(area.y1)),
                (int(area.x2), int(area.y2)),
                COLOR_PASS,
                3,
            )
            cv2.putText(
                output_image,
                "Zone",
                (int(area.x1), int(area.y1) - 10),
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
                bbox = BoundingBoxModel(x1=x1, y1=y1, x2=x2, y2=y2)

                # Handle segmentation masks if available
                in_zone = False
                if masks is not None:
                    try:
                        # Get the corresponding mask polygon points
                        mask_points = masks.xy[i]
                        # Convert mask points to list of tuples
                        mask_points_list = [(float(p[0]), float(p[1])) for p in mask_points]
                        # Determine location based on mask points
                        in_zone = is_in_area(mask_points_list, area)
                        # Convert mask points to integer for drawing
                        mask_points_int = mask_points.astype(int)

                        # Draw the polygon mask
                        cv2.polylines(
                            img=output_image,
                            pts=[mask_points_int],
                            isClosed=True,
                            color=COLOR_PASS if in_zone else COLOR_FAIL,
                            thickness=3,
                        )

                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Error processing mask for detection {i}: {e}")
                        # Fall back to bounding box if mask fails

                        in_zone = is_in_area(bbox, area)
                else:
                    # No masks available, use bounding box for location determination
                    logger.debug("Using bounding box for detection")

                    in_zone = is_in_area(bbox, area)
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


def resize_maintain_aspect_ratio(
    image: np.ndarray, width: int | None = None, height: int | None = None
) -> np.ndarray:
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


def add_timestamp_to_image(image: np.ndarray) -> None:
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


def load_polygon_from_file(file_path: Path) -> PolygonModel | None:
    """Load polygon points from a JSON file."""
    with Path.open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list) and all(isinstance(p, list) for p in data):
        # Convert to PointModel list
        points = [PointModel(x=float(p[0]), y=float(p[1])) for p in data]
        logger.info(f"Loaded {len(points)} polygon points from {file_path}")
        return PolygonModel(points=points)

    logger.warning(f"No valid points found in {file_path}")
    return None


def parse_arguments() -> argparse.Namespace:
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

    parser.add_argument("--show", action="store_true", help="Display output image")
    return parser.parse_args()


def main() -> None:
    """Primary routine"""
    # Parse command line arguments
    args = parse_arguments()
    setup_logger(log_level="DEBUG")
    # Load model
    logger.info("Loading model...")
    model = load_model()

    # Parse zone area - either from polygon file or from area coordinates

    area: PolygonModel | BoundingBoxModel | None = None
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
            Path(args.image),
            model,
            area,
            conf=args.confidence,
        )
        # Save the output image
        output_path = Path(args.output)
        cv2.imwrite(str(output_path), output_image)
        logger.info(f"Output saved to: {args.output}")
        # Display results

        logger.info(f"Objects detected: {zone_count} in zone, {outside_count} outside")

        if args.show:
            # Display the image
            cv2.imshow("YOLO Detection", resize_maintain_aspect_ratio(output_image, height=1024))
            logger.info("Press any key to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception:
        logger.exception("Error processing image.")


def torch_test() -> None:
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
