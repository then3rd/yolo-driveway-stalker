"""GUI polyon selector."""

import json
import logging
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .constants import DETECT_AREA_JSON, DETECT_IMAGE, TRIANGLE_SIDES

logger = logging.getLogger(__name__)


class PolygonPicker:  # noqa: D101
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.polygon_patch = None

    def pick_polygon_corners(self, min_points=3):
        """Display a UI for picking polygon corners on an image."""
        # Load and display the image
        img = mpimg.imread(self.image_path)
        self.ax.imshow(img)
        self.ax.set_title(f"Click to select at least {min_points} points. Press Enter when done.")
        self.points = []
        self.polygon_patch = None
        # Connect event handlers
        _cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        _cid_key = self.fig.canvas.mpl_connect(
            "key_press_event", lambda event: self._on_key(event, min_points)
        )
        plt.tight_layout()
        plt.show()
        # Return the points after the UI is closed
        return self.get_points()

    def _on_click(self, event):
        """Handle mouse click events to add points"""
        if event.xdata is None or event.ydata is None or event.button != 1:
            return
        # Add the point
        self.points.append((event.xdata, event.ydata))
        # Update display
        self._update_display()

    def _on_key(self, event, min_points):
        """Handle key press events to finish selection"""
        if event.key == "enter":
            if len(self.points) >= min_points:
                plt.close()
            else:
                logger.warning(f"Please select at least {min_points} points.")

    def _update_display(self):
        """Update the display with current points and polygon"""
        # Clear previous points and polygon
        for artist in self.ax.lines + self.ax.collections:  # pyright: ignore[reportOperatorIssue, reportOptionalMemberAccess]
            artist.remove()
        if self.polygon_patch:
            self.polygon_patch.remove()
        # Draw all points
        x, y = zip(*self.points, strict=False) if self.points else ([], [])
        self.ax.plot(x, y, "ro", markersize=4)  # Red dots for points
        # Draw lines connecting points
        if len(self.points) > 1:
            points_array = np.array([*self.points, self.points[0]])  # Close the polygon
            self.ax.plot(points_array[:, 0], points_array[:, 1], "b-", alpha=0.6)
        # Draw polygon if we have enough points
        if len(self.points) >= TRIANGLE_SIDES:
            self.polygon_patch = Polygon(np.array(self.points), alpha=0.2, color="green")
            self.ax.add_patch(self.polygon_patch)

        # Update the figure
        self.fig.canvas.draw()

    def get_points(self):
        """Return the array of selected polygon points."""
        return np.array(self.points)

    def save_points(self, output_file: Path):
        """Save the selected points to a file."""
        # Convert points to list for JSON serialization
        points_list = self.points
        # Prepare data with metadata
        data = {
            "image": Path(self.image_path),
            "points": points_list,
        }
        # Save to JSON file
        with Path.open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(points_list)} points to {output_file}")


if __name__ == "__main__":
    picker = PolygonPicker(DETECT_IMAGE)
    points = picker.pick_polygon_corners(min_points=3)

    logger.info("Selected polygon corners:")
    logger.info(points)

    # Save points to file
    picker.save_points(DETECT_AREA_JSON)
