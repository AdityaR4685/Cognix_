"""
Unfinished script for rendering videos
"""

import cv2
import os
import numpy as np
from tqdm import tqdm
import click


def get_image_paths(base_path, time_step):
    """
    Generate paths for images from different camera folders for a given time step.
    """
    cameras = ["rgb-front", "rgb-left", "rgb-right", "rgb-rear", "pcl-renders"]
    paths = {
        cam: os.path.join(base_path, cam, f"{time_step:06d}.jpg") for cam in cameras
    }
    return paths


def resize_images(images, size=(640, 480)):
    """
    Resize images to a common size.
    """
    return {cam: cv2.resize(img, size) for cam, img in images.items()}


def create_dummy_image(size=(480, 640)):
    """
    Read the image from the given path or create a dummy image if it does not exist.
    """
    return np.ones((size[0], size[1], 3), dtype=np.uint8) * 255


def combine_images(paths):
    """
    Combine images from different camera views into a single image.
    """
    images = {cam: cv2.imread(path) for cam, path in paths.items()}

    # Check if all images are read successfully
    if any(img is None for img in images.values()):
        print("Error reading images for one or more cameras.")
        return None

    images = resize_images(images)

    # Assuming all images are of the same size
    h, w, _ = images["rgb-front"].shape

    # Create the combined image
    top_row = cv2.hconcat(
        [create_dummy_image(), images["rgb-front"], create_dummy_image()]
    )
    middle_row = cv2.hconcat(
        [images["rgb-left"], images["pcl-renders"], images["rgb-right"]]
    )
    bottom_row = cv2.hconcat(
        [create_dummy_image(), images["rgb-rear"], create_dummy_image()]
    )

    combined_image = cv2.vconcat([top_row, middle_row, bottom_row])

    return combined_image


def process_images(base_path, output_path, num_time_steps):
    """
    Process images for a given number of time steps.
    """
    for time_step in tqdm(range(num_time_steps)):
        paths = get_image_paths(base_path, time_step)
        combined_image = combine_images(paths)
        if combined_image is not None:
            output_file = os.path.join(output_path, f"{time_step:06d}.jpg")

            cv2.imwrite(output_file, combined_image)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--output-path", "-o", type=click.Path(exists=False), default="output")
@click.option("--num-time-steps", "-n", type=int, default=750)
def main(path, output_path, num_time_steps):
    """
    Render cool video of a pointcloud and some camera images.
    """

    # NOTE: we assume that the rendered pointcloud images are in a folder called "pcl-renders"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    process_images(path, output_path, num_time_steps)


if __name__ == "__main__":
    main()
