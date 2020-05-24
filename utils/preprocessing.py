import cv2
import numpy as np
import re
from glob import glob
from typing import List, Tuple

from circle import Circle
from file_organizer import create_empty_dir
from image import Image
from preprocessing_exceptions import *

DATA_DIR = "../data/"
FILENAME_REGEX = r"(\d+)_(\d+)\.jpg"


def find_circle_nearest_point(binarized_image: Image, point: np.ndarray,
                              **kwargs) -> Circle:
    """
    Use Hough Transform to find circles in the image and return the one
    closest to the image center.

    :param binarized_image: Binary BW image
    :param point:
    :param kwargs:          Arguments to be passed to cv2.HoughCircles

    :return: Coordinates and radius of the circle found with the smallest
             Euclidean distance to the image's center
    """
    circles = cv2.HoughCircles(binarized_image.img,
                               method=cv2.HOUGH_GRADIENT, **kwargs)

    if circles is None:
        raise CirclesNotFoundException("No circles found")

    dist_from_centre = np.linalg.norm(circles[0, :, :2] - point, axis=1)
    circle_params = circles[0, np.argmin(dist_from_centre)]

    return Circle(*circle_params)


def extract_user_sample_ids(image_path: str) -> Tuple[str, str]:
    """
    Extract user ID and sample number from filename
    """
    image_filename = image_path.split("/")[-1]
    user_id, sample_id = re.match(FILENAME_REGEX, image_filename).groups()
    return user_id, sample_id


def circle_available_images(image_paths: List[str], target_dir: str):
    create_empty_dir(target_dir)
    failed_count = 0

    for image_path in image_paths:
        eye_image = Image(image_path=image_path)
        user_id, sample_id = extract_user_sample_ids(image_path)
        try:
            eye_image.circle_iris_and_pupil().save(
                f"{target_dir}/{user_id}_{sample_id}.jpg"
            )
        except ImageProcessingException:
            failed_count += 1

    print(f"{failed_count} failed preprocessings")


if __name__ == '__main__':
    circle_available_images(sorted(glob(DATA_DIR + "*")), "../circled_images")
