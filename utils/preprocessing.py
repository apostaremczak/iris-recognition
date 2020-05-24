import cv2
import numpy as np
import os
import re
from glob import glob
from preprocessing_exceptions import *

from circle import Circle
from image import Image

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


if __name__ == '__main__':
    image_paths = sorted(glob(DATA_DIR + "*"))
    target_dir = "../circled_images"

    if os.path.exists(target_dir):
        files = glob(target_dir + "/*")
        for file in files:
            os.remove(file)
    else:
        os.makedirs(target_dir)

    failed_count = 0

    for image_path in image_paths:
        # Extract user ID and the sample number from the filename
        image_filename = image_path.split("/")[-1]
        user_id, sample_id = re.match(FILENAME_REGEX, image_filename).groups()
        eye_image = Image(image_path=image_path)

        try:
            eye_image.circle_iris_and_pupil().save(
                f"{target_dir}/{user_id}_{sample_id}.jpg"
            )
        except ImageProcessingException:
            failed_count += 1

    print(f"{failed_count} failed preprocessings")
