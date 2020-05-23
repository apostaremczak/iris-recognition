import cv2
import numpy as np
from image import Image


def preprocess_iris_image(image_path: str) -> np.ndarray:
    pass


def find_circles(simg, **kwargs):
    circle_img = simg
    circles = cv2.HoughCircles(circle_img.img,
                               method=cv2.HOUGH_GRADIENT, **kwargs)

    assert circles is not None, "No circles found"

    circles = np.uint8(np.around(circles))
    circle_img = Image(cv2.cvtColor(circle_img.img, cv2.COLOR_GRAY2RGB))

    # Draw the circles found
    for i in circles[0, :]:
        cv2.circle(circle_img.img, (i[0], i[1]), i[2], (255, 30, 0), 2)

    return circles


def circle_iris_and_pupil(eye: Image) -> Image:
    pupil = eye.binarize(4.5)
    pupil_only = pupil.close(np.ones((4, 4), np.uint8))

    pupil_center = find_circles(pupil_only, dp=1.0, minDist=10, param1=1,
                                param2=15, minRadius=1, maxRadius=500)
    pupil_center = pupil_center[0, 0]
    print(f"Pupil center found at ({pupil_center[0]}, {pupil_center[1]}); "
          f"Pupil radius: {pupil_center[2]}px")

    iris = eye.binarize(1.5)
    iris_only = iris.close(np.ones((18, 18), np.uint8)).erode(
        np.ones((4, 4), np.uint8)).open(np.ones((5, 5), np.uint8))
    iris_center = find_circles(iris_only, dp=1.1, minDist=5, param1=10,
                               param2=20, minRadius=1, maxRadius=500)
    iris_center = iris_center[0, 0]

    print(f"Iris center found at ({iris_center[0]}, {iris_center[1]}); "
          f"Iris radius: {iris_center[2]}px")

    # Draw the pupil and the iris

    eye_rgb = eye.to_rgb()

    # Pupil
    cv2.circle(eye_rgb.img, (pupil_center[0], pupil_center[1]),
               pupil_center[2], (255, 30, 0), 2)
    # Iris
    cv2.circle(eye_rgb.img, (iris_center[0], iris_center[1]), iris_center[2],
               (0, 255, 0), 2)

    return eye_rgb
