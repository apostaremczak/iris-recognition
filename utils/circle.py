import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Circle:
    center_x: int
    center_y: int
    radius: int

    def to_numpy(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y, self.radius])

    def is_within(self, other) -> bool:
        """
        :param other: Another Circle instance
        :return: True is this circle is contained in another circle
        """
        center_dist = np.linalg.norm(
            self.to_numpy()[:-1] - other.to_numpy()[:-1])
        return other.radius > (center_dist + self.radius)

    def find_circle_coordinates(self,
                                img_size: Tuple[int, int],
                                n_sides: int = 600):
        """
        Find the coordinates of a circle based on its centre and radius.
        Source: https://github.com/thuyngch/Iris-Recognition/

        :param img_size: Size of the image that the circle will be plotted onto
        :param n_sides: Number of sides of the convex-hull bordering the circle

        :return x, y: Circle coordinates
        """
        a = np.linspace(0, 2 * np.pi, 2 * n_sides + 1)
        xd = np.round(self.radius * np.cos(a) + self.center_x)
        yd = np.round(self.radius * np.sin(a) + self.center_y)

        #  Get rid of values larger than image
        xd2 = xd
        coords = np.where(xd >= img_size[1])
        xd2[coords[0]] = img_size[1] - 1
        coords = np.where(xd < 0)
        xd2[coords[0]] = 0

        yd2 = yd
        coords = np.where(yd >= img_size[0])
        yd2[coords[0]] = img_size[0] - 1
        coords = np.where(yd < 0)
        yd2[coords[0]] = 0

        x = np.round(xd2).astype(int)
        y = np.round(yd2).astype(int)

        return x, y
