import numpy as np
from dataclasses import dataclass


@dataclass
class Circle:
    center_x: int
    center_y: int
    radius: int

    def is_within(self, other) -> bool:
        """
        :param other: Another Circle instance
        :return: True is this circle is contained in another circle
        """
        this_center = np.array([self.center_x, self.center_y])
        other_center = np.array([other.center_x, other.center_y])
        center_dist = np.linalg.norm(this_center - other_center)
        return other.radius > (center_dist + self.radius)
