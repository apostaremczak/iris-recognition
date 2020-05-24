import numpy as np
from dataclasses import dataclass


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
