from abc import ABC
from typing import Tuple, Any

import numpy as np
from numpy import ndarray


class Splitter(ABC):
    """
    Abstract base class for splitters.
    """

    def __init__(
            self,
            image: ndarray,
            split_function,
            merge_mean: callable,
            min_block_size: int = 10,
            **kwargs
    ):
        """
        Initializes the Splitter with an image and parameters for splitting and merging.
        :param image: ndarray of the image to be split.
        :param split_function: function to determine if a block should be split.
        :param merge_mean: calculates a value over a block, that will be used for merging.
        :param min_block_size: Minimum size of a block to be considered for splitting.
        :param kwargs: Additional arguments for compatibility with subclasses.
        """
        self._split_function = split_function
        self._merge_mean = merge_mean
        self._image = image
        self.split_image: ndarray = self._image.copy()  # This will be modified in the split process
        self.root = None
        self.quadtree = None
        self._min_block_size = min_block_size
        self.split_result = []

    def build_quadtree(self) -> Tuple[dict, Tuple[int, int, int, int, Any, int]]:
        """
        Builds the quadtree structure for the image.
        This method should be implemented in subclasses.
        """
        pass

    def get_split_image(self) -> ndarray:
        """
        Returns the split image.
        """
        pass