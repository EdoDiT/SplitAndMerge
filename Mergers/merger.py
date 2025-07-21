from abc import ABC
from typing import Callable, List, Tuple, Any

import numpy as np
from numpy import ndarray


class Merger(ABC):
    """
    Abstract base class for merging phase of split and merge algorithm.
    """
    def __init__(self,
            image: ndarray,
            merging_function: Callable,
            **kwargs
                 ):
        """
        Initializes the Merger with an image and a merging function.
        :param image: ndarray representing the image to be merged.
        :param merging_function: function that defines if two regions should be merged by operating on the
        weighted mean of the merge_mean function result calculated over the quadrants belonging to each region.
        :param kwargs: used for compatibility with other merger implementations.
        """
        self.merging_function = merging_function
        self.image = image
        self.merge_image = image.copy()  # Initialize with the original image
        self.merge_result = []
        self.graph_nodes = []

    def build_region_adjacency_graph(self, root: tuple, quadtree: dict):
        """
        Build the region adjacency graph for the image.
        """
        pass

    def merge(self):
        """
        Merges the regions in the image based on the adjacency graph.
        """
        pass

    def get_merge_image(self):
        """Returns the merged image"""
        for node in self.graph_nodes:
            if len(self.image.shape) == 2:  # Grayscale image
                avg_color = 0
            else:  # RGB image
                avg_color = [0, 0, 0]
            total_area = 0
            for quad_node in node:
                region = self.image[quad_node[1]:quad_node[1] + quad_node[3], quad_node[0]:quad_node[0] + quad_node[2]]
                region_color = region.mean(axis=(0, 1))
                area = quad_node[2] * quad_node[3]
                total_area += area
                if np.isscalar(region_color):
                    # Grayscale image
                    avg_color += region_color * area
                else:
                    # RGB image
                    avg_color = [avg_color[0] + region_color[0] * area,
                                 avg_color[1] + region_color[1] * area,
                                 avg_color[2] + region_color[2] * area]
            if np.isscalar(avg_color):
                avg_color /= total_area
            else:
                avg_color = [avg_color[0] / total_area, avg_color[1] / total_area, avg_color[2] / total_area]
            for quad_node in node:
                x = quad_node[0]
                y = quad_node[1]
                size_x = quad_node[2]
                size_y = quad_node[3]
                if np.isscalar(avg_color):
                    self.merge_image[y:y + size_y, x:x + size_x] = np.full((size_y, size_x), avg_color)
                else:
                    self.merge_image[y:y + size_y, x:x + size_x] = np.full((size_y, size_x, 3), avg_color)
        return self.merge_image

    def _are_similar(
            self,
            region1: List[Tuple[int, int, int, int, Any, int]],
            region2: List[Tuple[int, int, int, int, Any, int]],
    ) -> bool:
        """Checks if two regions are similar by computing the weighted mean based on blocks belonging to each region.
        Args:
            region1 (tuple): A tuple representing the first region (x, y, size).
            region2 (tuple): A tuple representing the second region (x, y, size).
        Returns:
            bool: True if the regions are similar, False otherwise.
        """
        means1 = [n[4] for n in region1]
        means2 = [n[4] for n in region2]
        areas1 = [n[5] for n in region1]
        areas2 = [n[5] for n in region2]
        weighted_means1 = np.average(means1, axis=0, weights=areas1)
        weighted_means2 = np.average(means2, axis=0, weights=areas2)
        return self.merging_function(weighted_means1, weighted_means2)