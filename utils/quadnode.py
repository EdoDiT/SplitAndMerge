import numpy as np
from typing import Optional, Tuple, List

from networkx.algorithms.boundary import node_boundary
from numpy import ndarray

class QuadNode:
    """Represents a node in a quadtree structure for image segmentation.
    Each node corresponds to a rectangular region in the image and can be split into four quadrants.
    """
    def __init__(
            self,
            x: int,
            y: int,
            size_x: int,
            size_y: int,
            image: ndarray,
            children: Optional[List['QuadNode']] = None
    ):
        self.x = x  # Top-left x coordinate
        self.y = y  # Top-left y coordinate
        self.size_x = size_x  # Size of the region
        self.size_y = size_y
        self.area = size_x * size_y  # Area of the node
        self.mean = np.mean(image[y:y + size_y, x:x + size_x]) # Value of the quadrant, pre-computed for merging
        self.children = children if children is not None else []
        self.image = image

    def split(self) -> ('QuadNode', 'QuadNode', 'QuadNode', 'QuadNode'):
        """Split the node into four quadrants and return the new nodes."""
        new_size_x = int(self.size_x / 2)
        new_size_y = int(self.size_y / 2)
        node_top_left = QuadNode(x=self.x, y=self.y, size_x=new_size_x, size_y=new_size_y, image=self.image)
        node_top_right = QuadNode(x=int(self.x + new_size_x), y=self.y, size_x=new_size_x, size_y=new_size_y, image=self.image)
        node_bottom_right = QuadNode(x=int(self.x + new_size_x), y=int(self.y + new_size_y), size_x=new_size_x, size_y=new_size_y, image=self.image)
        node_bottom_left = QuadNode(x=self.x, y=int(self.y + new_size_y), size_x=new_size_x, size_y=new_size_y, image=self.image)
        self.children = [node_top_left, node_top_right, node_bottom_right, node_bottom_left]
        return node_top_left, node_top_right, node_bottom_right, node_bottom_left

    @staticmethod
    def are_contiguous(node1:'QuadNode', node2:'QuadNode') -> bool:
        """ Check if two nodes are contiguous.
        Two nodes are considered contiguous if they touch each other either horizontally or vertically.
        :param node1: First node.
        :param node2: Second node.
        :return: True if nodes are contiguous, False otherwise.
        """
        if node1.x <= node2.x:
            pass
        else:
            node3 = node2
            node2 = node1
            node1 = node3
            pass
        if node1.x + node1.size_x == node2.x:
            if node1.y+node1.size_y <= node2.y:
                return False
            elif node1.y >= node2.y+node2.size_y:
                return False
            else:
                return True
        elif (node1.y + node1.size_y == node2.y) or (node2.y + node2.size_y == node1.y):
            if node1.x+node1.size_x <= node2.x:
                return False
            elif node1.x >= node2.x+node2.size_x:
                return False
            else:
                return True
        else:
            return False

    def is_homogeneous(self, split_function: callable) -> bool:
        """
        Check if the node is homogeneous by applying the split function to the image block.
        :param split_function: A function that takes a block of the image and returns True if it is homogeneous.
        :return: True if homogeneous, False otherwise.
        """
        return split_function(self.image[self.y:self.y + self.size_y, self.x:self.x + self.size_x])