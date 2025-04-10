import numpy as np
from typing import Optional, Tuple, List
from numpy import ndarray

class QuadNode:
    def __init__(
            self,
            x: int,
            y: int,
            size: int,
            children:Optional[List['QuadNode']] = []
    ):
        self.x = x  # Top-left x coordinate
        self.y = y  # Top-left y coordinate
        self.size = size  # Size of the region
        self.children = children

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (has no children)."""
        return self.children is []

    def split(self) ->('QuadNode', 'QuadNode', 'QuadNode', 'QuadNode'):
        node_top_left = QuadNode(x=self.x, y=self.y, size = self.size/2)
        node_top_right = QuadNode(x=self.x+1+self.size/2, y=self.y, size=self.size/2)
        node_bottom_right = QuadNode(x=self.x+1+self.size/2, y=self.y+1+self.size/2, size=self.size/2)
        node_bottom_left = QuadNode(x=self.x, y=self.y+1+self.size/2, size=self.size/2)
        self.children = [node_top_left, node_top_right, node_bottom_right, node_bottom_left]
        return node_top_left, node_top_right, node_bottom_right, node_bottom_left

    @staticmethod
    def are_contiguous(node1:'QuadNode', node2:'QuadNode') -> bool:
        if node1.x <= node2.x:
            pass
        else:
            node3 = node2
            node2 = node1
            node1 = node3
            pass
        if node1.x + node1.size == node2.x:
            if node1.y+node1.size <= node2.y:
                return False
            elif node1.y >= node2.y+node2.size:
                return False
            else:
                return True
        elif (node1.y + node1.size == node2.y) or (node2.y + node2.size == node1.y):
            if node1.x+node1.size <= node2.x:
                return False
            elif node1.x >= node2.x+node2.size:
                return False
            else:
                return True
        else:
            return False