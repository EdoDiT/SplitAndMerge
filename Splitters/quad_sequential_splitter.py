from collections import deque
from typing import Tuple, Any

from numpy import ndarray

from Splitters.splitter import Splitter
from utils.quadnode import QuadNode


class QuadNodeSequentialSplitter(Splitter):
    """
    QuadNodeSequentialSplitter is a class that implements a quadtree-based image splitter.
    Uses the class QuadNode to represent nodes in the quadtree.
    """


    def build_quadtree(self) -> Tuple[dict, Tuple[int, int, int, int, Any, int]]:
        """
        Builds the quadtree structure for the image.
        This method initializes the root node and recursively splits the image into quadrants
        until the minimum block size is reached or the node is homogeneous according to the split function.
        """
        print("Building quadtree...")
        task_stack = deque()
        self.root = QuadNode(x=0, y=0, size_x=self._image.shape[0], size_y=self._image.shape[1], image=self._image)
        task_stack.append(self.root)
        while len(task_stack) != 0:
            node = task_stack.pop()
            if node.size_x > self._min_block_size and node.size_y > self._min_block_size and not node.is_homogeneous(self._split_function):
                children = node.split()
                for child in children:
                    task_stack.append(child)
            else:
                self.split_result.append(node)
        print("Quadtree built successfully.")
        self.quadtree_to_dict_form()
        self.root = (self.root.x, self.root.y, self.root.size_x, self.root.size_y, self.root.mean, self.root.area)
        return self.quadtree, self.root

    def get_split_image(self) -> ndarray:
        """
        Returns the split image.
        This method modifies the split_image attribute based on the split results.
        """
        for node in self.split_result:
            block = self._image[node.y:node.y + node.size_y, node.x:node.x + node.size_x]
            avg_color = block.mean(axis=(0, 1), keepdims=True)
            self.split_image[node.y:node.y + node.size_y, node.x:node.x + node.size_x] = avg_color
        return self.split_image

    def quadtree_to_dict_form(self) -> dict:
        """Converts the quadtree into the dictionary form for compatibility with mergers."""
        task_stack = deque()
        self.quadtree = {}
        task_stack.append(self.root)
        while len(task_stack) != 0:
            node = task_stack.pop()
            if node.children:
                task_stack.extend(node.children)
                self.quadtree[(node.x, node.y, node.size_x, node.size_y, node.mean, node.area)] = [
                    (child.x, child.y, child.size_x, child.size_y, child.mean, child.area) for child in node.children]
        return self.quadtree