from collections import deque
from typing import Tuple, Any

from Splitters.splitter import Splitter


class DictSequentialSplitter(Splitter):
    """
    Sequential splitter that uses a dictionary to store split results.
    This class extends the Splitter class and implements the build_quadtree method.
    """
    def _is_homogeneous(self, node):
        """
        Checks if the block defined by the node is homogeneous using the split function.
        :param node: A tuple containing (x, y, size_x, size_y, mean, area).
        :return: True if the block is homogeneous, False otherwise.
        """
        x, y, size_x, size_y, mean, area = node
        block = self._image[y:y + size_y, x:x + size_x]
        return self._split_function(block)

    def _split(self, node):
        """
        Splits the block defined by the node into four quadrants.
        :param node: A tuple containing (x, y, size_x, size_y, mean, area).
        :return: A list of tuples representing the four child nodes.
        """
        size_x = node[2] / 2
        size_y = node[3] / 2
        area = int(size_x * size_y)
        node_top_left = (node[0], node[1], int(size_x), int(size_y), self._merge_mean(self._image[node[1]:int(node[1] + size_y), node[0]:int(node[0] + size_x)]), area)
        node_top_right = (int(node[0] + size_x), node[1], int(size_x), int(size_y), self._merge_mean(self._image[node[1]:int(node[1] + size_y), int(node[0] + size_x):node[0] + node[2]]), area)
        node_bottom_right = (int(node[0] + size_x), int(node[1] + size_y), int(size_x), int(size_y), self._merge_mean(self._image[int(node[1] + size_y):node[1] + node[3], int(node[0] + size_x):node[0] + node[2]]), area)
        node_bottom_left = (node[0], int(node[1] + size_y), int(size_x), int(size_y), self._merge_mean(self._image[int(node[1] + size_y):node[1] + node[3], node[0]:node[0] + int(size_x)]), area)
        return [node_top_left, node_top_right, node_bottom_right, node_bottom_left]

    def build_quadtree(self) -> Tuple[dict, Tuple[int, int, int, int, Any, int]]:
        """
        Builds the quadtree structure for the image using a dictionary.
        The tuple representing a quadrant will be used as key, while the value will be a list of its children nodes.
        """
        print("Building quadtree...")
        self.quadtree = {}
        task_stack = deque()
        size_x = self._image.shape[1]
        size_y = self._image.shape[0]
        self.root = (0, 0, size_x, size_y, self._merge_mean(self._image), size_x*size_y)  # (x, y, size_x, size_y, mean, area)
        task_stack.append(self.root)
        while len(task_stack) != 0:
            node = task_stack.pop()
            if node[2] > self._min_block_size and node[3] > self._min_block_size and not self._is_homogeneous(node):
                children = self._split(node)
                self.quadtree[node] = children
                for child in children:
                    task_stack.append(child)
            else:
                self.split_result.append(node)
        print("Quadtree built successfully.")
        return self.quadtree, self.root

    def get_split_image(self):
        """Returns the split image"""
        for node in self.split_result:
            x = node[0]
            y = node[1]
            size = node[2]
            block = self._image[y:y + size, x:x + size]
            avg_color = block.mean(axis=(0, 1), keepdims=True)
            self.split_image[y:y + size, x:x + size] = avg_color
        return self.split_image