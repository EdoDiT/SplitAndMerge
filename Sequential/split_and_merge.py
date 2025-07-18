import gc
from collections import deque
from typing import Callable, Tuple, List, Any
from queue import Queue

import networkx as nx
import numpy as np
from numpy import ndarray

from utils.utility_functions import are_contiguous


class SequentialSplitAndMerge:
    """A class to perform sequential split and merge on an image.
    This class builds a quadtree from the image, constructs a region adjacency graph,
    and merges regions based on specified functions for splitting and merging.
    """
    def __init__(
            self,
            image: ndarray,
            split_function: Callable,
            min_block_size: int,
            merging_function: Callable,
            merge_mean: Callable = np.mean,
    ) -> None:
        """        Initializes the SequentialSplitAndMerge class.
        Args:
            image (ndarray): The input image to be processed.
            split_function (Callable): Function to determine if a block is homogeneous.
            merging_function (Callable): Function to determine if two blocks can be merged.
            merge_mean (Callable): Function to compute the mean of a block for merging.
            min_block_size (int): Minimum size of blocks to consider for splitting.
        """
        self.split_function = split_function
        self.merging_function = merging_function
        self.merge_mean = merge_mean
        self.image = image
        self.split_image = image.copy()  # Initialize with the original image
        self.merge_image = image.copy()  # Initialize with the original image
        self.root = None
        self.quadtree = dict()
        self.region_adjacency_graph = None
        self.graph_nodes = [] # A single node is a list of tuples, where each tuple represents a block of the image
        # self.graph_edges = dict() # The first tuple of a node is used as primary key for the edges. The table is symmetric. Each entry contains the full tuple list of all the neighbours
        self.min_block_size = min_block_size
        self.split_result = []

    def _is_homogeneous(self, node: Tuple[int, int, int, Any, int]) -> bool:
        """Checks if a block of the image is homogeneous using the split function.
        Args:
            node (Tuple[int, int, int, Any, int]): A tuple representing the block (x, y, size, merge_mean, area).
        Returns:
            bool: True if the block is homogeneous, False otherwise.
        """
        x, y, size, mean, area = node
        block = self.image[y:y+size, x:x+size]
        return self.split_function(block)

    def _split(self, node: Tuple[int, int, int, Any, int]) -> [
        Tuple[int, int, int, Any, int], Tuple[int, int, int, Any, int], Tuple[int, int, int, Any, int],
        Tuple[int, int, int, Any, int]
    ]:
        """Splits a node into four quadrants. node is a tuple of (x, y, size, mean, area)"""
        size = node[2] / 2
        area = size * size
        node_top_left = (
        node[0], node[1], int(size), self.merge_mean(self.image[node[1]:int(node[1] + size), node[0]:int(node[0] + size)]), area)
        node_top_right = (int(node[0] + size), node[1], int(size),
                          self.merge_mean(self.image[node[1]:int(node[1] + size), int(node[0] + size):node[0] + node[2]]), area)
        node_bottom_right = (int(node[0] + size), int(node[1] + size), int(size), self.merge_mean(
            self.image[int(node[1] + size):node[1] + node[2], int(node[0] + size):node[0] + node[2]]), area)
        node_bottom_left = (node[0], int(node[1] + size), int(size),
                            self.merge_mean(self.image[int(node[1] + size):node[1] + node[2], node[0]:node[0] + int(size)]), area)
        return [node_top_left, node_top_right, node_bottom_right, node_bottom_left]

    def build_quadtree(self):
        """Builds a quadtree from the image by recursively splitting it into homogeneous blocks.
        The quadtree is built by checking if a block is homogeneous using the split function.
        If a block is not homogeneous, it is split into four children blocks.
        The process continues until all blocks are homogeneous or the minimum block size is reached.
        """
        self.split_result = []
        print("Building quadtree...")
        task_stack = deque()
        size = self.image.shape[0]
        self.root = (0, 0, size, self.merge_mean(self.image), size*size)  # (x, y, size, mean, area)
        task_stack.append(self.root)
        while len(task_stack) != 0:
            node = task_stack.pop()
            if node[2] > self.min_block_size and not self._is_homogeneous(node):
                children = self._split(node)
                self.quadtree[node] = children
                for child in children:
                    task_stack.append(child)
            else:
                self.split_result.append(node)
        print("Quadtree built successfully.")
        return

    def _are_similar(self, region1: tuple, region2: tuple) -> bool:
        """Checks if two regions are similar by computing the weighted mean based on blocks belonging to each region.
        Args:
            region1 (tuple): A tuple representing the first region (x, y, size).
            region2 (tuple): A tuple representing the second region (x, y, size).
        Returns:
            bool: True if the regions are similar, False otherwise.
        """
        means1 = [n[3] for n in region1]
        means2 = [n[3] for n in region2]
        areas1 = [n[4] for n in region1]
        areas2 = [n[4] for n in region2]
        weighted_means1 = np.average(means1, axis=0, weights=areas1)
        weighted_means2 = np.average(means2, axis=0, weights=areas2)
        return self.merging_function(weighted_means1, weighted_means2)


    def build_region_adjacency_graph(self):
        """Builds a region adjacency graph from the quadtree.
        The graph is built by adding nodes for each region and edges between consecutive children.
        When a new node is added, it checks if it is contiguous with any of father's neighbours and adds edges accordingly."""
        print("Building region adjacency graph...")
        queue = Queue()
        self.region_adjacency_graph = nx.Graph()
        self.region_adjacency_graph.add_node((self.root, ))
        queue.put(self.root)
        while queue.qsize() != 0:
            node = queue.get()
            if node in self.quadtree:
                children = self.quadtree[node]
                for child in children:
                    self.region_adjacency_graph.add_node((child, ))
                    queue.put(child)
                    for neighbour in self.region_adjacency_graph.neighbors((node, )):
                        if are_contiguous(neighbour[0], child):
                            self.region_adjacency_graph.add_edge(neighbour, (child, ))
                for i in range((len(children) - 1)):
                    self.region_adjacency_graph.add_edge((children[i], ), (children[i+1], ))
                self.region_adjacency_graph.add_edge((children[0], ), (children[3], ))
                self.region_adjacency_graph.remove_node((node, ))
        print("Region adjacency graph built successfully.")
        return

    def _merge_nodes(
            self,
            graph_node1: Tuple[Tuple[int, int, int, Any, int]],
            graph_node2: Tuple[Tuple[int, int, int, Any, int]]
    ) -> Tuple[Tuple[int, int, int, Any, int]]:
        """Merges two graph nodes into a new graph node.
        Args:
            graph_node1 (List[Tuple[int, int, int]]): The first graph node to merge.
            graph_node2 (List[Tuple[int, int, int]]): The second graph node to merge.
        Returns:
            List[Tuple[int, int, int]]: The new graph node created by merging the two input nodes.
        """
        new_graph_node = graph_node1 + graph_node2
        self.region_adjacency_graph.add_node(new_graph_node)
        neighbors1 = self.region_adjacency_graph.neighbors(graph_node1)
        neighbors2 = self.region_adjacency_graph.neighbors(graph_node2)
        new_neighbours = set(neighbors1).union(set(neighbors2))
        new_neighbours.remove(graph_node1)
        new_neighbours.remove(graph_node2)
        self.region_adjacency_graph.add_edges_from([(new_graph_node, n) for n in new_neighbours])
        self.region_adjacency_graph.remove_node(graph_node1)
        self.region_adjacency_graph.remove_node(graph_node2)
        return new_graph_node

    # def _merge_nodes(
    #         self,
    #         graph_node1: List[Tuple[int, int, int, Any, int]],
    #         graph_node2: List[Tuple[int, int, int, Any, int]]
    # ) -> List[Tuple[int, int, int, Any, int]]:
    #     """Merges two graph nodes into a new graph node.
    #     Args:
    #         graph_node1 (List[Tuple[int, int, int]]): The first graph node to merge.
    #         graph_node2 (List[Tuple[int, int, int]]): The second graph node to merge.
    #     Returns:
    #         List[Tuple[int, int, int]]: The new graph node created by merging the two input nodes.
    #     """
    #     new_graph_node = graph_node1 + graph_node2
    #     self.graph_nodes.remove(graph_node1)
    #     self.graph_nodes.remove(graph_node2)
    #     self.graph_nodes.append(new_graph_node)
    #     for neighbour in self.graph_edges.get(graph_node2[0]):
    #         self.graph_edges[neighbour[0]].remove(graph_node2)
    #     for neighbour in self.graph_edges.get(graph_node1[0]):
    #         self.graph_edges[neighbour[0]].remove(graph_node1)
    #     self.graph_edges[graph_node2[0]].remove(graph_node1)
    #     neighbours1 = self.graph_edges.get(graph_node1[0])
    #     neighbours2 = self.graph_edges.pop(graph_node2[0])
    #     new_neighbours = []
    #     for neighbour in neighbours2:
    #         if neighbour not in neighbours1:
    #             new_neighbours.append(neighbour)
    #     self.graph_edges[new_graph_node[0]].extend(new_neighbours)
    #     for neighbour in self.graph_edges.get(new_graph_node[0], []):
    #         self.graph_edges[neighbour[0]].extend([new_graph_node])
    #     return new_graph_node

    # def build_region_adjacency_graph(self):
    #     """Builds a region adjacency graph from the quadtree.
    #     The graph is constructed by iterating through the quadtree and adding edges between contiguous blocks.
    #     Each node in the graph represents a block of the image, and edges represent adjacency between blocks.
    #     The graph is built by checking if two blocks are contiguous using the are_contiguous function, or by knowing it from their parent.
    #     """
    #     print("Building region adjacency graph...")
    #     task_stack = deque()
    #     task_stack.append(self.root)
    #     self.graph_edges[self.root] = []
    #     while len(task_stack) != 0:
    #         node = task_stack.pop()
    #         children = self.quadtree.get(node)
    #         if children:
    #             for child in children:
    #                 task_stack.append(child)
    #                 for neighbour in self.graph_edges.get(node, []):
    #                     if are_contiguous(neighbour[0], child):
    #                         self.graph_edges[child] = self.graph_edges.get(child, []) + [neighbour]
    #                         self.graph_edges[neighbour[0]] = self.graph_edges.get(neighbour[0], []) + [[child]]
    #             for i in range((len(children) - 1)):
    #                 self.graph_edges[children[i]] = self.graph_edges.get(children[i], []) + [[children[i+1]]]
    #                 self.graph_edges[children[i+1]] = self.graph_edges.get(children[i+1], []) + [[children[i]]]
    #             self.graph_edges[children[0]] = self.graph_edges.get(children[0], []) + [[children[3]]]
    #             self.graph_edges[children[3]] = self.graph_edges.get(children[3], []) + [[children[0]]]
    #             for neighbour in self.graph_edges.pop(node, []):
    #                 self.graph_edges[neighbour[0]].remove([node])
    #         else:
    #             self.graph_nodes.append([node])
    #     del self.quadtree
    #     gc.collect()
    #     print("Region adjacency graph built successfully.")
    #     return

    # def merge(self):
    #     """Iteratively merges similar regions in the graph until no more merges can be performed.
    #     The merging is done by checking if two contiguous graph nodes are similar."""
    #     print("Merging regions...")
    #     queue = Queue()
    #     for graph_node in self.graph_nodes:
    #         queue.put(graph_node)
    #     while queue.qsize() != 0:
    #         graph_node = queue.get()
    #         if graph_node in self.graph_nodes:
    #             neighbours = self.graph_edges.get(graph_node[0], [])
    #             for neighbour in neighbours:
    #                 if self._are_similar(graph_node, neighbour):
    #                     new_graph_node = self._merge_nodes(graph_node, neighbour)
    #                     queue.put(new_graph_node)
    #                     break
    #     print("Regions merged successfully.")
    #     return

    def merge(self):
        """Iteratively merges similar regions in the graph until no more merges can be performed.
        The merging is done by checking if two contiguous graph nodes are similar."""
        print("Merging regions...")
        queue = Queue()
        for graph_node in self.region_adjacency_graph.nodes():
            queue.put(graph_node)
        while queue.qsize() != 0:
            graph_node = queue.get()
            if graph_node in self.region_adjacency_graph:
                neighbours = list(self.region_adjacency_graph.neighbors(graph_node))
                for neighbour in neighbours:
                    if self._are_similar(graph_node, neighbour):
                        new_graph_node = self._merge_nodes(graph_node, neighbour)
                        queue.put(new_graph_node)
                        break
        print("Regions merged successfully.")
        for graph_node in self.region_adjacency_graph.nodes():
            self.graph_nodes.append(graph_node)
        return

    def get_split_image(self):
        """Returns the split image"""
        for node in self.split_result:
            x = node[0]
            y = node[1]
            size = node[2]
            block = self.image[y:y + size, x:x + size]
            avg_color = block.mean(axis=(0, 1), keepdims=True)
            self.split_image[y:y + size, x:x + size] = avg_color
        return self.split_image

    def get_merge_image(self):
        """Returns the merged image"""
        for node in self.graph_nodes:
            if len(self.image.shape) == 2:  # Grayscale image
                avg_color = 0
            else:  # RGB image
                avg_color = [0, 0, 0]
            total_area = 0
            for quad_node in node:
                region = self.image[quad_node[1]:quad_node[1] + quad_node[2], quad_node[0]:quad_node[0] + quad_node[2]]
                region_color = region.mean(axis=(0, 1))
                area = quad_node[2] * quad_node[2]
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
                size = quad_node[2]
                if np.isscalar(avg_color):
                    self.merge_image[y:y + size, x:x + size] = np.full((size, size), avg_color)
                else:
                    self.merge_image[y:y + size, x:x + size] = np.full((size, size, 3), avg_color)
        return self.merge_image

    def process_image(self)-> Tuple[ndarray, ndarray]:
        """Processes the image by building the quadtree, merging regions, and returning the split and merged images.
        Returns:
            Tuple[ndarray, ndarray]: The split and merged images.
        """
        self.build_quadtree()
        self.build_region_adjacency_graph()
        self.merge()
        return self.get_split_image(), self.get_merge_image()