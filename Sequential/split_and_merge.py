from typing import Callable, Tuple
import networkx as nx
from queue import Queue

import numpy as np
from numpy import ndarray

from utils.quadnode import QuadNode


class SequentialSplitAndMerge:
    def __init__(
            self,
            image: ndarray,
            split_function: Callable,
            merging_function: Callable,
            merge_mean: Callable,
            min_block_size: int
    ) -> None:
        """Initializes the SequentialSplitAndMerge class.
        Args:
            image (ndarray): The input image to be segmented.
            split_function (Callable): A function that determines if a block should be split.
            merging_function (Callable): A function that determines if two blocks can be merged.
            min_block_size (int): The minimum size of a block to be considered for splitting.
        """
        self.split_function = split_function
        self.merging_function = merging_function
        self.merge_mean = merge_mean
        self.region_adjacency_graph = None
        self.image = image
        self.split_image = image.copy()  # Initialize with the original image
        self.merge_image = image.copy()  # Initialize with the original image
        self.quadtree = None
        self.min_block_size = min_block_size
        self.split_result = []
        self.merge_result = []

    def _is_homogeneous(self, node: QuadNode) -> bool:
        x = node.x
        y = node.y
        size = node.size
        block = self.image[y:y+size, x:x+size]
        return self.split_function(block)

    def _are_similar(self, region1: tuple, region2: tuple) -> bool:
        # Aggregate all pixels from both regions
        means1 = [self.merge_mean(self.image[n.y:n.y+n.size, n.x:n.x+n.size]) for n in region1]
        means2 = [self.merge_mean(self.image[n.y:n.y+n.size, n.x:n.x+n.size]) for n in region2]
        areas1 = [n.size * n.size for n in region1]
        areas2 = [n.size * n.size for n in region2]
        weighted_means1 = np.average(means1, axis=0, weights=areas1)
        weighted_means2 = np.average(means2, axis=0, weights=areas2)
        return self.merging_function(weighted_means1, weighted_means2)

    # 25 seconds per iteration on 1024x1024
    def build_quadtree(self):
        """Builds a quadtree from the image by recursively splitting it into homogeneous blocks.
        Saves leaf nodes in self.split_result, and tree in self.quadtree."""
        queue = Queue() # TODO: check if thread safe
        self.quadtree = QuadNode(x=0, y=0, size=self.image.shape[0])
        queue.put(self.quadtree)
        while queue.qsize() != 0:
            node = queue.get()
            if node.size > self.min_block_size and not self._is_homogeneous(node):
                node_top_left, node_top_right, node_bottom_right, node_bottom_left = QuadNode.split(node)
                queue.put(node_top_left)
                queue.put(node_top_right)
                queue.put(node_bottom_right)
                queue.put(node_bottom_left)
            else:
                self.split_result.append(node)
        return

    def build_region_adjacency_graph(self):
        """Builds a region adjacency graph from the quadtree.
        The graph is built by adding nodes for each region and edges between consecutive children.
        When a new node is added, it checks if it is contiguous with any of father's neighbours and adds edges accordingly."""
        queue = Queue()
        self.region_adjacency_graph = nx.Graph()
        self.region_adjacency_graph.add_node((self.quadtree, ))
        queue.put(self.quadtree)
        while queue.qsize() != 0:
            node = queue.get()
            if node.children:
                children = node.children
                for child in children:
                    self.region_adjacency_graph.add_node((child, ))
                    queue.put(child)
                    for neighbour in self.region_adjacency_graph.neighbors((node, )):
                        if QuadNode.are_contiguous(neighbour[0], child):#TODO: Find alternative since it doesn't work with not unary tuples
                            self.region_adjacency_graph.add_edge(neighbour, (child, ))
                for i in range((len(children) - 1)):
                    self.region_adjacency_graph.add_edge((children[i], ), (children[i+1], ))
                self.region_adjacency_graph.remove_node((node, ))
        return

    def _merge_nodes(self, graph_node1: Tuple[QuadNode], graph_node2: Tuple[QuadNode]) -> Tuple[QuadNode]:
        # if graph_node1 == graph_node2:
        #     return graph_node1
        # if not graph_node1 or not graph_node2:
        #     return graph_node1 if graph_node1 else graph_node2
        new_graph_node = graph_node1 + graph_node2
        self.region_adjacency_graph.add_node(new_graph_node)
        graph_node_neighbours = list(self.region_adjacency_graph.neighbors(graph_node1))
        for n in graph_node_neighbours:
            self.region_adjacency_graph.add_edge(new_graph_node, n)
        for n in self.region_adjacency_graph.neighbors(graph_node2):
            self.region_adjacency_graph.add_edge(new_graph_node, n)
        self.region_adjacency_graph.remove_node(graph_node1)
        self.region_adjacency_graph.remove_node(graph_node2)
        return new_graph_node

    def merge(self):
        self.build_region_adjacency_graph()
        queue = Queue()
        for graph_node in self.region_adjacency_graph.nodes():
            queue.put(graph_node)
        while queue.qsize() != 0:
            graph_node = queue.get()
            if graph_node in self.region_adjacency_graph:
                neighbours = list(self.region_adjacency_graph.neighbors(graph_node))
                merged = False
                for neighbour in neighbours:
                    if self._are_similar(graph_node, neighbour) and graph_node != neighbour:
                        merged = True
                        new_graph_node = self._merge_nodes(graph_node, neighbour)
                        queue.put(new_graph_node)
                        break
        for graph_node in self.region_adjacency_graph.nodes():
            self.merge_result.append(graph_node)
        return

    def get_split_image(self):
        """Returns the split image"""
        for node in self.split_result:
            x = node.x
            y = node.y
            size = node.size
            block = self.image[y:y + size, x:x + size]
            avg_color = block.mean(axis=(0, 1), keepdims=True)
            self.split_image[y:y + size, x:x + size] = avg_color
        return self.split_image

    def get_merge_image(self):
        """Returns the merged image"""
        for node in self.merge_result:
            if len(self.image.shape) == 2:  # Grayscale image
                avg_color = 0
            else:  # RGB image
                avg_color = [0, 0, 0]
            total_area = 0
            for quad_node in node:
                region = self.image[quad_node.y:quad_node.y + quad_node.size, quad_node.x:quad_node.x + quad_node.size]
                region_color = region.mean(axis=(0, 1))
                area = quad_node.size * quad_node.size
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
                x = quad_node.x
                y = quad_node.y
                size = quad_node.size
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