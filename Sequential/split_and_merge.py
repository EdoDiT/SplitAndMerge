from typing import Callable
import networkx as nx
from queue import Queue

from numpy import ndarray

from utils.quadnode import QuadNode


class SequentialSplitAndMerge:
    def __init__(self, image: ndarray, split_function: Callable, merging_function: Callable, min_block_size: int) -> None:
        self.split_function = split_function
        self.merging_function = merging_function
        self.region_adjacency_graph = None
        self.image = image
        self.quadtree = None
        self.min_block_size = min_block_size
        self.split_result = None

    def _is_homogeneous(self, block1: ndarray) -> bool:
        if not self.split_function(block1):
            return False
        return True

    def _are_similar(self, block1: ndarray, block2: ndarray) -> bool:
        if not self.merging_function(block1, block2):
            return False
        return True


    def _build_quadtree(self):
        queue = Queue() # TODO: check if thread safe
        self.quadtree = QuadNode(x=0, y=0, size=self.image.shape[0])
        self.split_result = []
        queue.put(self.quadtree)
        while queue.qsize() != 0:
            node = queue.get()
            if not self._is_homogeneous(node) and node.size > self.min_block_size:
                node_top_left, node_top_right, node_bottom_right, node_bottom_left = QuadNode.split(node)
                queue.put(node_top_left, node_top_right, node_bottom_right, node_bottom_left)
            else:
                self.split_result.append(node)
        return

    def _build_region_adjacency_graph(self):
        queue = Queue()
        self.region_adjacency_graph = nx.Graph()
        self.region_adjacency_graph.add_node([self.quadtree])
        queue.put(self.quadtree)
        while queue.qsize() != 0:
            node = queue.get()
            if node.children:
                children = node.children
                for child in children:
                    self.region_adjacency_graph.add_node([child])
                    queue.put(child)
                    for neighbour in self.region_adjacency_graph.neighbors([node]):
                        if QuadNode.are_contiguous(neighbour, child):
                            self.region_adjacency_graph.add_edge(neighbour, [child])
                for i in (len(children) - 1):
                    self.region_adjacency_graph.add_edge([children[i]], [children[i+1]])
                self.region_adjacency_graph.remove_node([node])
        self.region_adjacency_graph = self.region_adjacency_graph
        return

    def _merge(self) -> 'SequentialSplitAndMerge':
        self._build_region_adjacency_graph()
        queue = Queue()
        for graph_node in self.region_adjacency_graph.nodes():
            queue.put(graph_node)
        while queue.qsize() != 0:
            graph_node = queue.get()
            if graph_node in self.region_adjacency_graph:
                for neighbour in self.region_adjacency_graph.neighbors(graph_node):
                    if self.merging_function(graph_node, neighbour):
                        new_graph_node = graph_node + neighbour
                        self.region_adjacency_graph.add_node(new_graph_node)
                        for n in self.region_adjacency_graph.neighbors(graph_node):
                            self.region_adjacency_graph.add_edge(new_graph_node, n)
                        for n in self.region_adjacency_graph.neighbors(neighbour):
                            self.region_adjacency_graph.add_edge(new_graph_node, n)
                        self.region_adjacency_graph.remove_node(graph_node)
                        self.region_adjacency_graph.remove_node(neighbour)
                        queue.put(new_graph_node)
                    else:
                        pass
            else:
                pass
        pass