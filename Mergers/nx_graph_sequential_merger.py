from queue import Queue
from typing import Tuple, Any

import networkx as nx

from Mergers.merger import Merger
from utils.utility_functions import are_contiguous


class NxGraphSequentialMerger(Merger):
    """
    Sequential merger for the split and merge algorithm using a region adjacency graph implemented using nx graph.
    This class implements the merging phase of the algorithm.
    """
    def __init__(self, image, merging_function, **kwargs):
        """
        Initialize the NxGraphSequentialMerger with the image and merging function.
        :param image: ndarray representing the image to be merged.
        :param merging_function: function that defines if two regions should be merged by operating on the
        weighted mean of the merge_mean function result calculated over the quadrants belonging to each region.
        :param kwargs: used for compatibility with other merger implementations.
        """
        super().__init__(image, merging_function, **kwargs)
        self.region_adjacency_graph = None

    def build_region_adjacency_graph(self, root: tuple, quadtree: dict):
        """
        Build the region adjacency graph for the image. Given a quadtree structure, starting from the root node,
        it adds nodes, connects them based on adjacency, and then substitutes the nodes with their children.
        Args:
            root (tuple): The root node of the quadtree.
            quadtree (dict): The quadtree structure represented as a dictionary.
        """
        print("Building region adjacency graph...")
        queue = Queue()
        self.region_adjacency_graph = nx.Graph()
        self.region_adjacency_graph.add_node((root, ))
        queue.put(root)
        while queue.qsize() != 0:
            node = queue.get()
            if node in quadtree:
                children = quadtree[node]
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

    def _merge_nodes(
            self,
            graph_node1: Tuple[Tuple[int, int, int, Any, int], ...],
            graph_node2: Tuple[Tuple[int, int, int, Any, int], ...]
    ) -> Tuple[Tuple[int, int, int, Any, int], ...]:
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