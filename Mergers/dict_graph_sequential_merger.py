from collections import deque
from queue import Queue
from typing import List, Any, Tuple

from networkx.classes import neighbors

from Mergers.merger import Merger
from utils.utility_functions import are_contiguous


class DictGraphSequentialMerger(Merger):
    """
    Sequential merger for the split and merge algorithm using a dictionary-based region adjacency graph.
    This class implements the merging phase of the algorithm.
    """
    def __init__(self, image, merging_function, **kwargs):
        super().__init__(image, merging_function, **kwargs)
        self.graph_nodes = []
        self.graph_edges = {}

    def build_region_adjacency_graph(self, root: Tuple[int, int, int, int, Any, int], quadtree: dict):
        """Builds a region adjacency graph from the quadtree.
        The graph is constructed by iterating through the quadtree and adding edges between contiguous blocks.
        Each node in the graph represents a block of the image, and edges represent adjacency between blocks.
        The graph is built by checking if two blocks are contiguous using the are_contiguous function, or by knowing it from their parent.
        """
        self.graph_nodes = []
        self.graph_edges = {}
        print("Building region adjacency graph...")
        task_stack = deque()
        task_stack.append(root)
        self.graph_edges[root] = []
        while len(task_stack) != 0:
            node = task_stack.pop()
            children = quadtree.get(node)
            if children:
                for child in children:
                    task_stack.append(child)
                    for neighbour in self.graph_edges.get(node, []):
                        if are_contiguous(neighbour[0], child):
                            self.graph_edges[child] = self.graph_edges.get(child, []) + [neighbour]
                            self.graph_edges[neighbour[0]] = self.graph_edges.get(neighbour[0], []) + [[child]]
                for i in range((len(children) - 1)):
                    self.graph_edges[children[i]] = self.graph_edges.get(children[i], []) + [[children[i + 1]]]
                    self.graph_edges[children[i + 1]] = self.graph_edges.get(children[i + 1], []) + [[children[i]]]
                self.graph_edges[children[0]] = self.graph_edges.get(children[0], []) + [[children[3]]]
                self.graph_edges[children[3]] = self.graph_edges.get(children[3], []) + [[children[0]]]
                for neighbour in self.graph_edges.pop(node, []):
                    self.graph_edges[neighbour[0]].remove([node])
            else:
                self.graph_nodes.append([node])
        print("Region adjacency graph built successfully.")
        return

    def merge(self):
        """Iteratively merges similar regions in the graph until no more merges can be performed.
        The merging is done by checking if two contiguous graph nodes are similar."""
        print("Merging regions...")
        queue = Queue()
        for graph_node in self.graph_nodes:
            queue.put(graph_node)
        while queue.qsize() != 0:
            graph_node = queue.get()
            if graph_node in self.graph_nodes:
                neighbours = self.graph_edges.get(graph_node[0], [])
                for neighbour in neighbours:
                    if self._are_similar(graph_node, neighbour):
                        new_graph_node = self._merge_nodes(graph_node, neighbour)
                        queue.put(new_graph_node)
                        break
        print("Regions merged successfully.")
        return

    def _merge_nodes(
            self,
            graph_node1: List[Tuple[int, int, int, int, Any, int]],
            graph_node2: List[Tuple[int, int, int, int, Any, int]]
    ) -> List[Tuple[int, int, int, int, Any, int]]:
        """Merges two graph nodes into a new graph node.
        Args:
            graph_node1 (List[Tuple[int, int, int]]): The first graph node to merge.
            graph_node2 (List[Tuple[int, int, int]]): The second graph node to merge.
        Returns:
            List[Tuple[int, int, int]]: The new graph node created by merging the two input nodes.
        """
        new_graph_node = graph_node1 + graph_node2
        self.graph_nodes.remove(graph_node1)
        self.graph_nodes.remove(graph_node2)
        self.graph_nodes.append(new_graph_node)
        for neighbour in self.graph_edges.get(graph_node2[0]):
            self.graph_edges[neighbour[0]].remove(graph_node2)
        for neighbour in self.graph_edges.get(graph_node1[0]):
            self.graph_edges[neighbour[0]].remove(graph_node1)
        self.graph_edges[graph_node2[0]].remove(graph_node1)
        neighbours1 = self.graph_edges.get(graph_node1[0])
        neighbours2 = self.graph_edges.pop(graph_node2[0])
        new_neighbours = []
        for neighbour in neighbours2:
            if neighbour not in neighbours1:
                new_neighbours.append(neighbour)
        self.graph_edges[new_graph_node[0]].extend(new_neighbours)
        for neighbour in self.graph_edges.get(new_graph_node[0], []):
            self.graph_edges[neighbour[0]].extend([new_graph_node])
        return new_graph_node