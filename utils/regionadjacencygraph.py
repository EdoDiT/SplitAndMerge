import networkx as nx
import numpy as np

from utils.quadnode import QuadNode


class RegionAdjacencyGraph:
    def __init__(self, quadtree: QuadNode):
        self.graph = nx.Graph()
        self.quadtree = quadtree
        self._build_graph(self.quadtree.root)

    def _build_graph(self, node: QuadNode):
        """Recursively adds nodes to the graph."""
        if node.is_leaf():
            self.graph.add_node((node.x, node.y, node.size), image=node.image)
        else:
            for child in node.children:
                self._build_graph(child)

    def merge_regions(self, threshold: float = 15.0):
        """
        Merges adjacent regions that are similar.
        """
        merged_nodes = set()
        for node1, node2 in self.graph.edges:
            if np.abs(np.mean(self.graph.nodes[node1]['image']) - np.mean(self.graph.nodes[node2]['image'])) < threshold:
                merged_nodes.add(node1)
                merged_nodes.add(node2)

        """TODO: Finish mergin process and check if this is correct"""
        for node in merged_nodes:
            self.graph.remove_node(node)
