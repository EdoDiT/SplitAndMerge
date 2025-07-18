import gc
import multiprocessing as mp
import time
from collections import deque
from logging import warning
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager
from time import sleep
from typing import Callable, Tuple, List, Any

import numpy as np
from numpy import ndarray

from utils.utility_functions import are_contiguous, shuffle_queue


class ParallelSplitAndMerge:
    """A class to perform sequential split and merge on an image.
    This class builds a quadtree from the image, constructs a region adjacency graph,
    and merges regions based on specified functions for splitting and merging.
    """
    def __init__(
            self,
            image: ndarray,
            split_function: Callable,
            merging_function: Callable,
            merge_mean: Callable,
            min_block_size: int,
            num_workers: int = None
    ) -> None:
        """        Initializes the SequentialSplitAndMerge class.
        Args:
            image (ndarray): The input image to be processed.
            split_function (Callable): Function to determine if a block is homogeneous.
            merging_function (Callable): Function to determine if two blocks can be merged.
            merge_mean (Callable): Function to compute the mean of a block for merging.
            min_block_size (int): Minimum size of blocks to consider for splitting.
            num_workers (int): Number of worker processes for parallel merging. If None, uses CPU count.
        """
        self.split_function = split_function
        self.merging_function = merging_function
        self.merge_mean = merge_mean
        self.image = image
        self.split_image = image.copy()  # Initialize with the original image
        self.merge_image = image.copy()  # Initialize with the original image
        self.root = None
        self.quadtree = dict()
        self.graph_nodes = [] # A single node is a list of tuples, where each tuple represents a block of the image
        self.graph_edges = dict() # The first tuple of a node is used as primary key for the edges. The table is symmetric. Each entry contains the full tuple list of all the neighbours
        self.min_block_size = min_block_size
        self.split_result = []

        # Parallel processing setup
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.manager = None
        self.shared_graph_nodes = None
        self.shared_graph_edges = None
        self.shared_processing_list = None
        self.task_queue = None
        self.nodes_lock = None
        self.shared_image = None
        self.image_shape = image.shape
        self.image_dtype = image.dtype
        self._setup_shared_memory()

    def _setup_shared_memory(self):
        """Sets up shared memory for the image data."""
        # Create shared memory buffer for the image
        image_bytes = self.image.nbytes
        self.shared_image = shared_memory.SharedMemory(create=True, size=image_bytes)

        # Create numpy array from shared memory
        shared_array = np.ndarray(self.image_shape, dtype=self.image_dtype, buffer=self.shared_image.buf)
        shared_array[:] = self.image[:]

    def _cleanup_shared_memory(self):
        """Cleans up shared memory resources."""
        if self.shared_image is not None:
            self.shared_image.close()
            self.shared_image.unlink()

    def __del__(self):
        """Destructor to clean up shared memory."""
        try:
            self._cleanup_shared_memory()
        except Exception as e:
            warning(f"Failed to cleanup shared memory: {e}")

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

    def build_region_adjacency_graph(self):
        """Builds a region adjacency graph from the quadtree.
        The graph is constructed by iterating through the quadtree and adding edges between contiguous blocks.
        Each node in the graph represents a block of the image, and edges represent adjacency between blocks.
        The graph is built by checking if two blocks are contiguous using the are_contiguous function, or by knowing it from their parent.
        """
        print("Building region adjacency graph...")
        task_stack = deque()
        task_stack.append(self.root)
        self.graph_edges[self.root] = []
        while len(task_stack) != 0:
            node = task_stack.pop()
            children = self.quadtree.get(node)
            if children:
                for child in children:
                    task_stack.append(child)
                    for neighbour in self.graph_edges.get(node, []):
                        if are_contiguous(neighbour[0], child):
                            self.graph_edges[child] = self.graph_edges.get(child, []) + [neighbour]
                            self.graph_edges[neighbour[0]] = self.graph_edges.get(neighbour[0], []) + [[child]]
                for i in range((len(children) - 1)):
                    self.graph_edges[children[i]] = self.graph_edges.get(children[i], []) + [[children[i+1]]]
                    self.graph_edges[children[i+1]] = self.graph_edges.get(children[i+1], []) + [[children[i]]]
                self.graph_edges[children[0]] = self.graph_edges.get(children[0], []) + [[children[3]]]
                self.graph_edges[children[3]] = self.graph_edges.get(children[3], []) + [[children[0]]]
                for neighbour in self.graph_edges.pop(node, []):
                    self.graph_edges[neighbour[0]].remove([node])
            else:
                self.graph_nodes.append([node])
        del self.quadtree
        gc.collect()
        print("Region adjacency graph built successfully.")
        return


    def _are_similar_shared(self, region1: tuple, region2: tuple) -> bool:
        """Checks if two regions are similar using shared image data.
        Args:
            region1 (tuple): A tuple representing the first region (x, y, size).
            region2 (tuple): A tuple representing the second region (x, y, size).
        Returns:
            bool: True if the regions are similar, False otherwise.
        """
        # Convert shared image back to numpy array for processing
        means1 = [n[3] for n in region1]
        means2 = [n[3] for n in region2]
        areas1 = [n[4] for n in region1]
        areas2 = [n[4] for n in region2]
        weighted_means1 = np.average(means1, axis=0, weights=areas1)
        weighted_means2 = np.average(means2, axis=0, weights=areas2)
        return self.merging_function(weighted_means1, weighted_means2)

    @staticmethod
    def _merge_nodes_shared(
            graph_node1: List[Tuple[int, int, int]],
            graph_node2: List[Tuple[int, int, int]],
            shared_graph_nodes,
            shared_graph_edges
    ) -> List[Tuple[int, int, int]]:
        """Merges two graph nodes into a new graph node using shared data structures.
        Args:
            graph_node1 (List[Tuple[int, int, int]]): The first graph node to merge.
            graph_node2 (List[Tuple[int, int, int]]): The second graph node to merge.
            shared_graph_nodes: Shared list of graph nodes.
            shared_graph_edges: Shared dictionary of graph edges.
        Returns:
            List[Tuple[int, int, int]]: The new graph node created by merging the two input nodes.
        """
        print(f"{mp.current_process().name} merging nodes {graph_node1[0]} and {graph_node2[0]}")
        new_graph_node = graph_node1 + graph_node2
        neighbours1 = shared_graph_edges.get(graph_node1[0]).copy()
        neighbours2 = shared_graph_edges.pop(graph_node2[0])
        neighbours1.remove(graph_node2)
        neighbours2.remove(graph_node1)
        new_neighbours = []
        for neighbour in neighbours2:
            if neighbour not in neighbours1:
                new_neighbours.append(neighbour)
        for neighbour in neighbours1:
            pruned_list1 = shared_graph_edges[neighbour[0]].copy()
            pruned_list1.remove(graph_node1)
            shared_graph_edges[neighbour[0]] = pruned_list1
        for neighbour in neighbours2:
            pruned_list2 = shared_graph_edges[neighbour[0]].copy()
            pruned_list2.remove(graph_node2)
            shared_graph_edges[neighbour[0]] = pruned_list2
        neighbours1.extend(new_neighbours)
        shared_graph_edges[new_graph_node[0]] = neighbours1
        for neighbour in neighbours1:
            if neighbour == graph_node2:
                print(f"WARNING: THIS SHOULDN'T HAPPEN: {mp.current_process().name} has {graph_node2[0]} still in neigbours list of {graph_node1[0]}")
            if neighbour == graph_node1:
                print(f"WARNING: THIS SHOULDN'T HAPPEN: {mp.current_process().name} has {graph_node1[0]} still in neigbours list of {graph_node2[0]}")
            updated_list = shared_graph_edges[neighbour[0]].copy()
            updated_list.append(new_graph_node)
            shared_graph_edges[neighbour[0]] = updated_list
        shared_graph_nodes.remove(graph_node1)
        shared_graph_nodes.remove(graph_node2)
        shared_graph_nodes.append(new_graph_node)
        return new_graph_node

    def worker(
            self,
            task_queue,
            shared_graph_nodes,
            shared_graph_edges,
            shared_processing_list,
            nodes_lock,
            retry_counts):
        """Worker function for parallel merging of regions.
        Args:
            task_queue: Shared queue of tasks.
            shared_graph_nodes: Shared list of graph nodes.
            shared_graph_edges: Shared dictionary of graph edges.
            shared_processing_list: Shared list of nodes being processed.
            nodes_lock: Lock for graph_nodes and processing_list.
            retry_counts: Shared dictionary to track retry counts for nodes.
        """
        base = int(mp.current_process().name.split('-')[-1])

        while True:
            # Initialize control flags and variables
            sleep_time = 0
            node_conflict = False
            neighbour_conflict = False
            merged_node = False
            new_graph_node = None
            working_set = set()

            # Step 1: Get node from queue with lock
            with nodes_lock:
                if len(task_queue) == 0:
                    break
                print(f"{mp.current_process().name}: Queue length: {len(task_queue)}")
                graph_node = task_queue.pop()

            # Step 2: Check for conflicts and process node/merge
            if graph_node not in shared_graph_nodes:
                # Node was deleted, skip processing
                continue

            # Check if any neighbour is being processed or if this node is already being processed
            neighbours = shared_graph_edges.get(graph_node[0], [])
            conflict = False

            with nodes_lock:
                for neighbour in neighbours:
                    if neighbour[0] in shared_processing_list:
                        conflict = True
                        break

                if graph_node[0] in shared_processing_list or conflict:
                    # Mark for requeuing due to conflict
                    node_conflict = True
                    retry_counts[graph_node[0]] = retry_counts.get(graph_node[0], 0) + 1
                    sleep_time = 0.001 * (base ** retry_counts[graph_node[0]])
                else:
                    # Add to processing list
                    working_set.add(graph_node[0])
                    working_set.update(neighbour[0] for neighbour in neighbours)
                    shared_processing_list.extend(working_set)

            # Process the node if no initial conflict
            if not node_conflict:
                # Find similar neighbours and merge
                for neighbour in neighbours:
                    if self._are_similar_shared(graph_node, neighbour):
                        # Lock nodes to perform merge
                        with nodes_lock:
                            # Check if any of neighbour's neighbours (except current node) are being processed
                            neighbour_neighbours = shared_graph_edges.get(neighbour[0], [])
                            conflict = False
                            for nn in neighbour_neighbours:
                                if nn[0] not in working_set and nn[0] in shared_processing_list:
                                    conflict = True
                                    break

                            if conflict:
                                # Mark for requeuing due to neighbour conflict
                                neighbour_conflict = True
                                retry_counts[graph_node[0]] = retry_counts.get(graph_node[0], 0) + 1
                                sleep_time = 0.001 * (base ** retry_counts[graph_node[0]])
                                break
                            else:
                                shared_processing_list.extend([nn[0] for nn in neighbour_neighbours if nn[0] not in working_set])
                                working_set.update(nn[0] for nn in neighbour_neighbours)


                        # Perform merge if no conflict
                        if not node_conflict and not neighbour_conflict:
                            new_graph_node = self._merge_nodes_shared(graph_node, neighbour, shared_graph_nodes, shared_graph_edges)
                            merged_node = True
                            break

                # Clean up working set from processing list
                with nodes_lock:
                    for node_id in working_set:
                        shared_processing_list.remove(node_id)

            # Step 3: Sleep if needed
            if sleep_time > 0:
                sleep_time = min(sleep_time, 1)  # Cap sleep time to avoid excessive delays
                print(f"{mp.current_process().name} sleeping for {sleep_time:.3f} seconds")
                sleep(sleep_time)

            # Step 4: Reinsert node in queue or insert new node
            with nodes_lock:
                if node_conflict:
                    task_queue.appendleft(graph_node)
                elif neighbour_conflict and sleep_time<1:
                    task_queue.append(graph_node)
                elif neighbour_conflict and sleep_time>=1:
                    task_queue.appendleft(graph_node)
                elif merged_node:
                    retry_counts[new_graph_node[0]] = 0
                    task_queue.append(new_graph_node)

        return

    def merge(self):
        """Iteratively merges similar regions in the graph until no more merges can be performed.
        The merging is done by checking if two contiguous graph nodes are similar using parallel processing."""
        print("Merging regions...")
        # "fragile" workaround
        SyncManager.register(
            'Deque',
            callable=deque,
            exposed=('append', 'appendleft', 'pop', 'popleft', '__len__', '__bool__')
        )
        # Set up multiprocessing manager and shared data structures
        self.manager = mp.Manager()

        # Create shared graph structures
        self.shared_graph_nodes = self.manager.list(self.graph_nodes)
        self.shared_graph_edges = self.manager.dict(self.graph_edges)
        self.shared_processing_list = self.manager.list()
        retry_counts = self.manager.dict()  # Track retry counts for nodes

        # Create task queue and locks
        self.task_queue = self.manager.Deque()
        self.nodes_lock = self.manager.Lock()

        # Populate initial task queue
        shuffle_queue(self.graph_nodes, self.num_workers, self.task_queue)

        # Create and start worker processes
        processes = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self.worker,
                args=(
                    self.task_queue,
                    self.shared_graph_nodes,
                    self.shared_graph_edges,
                    self.shared_processing_list,
                    self.nodes_lock,
                    retry_counts
                )
            )
            processes.append(p)
            p.start()

        # Wait for all workers to complete
        for p in processes:
            p.join()

        # Copy results back to instance variables
        self.graph_nodes = list(self.shared_graph_nodes)
        self.graph_edges = dict(self.shared_graph_edges)

        print("Regions merged successfully.")
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
            for quadrant in node:
                region = self.image[quadrant[1]:quadrant[1] + quadrant[2], quadrant[0]:quadrant[0] + quadrant[2]]
                region_color = region.mean(axis=(0, 1))
                area = quadrant[2] * quadrant[2]
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
            for quadrant in node:
                x = quadrant[0]
                y = quadrant[1]
                size = quadrant[2]
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