import multiprocessing as mp
from collections import deque
from multiprocessing.managers import SyncManager
from time import sleep
from typing import Callable, List, Tuple, Any

import numpy as np
from numpy import ndarray

from Mergers.dict_graph_sequential_merger import DictGraphSequentialMerger
from utils.utility_functions import shuffle_queue


class DictGraphParallelMerger(DictGraphSequentialMerger):
    """A parallel implementation of the DictGraphSequentialMerger that uses multiprocessing to merge regions in a graph.
    This class extends DictGraphSequentialMerger and overrides the merge method to use multiprocessing for merging regions.
    It uses a shared queue for management, a processing list to track nodes being processed along with neighbours,
    and shared data structures(shared_graph_nodes and shared_graph_edges) for graph representation.
    """
    def __init__(self,
                 image: ndarray,
                 merging_function: Callable,
                 num_workers: int = None,
                 mp_manager: None = None,
                 **kwargs
                 ):
        super().__init__(image, merging_function, **kwargs)
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.mp_manager = mp_manager
        if not mp_manager:
            SyncManager.register(
                'Deque',
                callable=deque,
                exposed=('append', 'appendleft', 'pop', 'popleft', '__len__', '__bool__')
            )
        self.manager = mp_manager if mp_manager is not None else mp.Manager()
        self.shared_graph_nodes = None
        self.shared_graph_edges = None
        self.shared_processing_list = None
        self.task_queue = None
        self.nodes_lock = None
        pass

    @staticmethod
    def _are_similar_shared(
            region1: List[Tuple[int, int, int, int, Any, int]],
            region2: List[Tuple[int, int, int, int, Any, int]],
            merging_function: Callable
    ) -> bool:
        """Checks if two regions are similar using shared image data.
        Args:
            region1 (List[Tuple[int, int, int, int, Any, int]]): The first region to compare. It's a list of the
            quadrant that compose it.
            region2 (List[Tuple[int, int, int, int, Any, int]]): The second region to compare. It's a list of the
            quadrant that compose it.
            merging_function (Callable): The function used to compare the mean computed across all quadrants of the
            input merge_mean function.
        Returns:
            bool: True if the regions are similar, False otherwise.
        """
        # Convert shared image back to numpy array for processing
        means1 = [n[4] for n in region1]
        means2 = [n[4] for n in region2]
        areas1 = [n[5] for n in region1]
        areas2 = [n[5] for n in region2]
        weighted_means1 = np.average(means1, axis=0, weights=areas1)
        weighted_means2 = np.average(means2, axis=0, weights=areas2)
        return merging_function(weighted_means1, weighted_means2)

    @staticmethod
    def _merge_nodes_shared(
            graph_node1: List[Tuple[int, int, int]],
            graph_node2: List[Tuple[int, int, int]],
            shared_graph_nodes,
            shared_graph_edges
    ) -> List[Tuple[int, int, int]]:
        """Merges two graph nodes into a new graph node using shared data structures, and deletes the original two.
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
        """Worker function for parallel merging of regions. Keeps consuming nodes to process from the task queue.
        Given a node, it checks if it still exists, if it's not being processed, or too close to a node being processed.
        Then, if it can be merged with any of its neighbours, it's merged. Different appends and exponential backoff
        are used to handle conflicts and retries. Working set is used to track nodes being processed by each worker at
        each iteration.
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
                    if self._are_similar_shared(graph_node, neighbour, self.merging_function):
                        # Lock nodes to check for neigbour's neighbours conflicts before merging
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
        Sets up a multiprocessing manager and shared data structures, then creates worker processes to handle merging.
        """
        print("Merging regions...")

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