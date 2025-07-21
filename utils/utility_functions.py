from collections import deque
from multiprocessing.managers import BaseManager
from queue import Queue
from typing import Tuple, List, Any

import numpy as np


def are_contiguous(node1: Tuple[int, int, int, Any, int], node2: Tuple[int, int, int, Any, int]) -> bool:
    """Checks if two nodes are contiguous. Nodes are tuples of (x, y, size, mean, area)"""
    if node1[0] <= node2[0]:
        pass
    else:
        node3 = node2
        node2 = node1
        node1 = node3
        pass
    if node1[0] + node1[2] == node2[0]:
        if node1[1]+node1[2] <= node2[1]:
            return False
        elif node1[1] >= node2[1]+node2[2]:
            return False
        else:
            return True
    elif (node1[1] + node1[2] == node2[1]) or (node2[1] + node2[2] == node1[1]):
        if node1[0]+node1[2] <= node2[0]:
            return False
        elif node1[0] >= node2[0]+node2[2]:
            return False
        else:
            return True
    else:
        return False

def shuffle_queue(
        list_of_nodes: List[List[Tuple[int, int, int, int, Any, int]]],
        num_workers: int,
        task_queue: List
) -> List:
    """ Shuffles a list of nodes into sublists for parallel processing. When building the region adjacency graph,
    the tree is processed in a depth first manner. This leads to neighbour nodes being close to each other in the list,
    resultin in many conflicts during parallel processing. This function tries to mitigate this by shuffling the nodes
    in a round-robin fashion according to the number of workers.
    :param list_of_nodes: list of nodes to be shuffled, where each node is a list of
    tuple(x, y, size_x, size_y, mean, area)
    :param num_workers: Number of workers in the merging process.
    :param task_queue: List to append shuffled nodes to.
    :return: the shuffled task queue.
    """
    chunk_size = len(list_of_nodes) // num_workers
    remainder = len(list_of_nodes) % num_workers
    worker_sublists = []
    start_idx = 0
    for worker_id in range(num_workers):
        # Add extra element to first 'remainder' workers
        current_chunk_size = chunk_size + (1 if worker_id < remainder else 0)
        end_idx = start_idx + current_chunk_size
        worker_sublists.append(list_of_nodes[start_idx:end_idx])
        start_idx = end_idx

    # Round-robin insertion until all sublists are empty
    indices = [0] * num_workers
    while any(indices[i] < len(worker_sublists[i]) for i in range(num_workers)):
        for worker_id in range(num_workers):
            if indices[worker_id] < len(worker_sublists[worker_id]):
                task_queue.append(worker_sublists[worker_id][indices[worker_id]])
                indices[worker_id] += 1
    return task_queue


"""Default functions for split-merge segmentation algorithm."""
def split_function(block):
    return np.std(block) < 5

def merging_function(mean1, mean2):
    return np.abs(mean1 - mean2) < 15

def merge_mean(block):
    return np.mean(block)