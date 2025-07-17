from collections import deque
from multiprocessing.managers import BaseManager
from queue import Queue
from typing import Tuple, List, Any

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

def shuffle_queue(list_of_nodes: List[List[Tuple[int, int, int]]], num_workers: int, task_queue: List):
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

class DequeManager(BaseManager):
    pass

# Register a new type "Deque" that will create a collections.deque() in the manager server.
# Expose only the methods we want remote processes to call.
DequeManager.register(
    'Deque',
    callable=deque,
    exposed=('append', 'appendleft', 'pop', 'popleft', '__len__', '__bool__')
)