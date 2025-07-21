from collections import deque
from logging import warning
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager
from typing import Tuple, Any, Callable, List

import numpy as np

from Splitters.splitter import Splitter
import multiprocessing as mp


class DictParallelSplitter(Splitter):
    def __init__(
            self,
            image,
            split_function,
            merge_mean,
            min_block_size=10,
            num_workers: int = None,
            mp_manager: None = None,
            **kwargs
    ):
        """ Initializes the DictParallelSplitter with the given parameters, and sets up shared memory  and manager
        for parallel processing.
        Args:
            image: The input image to be split.
            split_function: Function to determine if a block is homogeneous.
            merge_mean: Function to compute the mean of a block.
            min_block_size: Minimum size of a block to be split.
            num_workers: Number of worker processes to use for parallel processing.
            mp_manager: Optional multiprocessing manager for shared resources.
        """
        super().__init__(image, split_function, merge_mean, min_block_size, **kwargs)
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.shared_image = None
        self.image_shape = image.shape
        self.image_dtype = image.dtype
        print("Starting Parallel setup")
        # Create shared memory for the image
        self._setup_shared_memory()

        # Multiprocessing synchronization primitives
        if not mp_manager:
            SyncManager.register(
                'Deque',
                callable=deque,
                exposed=('append', 'appendleft', 'pop', 'popleft', '__len__', '__bool__')
            )
        self.manager = mp_manager if mp_manager is not None else mp.Manager()
        self.shared_task_stack = self.manager.Deque()
        self.shared_split_result = self.manager.list()
        self.shared_quadtree = self.manager.dict()
        self.task_lock = mp.Lock()
        self.result_lock = mp.Lock()
        print("Parallel setup complete")

    def _setup_shared_memory(self):
        """Sets up shared memory for the image data."""
        # Create shared memory buffer for the image
        image_bytes = self._image.nbytes
        self.shared_image = shared_memory.SharedMemory(create=True, size=image_bytes)

        # Create numpy array from shared memory
        shared_array = np.ndarray(self.image_shape, dtype=self.image_dtype, buffer=self.shared_image.buf)
        shared_array[:] = self._image[:]

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
            warning("Error during cleanup of shared memory: {}".format(e))
            pass

    def build_quadtree(self) -> Tuple[dict, Tuple[int, int, int, int, Any, int]]:
        """Builds a quadtree from the image by recursively splitting it into homogeneous blocks using parallel
        processing. The quadtree is built by checking if a block is homogeneous using the split function.
        If a block is not homogeneous, it is split into four children blocks, and the children added to the
        stack queue.
        The process continues until all blocks are homogeneous or the minimum block size is reached.
        """
        # Initialize the root node and shared task stack
        size_x = self._image.shape[1]
        size_y = self._image.shape[0]
        self.root = (0, 0, size_x, size_y, self._merge_mean(self._image), size_x*size_y)  # (x, y, size, mean, area)
        self.shared_task_stack.append(self.root)

        # Clear previous results
        self.shared_split_result[:] = []
        self.shared_quadtree.clear()

        # Create and start worker processes
        processes = []
        print("Starting worker processes...")
        for _ in range(self.num_workers):
            p = mp.Process(
                target=self._worker_process,
                args=(
                    self.shared_image.name,
                    self.image_shape,
                    self.image_dtype,
                    self.shared_task_stack,
                    self.shared_split_result,
                    self.shared_quadtree,
                    self.task_lock,
                    self.result_lock,
                    self._split_function,
                    self._min_block_size,
                    self._merge_mean
                )
            )
            p.start()
            processes.append(p)
        print(f"Started {len(processes)} worker processes.")
        # Wait for all workers to complete
        for p in processes:
            p.join()

        # Copy results back to instance variables
        self.quadtree = dict(self.shared_quadtree)
        self.split_result = list(self.shared_split_result)
        print("Finished building quadtree with {} blocks.".format(len(self.split_result)))
        return self.quadtree, self.root

    @staticmethod
    def _worker_process(shared_image_name, image_shape, image_dtype, shared_task_stack,
                        shared_split_result, shared_quadtree, task_lock, result_lock,
                        split_function, min_block_size, merge_mean):
        """Worker process function for parallel quadtree building. Until the shared task stack is empty,
        it pops a node from the stack, checks if it is homogeneous using the split function, and if not,
        splits it into four children nodes. The children nodes are then added back to the shared task stack.
        Args:
            shared_image_name: Name of the shared memory block containing the image
            image_shape: Shape of the image
            image_dtype: Data type of the image
            shared_task_stack: Shared task stack (managed list)
            shared_split_result: Shared split result list (managed list)
            shared_quadtree: Shared quadtree dictionary (managed dict)
            task_lock: Lock for task stack access
            result_lock: Lock for result list access
            split_function: Function to determine if a block is homogeneous
            min_block_size: Minimum block size for splitting
        """
        # Connect to shared memory
        try:
            existing_shm = shared_memory.SharedMemory(name=shared_image_name)
            image = np.ndarray(image_shape, dtype=image_dtype, buffer=existing_shm.buf)
        except FileNotFoundError:
            return  # Shared memory not available, exit worker

        while True:
            # Get task from shared stack with lock
            with task_lock:
                if len(shared_task_stack) == 0:
                    break
                node = shared_task_stack.pop()

            # Process the node
            if node[2] > min_block_size and not DictParallelSplitter._is_homogeneous(node, image, split_function):
                # Split the node
                children = DictParallelSplitter._split(node, image, merge_mean)
                # Store in shared quadtree (no lock needed as each worker uses different keys)
                shared_quadtree[node] = children

                # Add children to task stack with lock
                with task_lock:
                    for child in children:
                        shared_task_stack.append(child)
            else:
                # Add to split result with lock
                with result_lock:
                    shared_split_result.append(node)

        # Clean up shared memory reference
        existing_shm.close()

    @staticmethod
    def _is_homogeneous(node: Tuple[int, int, int, int, Any, int], image: np.ndarray, split_function: Callable):
        """
        Checks if the block defined by the node is homogeneous using the split function.
        :param node: A tuple containing (x, y, size_x, size_y, mean, area).
        :return: True if the block is homogeneous, False otherwise.
        """
        x, y, size_x, size_y, mean, area = node
        block = image[y:y + size_y, x:x + size_x]
        return split_function(block)

    @staticmethod
    def _split(
            node: Tuple[int, int, int, int, Any, int],
            image: np.ndarray,
            merge_mean: Callable
    ) -> List[
        Tuple[int, int, int, int, Any, int]
    ]:
        """
        Splits the block defined by the node into four quadrants.
        :param node: A tuple containing (x, y, size_x, size_y, mean, area).
        :return: A list of tuples representing the four child nodes.
        """
        size_x = node[2] / 2
        size_y = node[3] / 2
        area = int(size_x * size_y)
        node_top_left = (node[0], node[1], int(size_x), int(size_y), merge_mean(image[node[1]:int(node[1] + size_y), node[0]:int(node[0] + size_x)]), area)
        node_top_right = (int(node[0] + size_x), node[1], int(size_x), int(size_y), merge_mean(image[node[1]:int(node[1] + size_y), int(node[0] + size_x):node[0] + node[2]]), area)
        node_bottom_right = (int(node[0] + size_x), int(node[1] + size_y), int(size_x), int(size_y), merge_mean(image[int(node[1] + size_y):node[1] + node[3], int(node[0] + size_x):node[0] + node[2]]), area)
        node_bottom_left = (node[0], int(node[1] + size_y), int(size_x), int(size_y), merge_mean(image[int(node[1] + size_y):node[1] + node[3], node[0]:node[0] + int(size_x)]), area)
        return [node_top_left, node_top_right, node_bottom_right, node_bottom_left]

    def get_split_image(self):
        """Returns the split image"""
        for node in self.split_result:
            x = node[0]
            y = node[1]
            size = node[2]
            block = self._image[y:y + size, x:x + size]
            avg_color = block.mean(axis=(0, 1), keepdims=True)
            self.split_image[y:y + size, x:x + size] = avg_color
        return self.split_image