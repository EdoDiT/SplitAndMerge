import json
import time

import cv2
import numpy as np
from typing import Dict, Any

# Import utility functions
from utils.utility_functions import split_function, merging_function, merge_mean

# Import splitters
from Splitters.dict_sequential_splitter import DictSequentialSplitter
from Splitters.dict_parallel_splitter import DictParallelSplitter
from Splitters.quad_sequential_splitter import QuadNodeSequentialSplitter

# Import mergers
from Mergers.dict_graph_sequential_merger import DictGraphSequentialMerger
from Mergers.dict_graph_parallel_merger import DictGraphParallelMerger
from Mergers.nx_graph_sequential_merger import NxGraphSequentialMerger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_image(image_path: str) -> np.ndarray:
    """Load image from path and convert to numpy array."""
    import cv2
    img = cv2.imread(image_path)
    if img is not None:
        return img

def create_splitter(splitter_config: Dict[str, Any], image: np.ndarray):
    """Create splitter object based on configuration."""
    splitter_type = splitter_config["type"]
    kwargs = splitter_config.get("kwargs", {})

    # Common parameters for all splitters
    common_params = {
        "image": image,
        "split_function": split_function,
        "merge_mean": merge_mean,
        **kwargs
    }

    if splitter_type == "dict_sequential_splitter":
        return DictSequentialSplitter(**common_params)
    elif splitter_type == "dict_parallel_splitter":
        return DictParallelSplitter(**common_params)
    elif splitter_type == "quad_sequential_splitter":
        return QuadNodeSequentialSplitter(**common_params)
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")


def create_merger(merger_config: Dict[str, Any], image: np.ndarray, splitter=None):
    """Create merger object based on configuration."""
    merger_type = merger_config["type"]
    kwargs = merger_config.get("kwargs", {})

    # Common parameters for all mergers
    common_params = {
        "image": image,
        "merging_function": merging_function,
        **kwargs
    }

    # If both splitter and merger are parallel, share the manager
    if (splitter is not None and
        hasattr(splitter, 'manager') and
        merger_type == "dict_graph_parallel_merger"):
        common_params["mp_manager"] = splitter.manager

    if merger_type == "dict_graph_sequential_merger":
        return DictGraphSequentialMerger(**common_params)
    elif merger_type == "dict_graph_parallel_merger":
        return DictGraphParallelMerger(**common_params)
    elif merger_type == "nx_graph_sequential_merger":
        return NxGraphSequentialMerger(**common_params)
    else:
        raise ValueError(f"Unknown merger type: {merger_type}")


def main():
    """Main entrypoint for the split-merge segmentation algorithm."""
    # Load configuration
    config = load_config("config.json")

    # Load image
    image = load_image(config["image"])
    print(f"Loaded image with shape: {image.shape}")

    # Build splitter
    splitter = create_splitter(config["splitter"], image)
    print(f"Created splitter: {config['splitter']['type']}")

    # Build merger
    merger = create_merger(config["merger"], image, splitter)
    print(f"Created merger: {config['merger']['type']}")

    # Performance measurement
    loop_count = config["loop_count"]
    total_time = 0.0

    print(f"Running {loop_count} iterations...")

    for i in range(loop_count):
        start_time = time.time()

        # Run quadtree splitting
        quadtree, root = splitter.build_quadtree()

        # Build region adjacency graph
        merger.build_region_adjacency_graph(root, quadtree)

        # Perform merging
        merger.merge()

        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += iteration_time

        print(f"Iteration {i+1}/{loop_count} completed in {iteration_time:.4f} seconds")

    # Calculate and print average time
    average_time = total_time / loop_count
    print(f"\nAverage execution time: {average_time:.4f} seconds")
    print(f"Total time for {loop_count} iterations: {total_time:.4f} seconds")

    # Get and display images
    img1 = splitter.get_split_image()
    img2 = merger.get_merge_image()

    print(f"Split image shape: {img1.shape}")
    print(f"Merged image shape: {img2.shape}")
    cv2.imshow('Split Result', img1)
    cv2.imshow('Merged Result', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
