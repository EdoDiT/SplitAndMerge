# Split-Merge Image Segmentation Algorithm

A Python implementation of the split-merge segmentation algorithm for image processing, featuring multiple data structure approaches and parallel processing capabilities.

## Overview

This project implements the classic split-merge segmentation algorithm using quadtree data structures. The algorithm recursively splits an image into homogeneous regions and then merges adjacent regions based on similarity criteria. The implementation provides multiple approaches for both splitting and merging phases, including sequential and parallel versions.

## Features

- **Multiple Splitter Implementations:**
  - `QuadNodeSequentialSplitter`: Uses custom QuadNode objects to represent quadrants in the quadtree
  - `DictSequentialSplitter`: Dictionary-based implementation with nodes as keys and children as values
  - `DictParallelSplitter`: Parallel version of the dictionary-based splitter

- **Multiple Merger Implementations:**
  - `NxGraphSequentialMerger`: Uses NetworkX graph for region adjacency representation
  - `DictGraphSequentialMerger`: Dictionary-based implementation for edges with node lists
  - `DictGraphParallelMerger`: Parallel version of the dictionary-based merger

- **Configurable Parameters:** JSON-based configuration system
- **Performance Benchmarking:** Built-in timing and iteration support
- **Customizable Functions:** User-definable split, merge, and mean calculation functions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `opencv-python==4.11.0.86` - Image processing operations
- `numpy==2.2.4` - Numerical computations
- `networkx==3.4.2` - Graph data structures (for NxGraphSequentialMerger)

## Configuration

The project uses a JSON configuration file (`config.json`) to specify algorithm parameters:

```json
{
  "splitter": {
    "type": "dict_parallel_splitter",
    "kwargs": {
      "min_block_size": 10
    }
  },
  "merger": {
    "type": "dict_graph_parallel_merger",
    "kwargs": {}
  },
  "image": "imgs_lab/sample_image.png",
  "loop_count": 5
}
```

### Configuration Parameters

- **splitter.type**: Choose from `"dict_sequential_splitter"`, `"dict_parallel_splitter"`, or `"quad_sequential_splitter"`
- **merger.type**: Choose from `"dict_graph_sequential_merger"`, `"dict_graph_parallel_merger"`, or `"nx_graph_sequential_merger"`
- **splitter.kwargs**: Additional parameters for the splitter (e.g., `min_block_size`)
- **merger.kwargs**: Additional parameters for the merger
- **image**: Path to the input image file
- **loop_count**: Number of iterations for performance benchmarking

## Usage

Run the main script with your configured parameters:

```bash
python main.py
```

The script will:
1. Load the configuration from `config.json`
2. Load the specified image
3. Create the selected splitter and merger instances
4. Execute the split-merge algorithm for the specified number of iterations
5. Display performance statistics
6. Show the split and merged result images

## Customizing Algorithm Functions

The project provides three default functions in `utils/utility_functions.py` that you can customize:

### 1. Split Function
```python
def split_function(block):
    return np.std(block) < 5
```
Determines whether a block should be split further. Return `True` to continue splitting, `False` to stop.

### 2. Merging Function
```python
def merging_function(mean1, mean2):
    return np.abs(mean1 - mean2) < 15
```
Determines whether two adjacent regions should be merged. Return `True` to merge regions, `False` to keep them separate.

### 3. Merge Mean Function
```python
def merge_mean(block):
    return np.mean(block)
```
Calculates the representative value for a block/region.

To customize these functions:
1. Edit the functions in `utils/utility_functions.py`
2. Or create new functions and update the imports in `main.py`

## Project Structure

```
├── main.py                    # Main entry point
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── imgs_lab/                 # Sample images
│   └── sample_image.png
├── Splitters/               # Splitter implementations
│   ├── dict_sequential_splitter.py
│   ├── dict_parallel_splitter.py
│   ├── quad_sequential_splitter.py
│   └── splitter.py          # Base splitter class
├── Mergers/                 # Merger implementations
│   ├── dict_graph_sequential_merger.py
│   ├── dict_graph_parallel_merger.py
│   ├── nx_graph_sequential_merger.py
│   └── merger.py            # Base merger class
└── utils/                   # Utility functions and data structures
    ├── quadnode.py          # QuadNode class definition
    └── utility_functions.py # Helper functions and customizable algorithms
```

## Algorithm Overview

### Split Phase
1. Start with the entire image as the root region
2. Calculate homogeneity criterion using the `split_function`
3. If region is not homogeneous, split it into four quadrants
4. Recursively apply splitting until all regions meet the homogeneity criterion or minimum size is reached

### Merge Phase
1. Build a region adjacency graph from the quadtree leaves
2. For each pair of adjacent regions, apply the `merging_function`
3. Merge regions that satisfy the merging criterion
4. Update the final segmented image

## Performance Considerations

- **Parallel Implementations**: The parallel splitter and merger can significantly improve performance on multi-core systems
- **Memory Sharing**: When using both parallel splitter and merger, they share the same multiprocessing manager for efficiency
- **Load Balancing**: The parallel merger includes a shuffle mechanism to distribute work evenly among workers

## Example Output

The program outputs performance statistics:
```
Loaded image with shape: (height, width, channels)
Created splitter: dict_parallel_splitter
Created merger: dict_graph_parallel_merger
Running 5 iterations...
Iteration 1/5 completed in 6.1234 seconds
...
Average execution time: 6.1200 seconds
Total time for 5 iterations: 32.6000 seconds
```
Parallel mergers outputs queue status, due to length of the process, to avoid being misinterpreted as a hang.
And displays both the split result (quadtree visualization) and the final merged segmentation.
