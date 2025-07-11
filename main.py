import time

import cv2
import numpy as np

from Parallel.future_based import FutureSplitAndMerge
from Parallel.ipc_queue_version import PosixQueueSplitAndMerge
from Sequential.split_and_merge import SequentialSplitAndMerge
from Parallel.ipc_shared_memory_based import SharedMemoryParallelSplitAndMerge

def split_function(block):
    return np.std(block) < 5

def merging_function(mean1, mean2):
    return np.abs(mean1 - mean2) < 15

def merge_mean(block):
    return np.mean(block)

loop_count = 1
# Load the image
image = cv2.imread('imgs_lab/sample_image.png')
if image is None:
    raise ValueError("Image not found or unable to load.")
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
start_time = time.time()
for i in range(loop_count):
    split_and_merger = SharedMemoryParallelSplitAndMerge(
        image=image,
        split_function=split_function,  # Example threshold for homogeneity
        merging_function=merging_function,  # Example threshold for merging
        merge_mean=merge_mean,  # Mean function for merging
        min_block_size=10  # Minimum size of blocks to consider
    )
    img_1, img_2 = split_and_merger.process_image()
end_time = time.time()
print(f"Processing time: {(end_time - start_time)/loop_count:.4f} seconds")
cv2.imshow('Split Result', img_1)
cv2.imshow('Merged Result', img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
