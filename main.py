import time

import cv2
import numpy as np

from Parallel.parallel_splitter import ParallelSplit
from Parallel.parallel_merger import ParallelSplitAndMerge
from Sequential.split_and_merge import SequentialSplitAndMerge

def split_function(block):
    return np.std(block) < 5

def merging_function(mean1, mean2):
    return np.abs(mean1 - mean2) < 15

def merge_mean(block):
    return np.mean(block)


# Load the image
loop_number = 1
image = cv2.imread('imgs_lab/sample_image.png')
if image is None:
    raise ValueError("Image not found or unable to load.")
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
start_time = time.time()
for i in range(loop_number):
    split_and_merger = ParallelSplitAndMerge(
        image=image,
        split_function=split_function,  # Example threshold for homogeneity
        merging_function=merging_function,  # Example threshold for merging
        merge_mean=merge_mean,  # Mean function for merging
        min_block_size=10,  # Minimum size of blocks to consider
        # num_workers=2  # Number of parallel workers
    )
    img_1, img_2 = split_and_merger.process_image()
end_time = time.time()
print(f"Average time per loop: {(end_time - start_time) / loop_number:.4f} seconds")
cv2.imshow('Split Result', img_1)
cv2.imshow('Merged Result', img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
