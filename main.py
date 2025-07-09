import cv2
import numpy as np

from Sequential.split_and_merge import SequentialSplitAndMerge

# Load the image
image = cv2.imread('imgs_lab/sample_image.png')
if image is None:
    raise ValueError("Image not found or unable to load.")
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
split_and_merger = SequentialSplitAndMerge(
    image=image,
    split_function=lambda block: np.std(block) < 5,  # Example threshold for homogeneity
    merging_function=lambda mean1, mean2: np.abs(mean1 - mean2) < 15,  # Example threshold for merging
    merge_mean=lambda block: np.mean(block),  # Mean function for merging
    min_block_size=10  # Minimum size of blocks to consider
)
img_1, img_2 = split_and_merger.process_image()
cv2.imshow('Split Result', img_1)
cv2.imshow('Merged Result', img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
