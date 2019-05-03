"""
This is a small demo using the Tracker and Plotter for static image input.
We load one image, and as comparison image we use a shifted version.
For demonstration purpose we do not use the combined tracking function Tracker.track() but rather all steps individual.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

from src import Tracker
from src import Plotter


# load the sample image
reference_image = cv2.imread('./data/OP2.png')
height, width, _ =reference_image.shape
#comparison_image =  cv2.imread('./data/OP2.png')
M = np.float32([[1, 0, 40], [0, 1, 50]])
comparison_image = cv2.warpAffine(reference_image, M, (width, height))

img = reference_image[:960, :]
segmentation = False
# instantiate the Tracker with the two images
tracker = Tracker.Tracker(segmentation)
# instantiate the Plotter
plotter = Plotter.Plotter()

keypoints_ref, keypoints_comp, matches = tracker.extract_and_match(reference_image, comparison_image)
model, mask = tracker.compute_affine_transform(keypoints_ref, keypoints_comp, matches)
print(model.params)
#label = tracker.semantic_segmentation(img)
#cv2.imwrite('label.png', label)

"""
blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
radius = 10
# plot the keypoints in their images
keypoints_ref_image = plotter.plot_keypoints(reference_image, keypoints_ref, radius, blue)
keypoints_comp_image = plotter.plot_keypoints(comparison_image, keypoints_comp, radius, red)
seg = plotter.plot_segmentation_results(frameReference, label, (0, 0, 255), (0, 0, 255))
"""
