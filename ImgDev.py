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
from math import asin,acos, degrees, atan2


# load the sample image
img = cv2.imread('./data/OP2.png')
height, width, _ = img.shape
#reference_image = cv2.imread('./data/OP2.png')
#comparison_image =  cv2.imread('./data/OP2 Kopie.png')

#reference_image = img[:960, :]
#cv2.imwrite('refCropped.png', reference_image)
#reference_image = img[ 0:(height-70), 0:(width-100)]
comparison_image = cv2.imread('data/rotated.png')
#comparison_image = img[70:height, 100:width]
#cv2.imwrite('ref.png', reference_image)
#cv2.imwrite('comp.png', comparison_image)

segmentation = False
# instantiate the Tracker with the two images
tracker = Tracker.Tracker(segmentation)
# instantiate the Plotter
plotter = Plotter.Plotter()
#label = tracker.semantic_segmentation(reference_image)
keypoints_ref, keypoints_comp, matches = tracker.extract_and_match(img, comparison_image)
model, mask = tracker.compute_affine_transform(keypoints_ref, keypoints_comp, matches)
print(model)
A11 =  model[0][0]
A21 = model[1][0]

if A11 > 1:
    A11 = 1
if A11 < -1:
    A11 = -1

if A21 > 1:
    A21 = 1
if A21 < -1:
    A21 = -1


angle = atan2(A21,A11)
blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
black = (0, 0, 0)
#plot = plotter.plot_matches_one_image(comparison_image, keypoints_ref, keypoints_comp, matches, mask, black, green)

#cv2.imwrite('matcheswhat.png', plot)


#label = tracker.semantic_segmentation(img)
#cv2.imwrite('label.png', label)
# plot the keypoints in their images
#keypoints_ref_image = plotter.plot_keypoints(reference_image, keypoints_ref, radius, blue)
#keypoints_comp_image = plotter.plot_keypoints(comparison_image, keypoints_comp, radius, blue)

#seg = plotter.plot_segmentation_results(frameReference, label, (0, 0, 255), (0, 0, 255))
#cv2.imwrite('kpRef.png', keypoints_ref_image)
#cv2.imwrite('kpComp.png', keypoints_comp_image)

blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 225, 0)
radius = 5
#matches = plotter.plot_matches(reference_image, keypoints_ref, comparison_image, keypoints_comp, matches, green, blue, mask, red)
#cv2.imwrite('matcheswhat.png', matches)
