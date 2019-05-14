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

start = datetime.datetime.now()

# load the sample image
reference_image = cv2.imread('./data/SurgeryExample.png')
height, width, _ = reference_image.shape
print(reference_image.shape)

# shifts the image 400 pixels to the right and 500 pixels down
M = np.float32([[1, 0, 40], [0, 1, 50]])
comparison_image = cv2.warpAffine(reference_image, M, (width, height))

segmentation = False
# instantiate the Tracker with the segmentation option
tracker = Tracker.Tracker(segmentation)
# instantiate the Plotter
plotter = Plotter.Plotter()

# extract the keypoints and match them according to their descriptors
keypoints_ref, keypoints_comp, matches = tracker.extract_and_match(reference_image, comparison_image)
# compute the affine transformation model, we ignore the mask here, see the documentation for more information
model, _ = tracker.compute_affine_transform(keypoints_ref, keypoints_comp, matches)

end = datetime.datetime.now()
print("The transformation was computed as ")
print(model)
print("Computation time ", end - start)

print("Plotting keypoints...")
blue = (255, 0, 0)
red = (0, 0, 255)
radius = 10
# plot the keypoints in their images
keypoints_ref_image = plotter.plot_keypoints(reference_image, keypoints_ref, radius, blue)
keypoints_comp_image = plotter.plot_keypoints(comparison_image, keypoints_comp, radius, red)

figure = plt.figure(figsize=(12, 8))
plt.title("Keypoints")
plt.axis('off')
figure.add_subplot(121)
plt.imshow(keypoints_ref_image[:,:,::-1])
figure.add_subplot(122)
plt.imshow(keypoints_comp_image[:,:,::-1])
plt.show()

figure2 = plt.figure(figsize=(12, 8))
plt.title("Matches")
plt.axis('off')
print("Plotting matches...")
# plot the matches in their images
matches_image = plotter.plot_matches(reference_image, keypoints_ref, comparison_image, keypoints_comp, matches, blue, red)
plt.imshow(matches_image[:,:,::-1])
plt.show()
