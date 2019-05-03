"""
!!! Do not publish this file, for test purpose only!
"""
import cv2
from src import Tracker
from src import Plotter
import matplotlib.pyplot as plt
import numpy as np
import datetime

import json

reference_image = cv2.imread('./data/Tomate1.png')
comparison_image = cv2.imread('./data/Tomate2.png')

segmentation = False
# instantiate the Tracker with the two images
# instantiate the Plotter
plotter = Plotter.Plotter()

number_keypoints_reference = []
number_keypoints_comparison = []
number_matches = []
match_rates = []
models = []
# inlier rate
inlier_rates = []

#perform tests n=X times -> add keypoints, matches and models to a list for each
n = 1
for _ in range(n):
    tracker = Tracker.Tracker(segmentation)
    # extract the keypoints and match them according to their descriptors
    k1, k2, m = tracker.extract_and_match(reference_image, comparison_image)
    # compute the affine transformation model, we ignore the mask here, see the documentation for more information
    model, mask = tracker.compute_affine_transform(k1, k2, m)

    number_keypoints_reference.append(len(k1))
    number_keypoints_comparison.append(len(k2))
    number_matches.append(len(m))
    m_rate = (len(m)/len(k1)) if (len(k1) <= len(k2)) else (len(m)/len(k2))
    match_rates.append(m_rate)
    models.append(model.tolist())

    num_inliers = np.count_nonzero(mask)
    inlier_rates.append(num_inliers/len(mask))


data = {}
data['number_keypoints_reference'] = number_keypoints_reference
data['number_keypoints_comparison'] = number_keypoints_comparison
data['number_matches'] = number_matches
data['match_rates'] = match_rates
data['models'] = models
data['inlier_rates'] = inlier_rates


test_name = 'test1'
file_name = 'tests/' + test_name + '_accuracy.json'

#with open(file_name, 'w') as outfile:
#    json.dump(data, outfile)
##debugging
"""
blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 225, 0)
radius = 5
# plot the keypoints in their images
keypoints_ref_image = plotter.plot_keypoints(reference_image, keypoints_ref, radius, blue)
keypoints_comp_image = plotter.plot_keypoints(comparison_image, keypoints_comp, radius, blue)
matches = plotter.plot_matches(reference_image, keypoints_ref, comparison_image, keypoints_comp, matches, green, blue, mask, red)
seg = plotter.plot_segmentation_results(reference_image, label, blue, 2)
cv2.imwrite("KPref.png", keypoints_ref_image)
cv2.imwrite("KPcomp.png", keypoints_comp_image)
cv2.imwrite('bb.png', seg)
cv2.imwrite('matches.png', matches)
"""
