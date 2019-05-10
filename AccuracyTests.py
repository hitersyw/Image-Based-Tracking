"""
!!! Do not publish this file, for test purpose only!
"""
import cv2
from src import Tracker
from src import Plotter
import matplotlib.pyplot as plt
import numpy as np
import datetime
from skimage import transform as tf
from skimage import io
import json

#reference_image = cv2.imread('./data/OP2.png')
img = cv2.imread('./data/Test1.jpg')
height, width, _ = img.shape
#comparison_image =  cv2.imread('./data/OP2.png')
#
#reference_image = img[ 0:(height-70), 0:(width-100)]
#comparison_image = img[70:height, 100:width]
comparison_image = cv2.imread('data/Test2.jpg')
#ref = img[0:height, 0:height]
ref = img
image_center = tuple(np.array(ref.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, 90, 1.0)
comp = cv2.warpAffine(ref, rot_mat, ref.shape[1::-1], flags=cv2.INTER_LINEAR)
#cv2.imwrite('ref.png', ref)
cv2.imwrite('comp.png', comp)
#test = tf.rotate(ref, 90)
#io.imsave('testtest.png', test)

segmentation = False

plotter = Plotter.Plotter()

models = []
# inlier rate
inlier_rates = []

#perform tests n=X times -> add keypoints, matches and models to a list for each
n = 10
for i in range(n):
    print('Round ', i)
    tracker = Tracker.Tracker(segmentation)
    k1, k2, m = tracker.extract_and_match(img, comparison_image)
    model, mask = tracker.compute_affine_transform(k1, k2, m)
    print(model)
    models.append(model.tolist())

    num_inliers = np.count_nonzero(mask)
    inlier_rates.append(num_inliers/len(mask))


data = {}
data['number_keypoints_reference'] = len(k1)
data['number_keypoints_comparison'] = len(k2)
data['number_matches'] = len(m)
data['match_rate'] = (len(m)/len(k1)) if (len(k1) <= len(k2)) else (len(m)/len(k2))
data['models'] = models
data['inlier_rates'] = inlier_rates


test_name = 'test1'
file_name = 'tests/' + test_name + '_accuracy.json'

with open(file_name, 'w') as outfile:
    json.dump(data, outfile)
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
