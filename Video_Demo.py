"""
This is a small demo using the Tracker and Plotter for video stream input.
We load one frame at a time.
For demonstration purpose we do not use the combined tracking function Tracker.track() but rather all steps individual.
"""
#TODO: Kommentare zu einzelnen Zeilen
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

from src import Tracker
from src import Plotter

# prepare the video input stream
vc = cv2.VideoCapture('./data/Surgery.avi')
# read the first frame of the video stream
_, frameReference = vc.read()

count = 0
plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
plt.axis('off')

segmentation = False
# instantiate the Tracker with the segmentation option
tracker = Tracker.Tracker(segmentation)
plotter = Plotter.Plotter()

start = datetime.datetime.now()
while vc.isOpened():
    time1 = datetime.datetime.now()
    # grab the next frame
    _, frameNew = vc.read()

    if frameNew is None:
        end = datetime.datetime.now()
        print("finishing...")
        break

    assert frameReference.shape == frameNew.shape

    # extract the keypoints and match them according to their descriptors
    keypoints1, keypoints2, matches = tracker.extract_and_match(frameReference, frameNew)
    # compute the affine transformation model and get the inlier mask
    model, mask = tracker.compute_affine_transform(keypoints1, keypoints2, matches)

    red = (0, 0, 255)
    blue = (255, 0, 0)
    green = (0, 255, 0)

    # plot the matches in their images. All inliers will be drawn in green, the outliers in red
    plot = plotter.plot_matches(frameReference, keypoints1, frameNew, keypoints2, matches, green, blue, mask, red)
    plt.imshow(plot[:,:,::-1])
    plt.draw()
    plt.pause(0.001)

    time2 = datetime.datetime.now()
    print("--- Frame ", count, " to frame ", count + 1);
    print("time to process image: ", time2 - time1)
    print(model)

    frameReference = frameNew
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

plt.show(block=True)
print("Time to process %d images: "%count, end-start)
vc.release()
