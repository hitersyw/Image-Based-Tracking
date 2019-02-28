"""
!!! Do not publish this file, for test purpose only!
"""
import cv2
from src import Tracker
from src import Plotter
import matplotlib.pyplot as plt
import numpy as np
import datetime


vc = cv2.VideoCapture('data/Surgery.avi')
frame_width = int(vc.get(3))
frame_height = int(vc.get(4))
fps = int(vc.get(5))
vw = cv2.VideoWriter('data/SurgeryKeypoints.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

_, frameReference = vc.read()

count = 0
"""plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
plt.axis('off')"""

i  = 0
start = datetime.datetime.now()
while vc.isOpened():
    time1 = datetime.datetime.now()
    _, frameNew = vc.read()

    i += 1
    #frameReference = cv2.imread('data/image2_pred.png')
    #frameNew = cv2.imread('data/image15.png')

    if frameNew is None or frameReference is None:
        print('done, writing video...')
        break

    """orb = Tracker(frameReference, frameNew)
    #preRef, preNew = orb.preprocess()
    orb.preprocess()
    k1, k2, m = orb.extract_and_match()"""

    """plotter = Plotter.ORB_Plotter()

    color1 = (0, 0, 255)
    color = (255, 0, 0)
    plot = plotter.plot_segmentation_results(frameNew, frameReference, color, color1)
    cv2.imwrite('plot.png', plot)"""

    orb = Tracker.Tracker(frameNew, frameNew)
    plotter = Plotter.ORB_Plotter()
    label = orb.semantic_segmentation(frameNew)
    k1, _, _ = orb.extract_and_match()
    seg = plotter.plot_segmentation_results(frameNew, label, (0, 0, 255), (0, 0, 255))
    kp = plotter.plot_keypoints(seg, k1, 4, (255, 0, 0))
    vw.write(kp)

    time2 = datetime.datetime.now()
    print("--- Frame ", count, " to frame ", count + 1);
    print("time to process image: ", time2 - time1)
    #print(matrix)

    frameReference = frameNew
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
       print(vw.get(7))
       break

#plt.show(block=True)
end = datetime.datetime.now()
print("Time to process %d images: "%count, end-start)
vc.release()
vw.release()
