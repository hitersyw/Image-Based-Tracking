"""
!!! Do not publish this file, for test purpose only!
"""
import cv2
from src import Tracker
from src import Plotter
import matplotlib.pyplot as plt
import numpy as np
import datetime


vc = cv2.VideoCapture('data/Geschnittene_Videos/Instrumente/Paprika_innen_Instrumente.mp4')
frame_width = int(vc.get(3))
frame_height = int(vc.get(4))
fps = int(vc.get(5))
vw = cv2.VideoWriter('data/Geschnittene_Videos/Instrumente/Keypoints/Paprika_innen_Instrumente_Keypoints.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

_, frameReference = vc.read()

count = 0
"""plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
plt.axis('off')"""

i  = 0
segmentation = True
orb = Tracker.Tracker(segmentation)
plotter = Plotter.Plotter()
start = datetime.datetime.now()
while vc.isOpened():
    time1 = datetime.datetime.now()
    print('frame read')
    _, frameNew = vc.read()

    i += 1
    #frameReference = cv2.imread('data/image2_pred.png')
    #frameNew = cv2.imread('data/image15.png')

    if frameNew is None or frameReference is None:
        print('done, writing video...')
        break

    label = orb.semantic_segmentation(frameReference)
    print('label computed')
    k1, k2, m = orb.extract_and_match(frameReference, frameNew)
    print('keypoints found')
    time2 = datetime.datetime.now()


    seg = plotter.plot_segmentation_results(frameReference, label, (0, 0, 255), (0, 0, 255))
    print('seg plot done')
    if k1 and k2 and m:
        kp = plotter.plot_keypoints(seg, k1, 4, (255, 0, 0))
        vw.write(kp)
    else:
        vw.write(seg)
    """if k1 and k2 and m:
        _, mask = orb.compute_affine_transform(k1, k2, m)
        inlier = (0, 255, 0) #gruen
        outlier = (0, 0, 255) #red
        keypointsColor = (255, 0, 0) #blue
        matches = plotter.plot_matches(frameReference, k1, frameNew, k2, m, inlier, keypointsColor, mask, color_outliers=outlier)
        vw.write(matches)
    else:
        vw.write(frameReference)
    """
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
