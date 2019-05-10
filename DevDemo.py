"""
!!! Do not publish this file, for test purpose only!
"""
import cv2
from src import Tracker
from src import Plotter
import matplotlib.pyplot as plt
import numpy as np
import datetime


vc = cv2.VideoCapture('data/CI522 insertion strekin.mpg')
frame_width = int(vc.get(3))
frame_height = int(vc.get(4))
fps = int(vc.get(5))
vw = cv2.VideoWriter('data/CI522TestMatches.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

_, frameReference = vc.read()

count = 0
"""plt.ion()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
plt.axis('off')"""

i  = 0
segmentation = False
orb = Tracker.Tracker(segmentation)
plotter = Plotter.Plotter()
start = datetime.datetime.now()
while vc.isOpened():
    #time1 = datetime.datetime.now()
    _, frameNew = vc.read()

    i += 1
    #frameReference = cv2.imread('data/image2_pred.png')
    #frameNew = cv2.imread('data/image15.png')

    if frameNew is None or frameReference is None:
        print('done, writing video...')
        break

    #label = orb.semantic_segmentation(frameReference)
    #print('label computed')
    k1, k2, m = orb.extract_and_match(frameReference, frameNew)


    """seg = plotter.plot_segmentation_results(frameReference, label, (0, 0, 255), (0, 0, 255))
    print('seg plot done')
    if k1 and k2 and m:
        kp = plotter.plot_keypoints(seg, k1, 4, (255, 0, 0))
        vw.write(kp)
    else:
        vw.write(seg)"""
    if k1 and k2 and m:
        time1 = datetime.datetime.now()

        _, mask = orb.compute_affine_transform(k1, k2, m)
        time2 = datetime.datetime.now()


        blue = (255, 0, 0)
        red = (0, 0, 255)
        green = (0, 225, 0)
        black = (0, 0, 0)
        radius = 5
        matches = plotter.plot_matches_one_image(frameNew, k1, k2, m, mask, black, green)
        vw.write(matches)
    else:
        time1 = datetime.datetime.now()
        vw.write(frameNew)

    print("--- Frame ", count, " to frame ", count + 1);
    print("time to process image: ", time2 - time1)

    #print(matrix)

    #frameReference = frameNew
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
       print(vw.get(7))
       break

#plt.show(block=True)
end = datetime.datetime.now()
print("Time to process %d images: "%count, end-start)
vc.release()
vw.release()
