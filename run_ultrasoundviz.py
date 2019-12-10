#!/usr/bin/env python3
"""Example display of ultrasound image tracking.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_ultrasoundviz.py

Todo:
    implement all the things!
"""

import time

import cv2
import numpy as np

#from multisensorimport.dataobj import trialdata as td

READ_PATH = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/ultrasound_t5w1/'


def main():
    """Execute ultrasound image tracking visualization."""
    # set frame numbers of desired data
    ind_low = 1495
    ind_high = 2282

    # set Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(25, 25),
                     maxLevel=8,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # extract initial contour from keyframe
    keyframe_path = READ_PATH + '0.png'
    img = cv2.imread(keyframe_path, -1)
    alpha_channel = img[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
    color = img[:, :, :3]
    new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    new_img = (255-new_img)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # convert contour to array
    currPoints = []
    for i in range(len(contours[0])):
        currPoints.append(np.array(contours[0][i], dtype=np.float32))
    npCurrPoints = np.array(currPoints)
    print(len(currPoints))

    # create named window
    cv2.namedWindow('Frame')

    # read initial frame
    filename = READ_PATH + str(ind_low) + '.pgm'
    old_frame = cv2.imread(filename, -1)

    # track and display specified points through images
    for i in range(ind_low + 1, ind_high + 1):

        # read in new frame
        filename = READ_PATH + str(i) + '.pgm'
        frame = cv2.imread(filename, -1)

        # calculate new point locations
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_frame, frame, npCurrPoints, None, **lk_params)

        # save old frame for optical flow calculation
        old_frame = frame.copy()

        # reset point locations
        npCurrPoints = new_points
        for j in range(len(npCurrPoints)):
            x, y = npCurrPoints[j].ravel()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # display to frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == 27:  # stop on escape key
            break
        time.sleep(0.05)

        # print contour area
        print(cv2.contourArea(npCurrPoints))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
