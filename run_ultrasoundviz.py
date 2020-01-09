#!/usr/bin/env python3
"""Example display of ultrasound image tracking.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_ultrasoundviz.py

Todo:
    implement all the things!
"""

import time
import os

import cv2
import numpy as np

from multisensorimport.tracking import us_tracking_utils as track

READ_PATH = '/Users/akashvelu/Documents/Research_HART2/tracking_data/ultrasound_t5w1/'


def main():
    """Execute ultrasound image tracking visualization."""
    window_size = 17
    # set Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(19, 19),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7)

    # set some optional manually-selected points
    curr_pts = [
        np.array([[63, 2]], dtype=np.float32),
        np.array([[58, 20]], dtype=np.float32),
        np.array([[52, 45]], dtype=np.float32),
        np.array([[49, 69]], dtype=np.float32),
        np.array([[48, 94]], dtype=np.float32),
        np.array([[49, 126]], dtype=np.float32),
        np.array([[56, 157]], dtype=np.float32),
        np.array([[69, 184]], dtype=np.float32),
        np.array([[79, 207]], dtype=np.float32),
        np.array([[91, 224]], dtype=np.float32),
        np.array([[115, 208]], dtype=np.float32),
        np.array([[131, 189]], dtype=np.float32),
        np.array([[138, 174]], dtype=np.float32),
        np.array([[151, 152]], dtype=np.float32),
        np.array([[152, 131]], dtype=np.float32),
        np.array([[146, 106]], dtype=np.float32),
        np.array([[133, 75]], dtype=np.float32),
        np.array([[120, 54]], dtype=np.float32),
        np.array([[116, 29]], dtype=np.float32),
        np.array([[111, 16]], dtype=np.float32),
        np.array([[115, 4]], dtype=np.float32)
    ]
    pts_manual = np.array(curr_pts)

    # try some good points
    init_path = READ_PATH + '1495.pgm'
    init_img = cv2.imread(init_path, -1)
    pts_good = cv2.goodFeaturesToTrack(init_img, mask=None, **feature_params)


    # extract initial contour from keyframe
    keyframe_path = READ_PATH + '0.png'
    pts = track.extract_contour_pts(keyframe_path)

    # filter to be used (1: median filter, 2: bilateral filter, 3: course bilateral, 4: anisotropicDiffuse anything else no filter )
    fineFilterNum = 2
    courseFilterNum = 3

    # remove points that have low corner scores (Shi Tomasi Corner scoring)
    fineFilteredPoints = track.filter_points(window_size, pts, 0.0015, fineFilterNum, init_img, .5)
    courseFilteredPoints = track.filter_points(window_size, pts, 0.0015, courseFilterNum, init_img, .45)


    # find points which differ
    coursePointsIndeces = set()
    for i in range(len(courseFilteredPoints)):
        coursePoint = courseFilteredPoints[i]
        add = True
        for j in range(len(fineFilteredPoints)):
            finePoint = fineFilteredPoints[j]
            if np.linalg.norm(finePoint - coursePoint) < 0.0001:
                add = False
        if add:
            coursePointsIndeces.add(i)


    coursePoints = []
    for index in coursePointsIndeces:
        coursePoints.append(courseFilteredPoints[index])

    courseFilteredPoints = np.array(coursePoints)

    # track points
    contour_areas = track.track_pts(READ_PATH, fineFilteredPoints, courseFilteredPoints, lk_params, True, fineFilterType = fineFilterNum, courseFilterType = courseFilterNum)

    # write contour areas to csv file
    out_path = READ_PATH + 'csa.csv'
    with open(out_path, 'w') as outfile:
        for ctr in contour_areas:
            outfile.write(str(ctr))
            outfile.write('\n')

def matchPoints(contourPoints, goodPoints):
    epsilon = 10
    finalIndeces = set()
    finalPoints = []
    print("LENGTH", len(contourPoints))

    listContourPoints = contourPoints.tolist()

    for i in range(len(listContourPoints)):
        cPoint = listContourPoints[i]
        for gPoint in goodPoints.tolist():
            diff = np.linalg.norm(np.array(gPoint) - np.array(cPoint))
            if diff <= epsilon:
                if i not in finalIndeces:
                    finalIndeces.add(i)
                    finalPoints.append(cPoint)

    return np.float32(np.array(finalPoints))



if __name__ == "__main__":
    main()
