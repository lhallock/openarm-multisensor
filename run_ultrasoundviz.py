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

from multisensorimport.tracking import supporters_simple as supporters_simple
from multisensorimport.tracking import us_tracking_utils as track

# READ_PATH = '/Users/akashvelu/Documents/Research_HART2/tracking_data/ultrasound_t5w1_expanded/'
READ_PATH = '/Users/akashvelu/Documents/Research_HART2/tracking_data/sub1/t5w1/ultrasound_t5w1/'
SEG_PATH = '/Users/akashvelu/Documents/Research_HART2/tracking_data/sub1/t5w1/segmented_t5w1/'
OUT_PATH = '/Users/akashvelu/Documents/Research_HART2/tracking_data/sub1/t5w1/data_t5w1/'



def main():
    """Execute ultrasound image tracking visualization."""
    window_size = 35
    # set Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(window_size, window_size),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 300,
                          qualityLevel = 0.2,
                          minDistance = 2,
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
    # init_path = READ_PATH + '256.122143234.pgm'
    init_path = READ_PATH + '1495.pgm'
    init_img = cv2.imread(init_path, -1)
    filtered_init_img = track.course_bilateral_filter(init_img)

    # filter to be used (1: median filter, 2: bilateral filter, 3: course bilateral, 4: anisotropicDiffuse anything else no filter )
    fineFilterNum = 2
    courseFilterNum = 3

    # extract initial contour from keyframe
    keyframe_path = SEG_PATH + '1495.pgm'


    course_filtered_points, course_pts_inds, fine_filtered_points, fine_pts_inds, supporters_tracking, _ = track.initialize_points(READ_PATH, keyframe_path, init_img, feature_params, lk_params, 2)
    # filter supporter points
    #indeces_of_supporters_to_keep = track.filter_supporters(supporters_tracking, READ_PATH, lk_params)
    #supporters_tracking = supporters_tracking[indeces_of_supporters_to_keep]

    print(course_filtered_points)

    supporter_params = []
    for i in range(len(course_filtered_points)):
        point = course_filtered_points[i][0]
        _, params = supporters_simple.initialize_supporters(supporters_tracking, point, 10)
        supporter_params.append(params)
    # track points

    tracking_contour_areas, ground_truth_contour_areas, ground_truth_thickness, ground_truth_thickness_ratio, predicted_thickness, predicted_thickness_ratio, iou_error = track.track_pts(SEG_PATH, READ_PATH, fine_filtered_points, fine_pts_inds, course_filtered_points, course_pts_inds, supporters_tracking, supporter_params, lk_params, True, feature_params, True, fine_filter_type=fineFilterNum, course_filter_type=courseFilterNum)

    thickness_error = np.linalg.norm(np.array([ground_truth_thickness]) - np.array([predicted_thickness])) / len(predicted_thickness)
    thickness_ratio_error = np.linalg.norm(np.array([ground_truth_thickness_ratio]) - np.array([predicted_thickness_ratio])) / len(predicted_thickness_ratio)

    print("THICKNESS ERROR: ", thickness_error)
    print("THICKNESS RATIO ERROR: ", thickness_ratio_error)
    # write contour areas to csv file

    # change to True if ground truth needs to be written
    write_ground_truth = False

    if write_ground_truth:
        out_path_ground_truth = OUT_PATH + 'ground_truth_csa.csv'
        with open(out_path_ground_truth, 'w') as outfile:
            for ctr in ground_truth_contour_areas:
                outfile.write(str(ctr))
                outfile.write('\n')

    out_path_tracking = OUT_PATH + 'tracking_csa.csv'
    with open(out_path_tracking, 'w') as outfile:
        for ctr in tracking_contour_areas:
            outfile.write(str(ctr))
            outfile.write('\n')

    if write_ground_truth:
        out_path_thickness_ground_truth = OUT_PATH + 'ground_truth_thickness.csv'
        with open(out_path_thickness_ground_truth, 'w') as outfile:
            for thickness in ground_truth_thickness:
                outfile.write(str(thickness))
                outfile.write('\n')

    out_path_thickness_ratio_ground_truth = OUT_PATH + 'ground_truth_thickness_ratio.csv'
    with open(out_path_thickness_ratio_ground_truth, 'w') as outfile:
        for thickness_ratio in ground_truth_thickness_ratio:
            outfile.write(str(thickness_ratio))
            outfile.write('\n')

    print("FINAL AVERAGE ERROR: ", iou_error)

def matchPoints(contourPoints, goodPoints):
    epsilon = 10
    finalIndeces = set()
    finalPoints = []

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
