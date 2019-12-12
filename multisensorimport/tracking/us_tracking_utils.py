#!/usr/bin/env python3
"""Utility functions for ultrasound tracking, processing, and visualization.

This module contains functions to extract desired contour points for tracking,
track these points through ultrasound scans, and write desired time series
muscle deformation parameters to CSV files compatible with the
dataobj.TimeSeriesData class.

"""

import time
import os

import cv2
import numpy as np

def extract_contour_pts(filename):
    """Extract points from largest contour in PNG image.

    This function is used to extract ordered points along the largest detected
    contour in the provided PNG image and format them for use by OpenCV image
    tracking. In particular, this function is used to extract the fascial
    border of the brachioradialis muscle in a mask manually segmented from a
    given ultrasound frame.

    Args:
        filename (str): full path to PNG file

    Returns:
        numpy.ndarray of contour points
    """
    # convert PNG to OpenCV mask
    img = cv2.imread(filename, -1)
    alpha_channel = img[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)  # binarize mask
    color = img[:, :, :3]
    new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    new_img = (255-new_img)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)

    # extract contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # convert largest contour to tracking-compatible array
    points = []
    for i in range(len(contours[0])):
        points.append(np.array(contours[0][i], dtype=np.float32))
    np_points = np.array(points)

    return np_points


def track_pts(filedir, pts, lk_params, viz=True):
    """Track specified points through all (ordered) images in directory.

    This function is used to track, record, and visualize points through a full
    directory of images, recording values of interest at each frame. In
    particular, this function is used to calculate the cross-sectional area of
    the brachioradialis muscle at each frame.

    Args:
        filedir (str): directory in which files are stored, including final '/'
        pts (numpy.ndarray): list of points to track; assumed to correspond to
            the first image file in filedir, and to be in counter-clockwise
            order (TODO: check this)
        lk_params (dict): parameters for Lucas-Kanade image tracking in OpenCV
            TODO: set appropriate default values
        viz (bool): whether to visualize the tracking process

    Returns:
        list of contour areas of each frame
    """
    # keep track of contour areas
    contour_areas = []
    contour_areas.append(cv2.contourArea(pts))

    # create OpenCV window
    cv2.namedWindow('Frame')

    # track and display specified points through images
    first_loop = True
    for filename in sorted(os.listdir(filedir)):
        if filename.endswith('.pgm'):
            filepath = filedir + filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)

                # calculate new point locations
                new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame, frame, pts, None, **lk_params)

                # save old frame for optical flow calculation
                old_frame = frame.copy()

                # reset point locations
                pts = new_pts
                for i in range(len(pts)):
                    x, y = pts[i].ravel()
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # display to frame
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(1)
                if key == 27: # stop on escape key
                    break
                time.sleep(0.05)

                # append new contour area
                contour_areas.append(cv2.contourArea(pts))

    cv2.destroyAllWindows()

    return contour_areas


def write_us_csv(outfile, vals, val_labels=None):
    pass
