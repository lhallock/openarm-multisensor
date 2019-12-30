#!/usr/bin/env python3
"""Utility functions for ultrasound tracking, processing, and visualization.

This module contains functions to extract desired contour points for tracking,
track these points through ultrasound scans, and write desired time series
muscle deformation parameters to CSV files compatible with the
dataobj.TimeSeriesData class.

"""

import time
import os

import csv

import cv2
import numpy as np
from scipy import signal

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


def track_pts(filedir, pts, lk_params, viz=True, filterType = 0):
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
        filter (int): what kind of filter to use (0 for none, 1 for median, 2 for anisotropic diffusion)

    Returns:
        list of contour areas of each frame
    """
    # keep track of contour areas
    contour_areas = []
    # add first contour area
    contour_areas.append(cv2.contourArea(pts))

    # create OpenCV window (if visualization is desired)
    if viz:
        cv2.namedWindow('Frame')

    # set which filter function to use
    filter = None

    if filterType == 1:
        filter = medianFilter
    elif filterType == 2:
        filter = bilateralFilter
    elif filterType == 3:
        filter = anisotropicDiffuse
    else:
        filter = noFilter

    # track and display specified points through images
    first_loop = True
    for filename in sorted(os.listdir(filedir)):
        if filename.endswith('.pgm'):
            filepath = filedir + filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                # apply filter to frame
                old_frame = filter(old_frame)
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)
                # print("SHAPE", frame.shape)
                # apply filter to frame
                frame = filter(frame)

                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # calculate new point locations
                new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame, frame, pts, None, **lk_params)

                # save old frame for optical flow calculation
                old_frame = frame.copy()

                # reset point locations
                pts = new_pts
                for i in range(len(pts)):
                    x, y = pts[i].ravel()

                    cv2.circle(frame_color, (x, y), 5, (0, 255, 0), -1)

                # display to frame
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27: # stop on escape key
                        break
                    time.sleep(0.01)

                # append new contour area
                contour_areas.append(cv2.contourArea(pts))

    if viz:
        cv2.destroyAllWindows()

    return contour_areas


def track_pts_to_keyframe(filedir, pts, lk_params, viz=True, filterType = 0):
    """Track specified points, resetting on ground-truth keyframe.

    This function is used to track, record, and visualize points through a full
    directory of images as in function track_pts; the only difference is that
    when a manually-segmented keyframe is reached, the points reset to the
    specified contour. As above, this function is used to calculate the
    cross-sectional area of the brachioradialis muscle at each frame.

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

    # import keyframe dictionary
    keyframes = get_keyframes(filedir)

    # create OpenCV window (if visualization is desired)
    if viz:
        cv2.namedWindow('Frame')

    # set which filter function to use
    filter = None

    if filterType == 1:
        filter = medianFilter
    elif filterType == 2:
        filter = anisotropicDiffuse
    else:
        filter = noFilter

    # track and display specified points through images
    first_loop = True
    for filename in sorted(os.listdir(filedir)):
        if filename.endswith('.pgm'):
            filepath = filedir + filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                # apply filter to frame
                old_frame = filter(old_frame)
                first_loop = False

            else:

                # read in new frame
                frame = cv2.imread(filepath, -1)
                # apply filter to frame
                frame = filter(frame)

                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # if it's a keyframe, get contour points from that
                filenum = filename.split('.')[0]
                if filenum in keyframes:
                    print("KEY FRAME!")
                    keyframe_path = filedir + str(keyframes[filenum]) + '.png'
                    new_pts = extract_contour_pts(keyframe_path)

                # otherwise, calculate new point locations from optical flow
                else:
                    new_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame, frame, pts, None, **lk_params)

                # save old frame for optical flow calculation
                old_frame = frame.copy()

                # reset point locations
                pts = new_pts
                for i in range(len(pts)):
                    x, y = pts[i].ravel()
                    #  cv2.circle(frame_color, (x, y), 5, (0, 255, 0), -1)

                # display to frame
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27: # stop on escape key
                        break
                    time.sleep(0.01)

                # append new contour area
                contour_areas.append(cv2.contourArea(pts))

    if viz:
        cv2.destroyAllWindows()

    return contour_areas

def get_keyframes(filedir):
    """Read in dictionary of keyframe numbers and corresponding file numbers.

    This function is used to generate a dictionary containing frame numbers as
    keys (only frames that are keyframes) and mapping them to their
    corresponding keyframe number (i.e., filename). This function assumes the
    existence of a file called 'keyframes.csv' in the directory of interest.

    Args:
        filedir (str): directory in which 'keyframes.csv' is stored, including
            final '/'

    Returns:
        dict mapping keyframe frame numbers to keyframe numbers
    """
    keyframes = {}

    filepath = filedir + 'keyframes.csv'
    print(filepath)
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            frame = str(row[1])
            key_filenum = str(row[0])
            keyframes[frame] = key_filenum

    return keyframes

def write_us_csv(outfile, vals, val_labels=None):
    pass

# various helpers methods
def shiTomasiCornerScore(point, blockSize, img):
    print(point[0])
    point = point[0]
    x = int(round(point[0]))
    y = int(round(point[1]))

    # sets dimension of Sobel derivative kernel
    kSize = 3
    # obtain eigenvalues and corresponding eigenvectors of image structure tensor
    eigen = cv2.cornerEigenValsAndVecs(img, blockSize, ksize = kSize)

    # extract eigenvalues
    lambdaOne = getImageValue(x, y, eigen)[0]
    lambdaTwo = getImageValue(x, y, eigen)[1]

    # return Shi-Tomasi corner score (min value of eigenvalues)
    return min(lambdaOne, lambdaTwo)

def filterPoints(window_size, pts, eps, img):
    pts = list(pts)
    filteredPts = []
    for point in pts:
        if (shiTomasiCornerScore(point, window_size, img) >= eps):
            filteredPts.append(point)
    return np.array(filteredPts)

def getImageValue(x, y, img):
    return img[y][x]


# filtering
def noFilter(colorImage):
    return colorImage

def medianFilter(colorImage):
    # hyperparameter
    kernelSize = 5
    return cv2.medianBlur(colorImage, kernelSize)

def bilateralFilter(colorImage):
    colorImage = cv2.cvtColor(colorImage, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    diam = 30
    sigmaColor = 100
    sigmaSpace = 100
    bilateralColor = cv2.bilateralFilter(colorImage, diam, sigmaColor, sigmaSpace)
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)


def anisotropicDiffuse(colorImage):
    colorImage = cv2.cvtColor(colorImage, cv2.COLOR_GRAY2RGB)
    print("SHAPE", colorImage)
    # hyperparameters
    alphaVar = 0.1
    KVar = 5
    nitersVar = 5
    diffusedColor = cv2.ximgproc.anisotropicDiffusion(src = colorImage, alpha = alphaVar, K = KVar, niters = nitersVar)
    return cv2.cvtColor(diffusedColor, cv2.COLOR_RGB2GRAY)
