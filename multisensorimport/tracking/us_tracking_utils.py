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


def track_pts(filedir, finePts, coursePts, lk_params, viz=True, fineFilterType = 0, courseFilterType = 0):
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

    # combine course and fine points
    pts = np.concatenate((finePts, coursePts), axis=0)

    # keep track of contour areas
    contour_areas = []
    # add first contour area
    contour_areas.append(cv2.contourArea(pts))

    # create OpenCV window (if visualization is desired)
    if viz:
        cv2.namedWindow('Frame')

    # set filters (course filter is a less aggresive filter, fine_filter more aggressive)
    course_filter = get_filter_from_num(courseFilterType)
    fine_filter = get_filter_from_num(fineFilterType)

    # track and display specified points through images
    first_loop = True
    for filename in sorted(os.listdir(filedir)):
        if filename.endswith('.pgm'):
            filepath = filedir + filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                # apply filter to frame
                old_frame_course = course_filter(old_frame)
                old_frame_fine = fine_filter(old_frame)
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)
                # print("SHAPE", frame.shape)
                # apply filter to frame
                frame_course = course_filter(frame)
                frame_fine = fine_filter(frame)
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


                # # CSRT tracking
                # csrt_tracking(old_frame, frame, pts, (7, 7))

                # calculate new point locations for fine_points using frame filtered by the fine filter
                new_fine_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame_fine, frame_fine, finePts, None, **lk_params)

                # calculate new point locations for course_points using frame filtered by the course filter
                new_course_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame_course, frame_course, coursePts, None, **lk_params)

                # save old frame for optical flow calculation
                old_frame_course = frame_course.copy()
                old_frame_fine = frame_fine.copy()

                # reset point locations
                finePts = new_fine_pts
                coursePts = new_course_pts
                pts = np.concatenate((finePts, coursePts), axis=0)
                print(len(pts))
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
    filter = get_filter_from_num(filterType)

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
def shi_tomasi_corner_score(point, block_size, img):
    # point is a 1 element numpy array whose element is a numpy array of x, y so unpack
    point = point[0]
    # get x,y coords
    x = int(round(point[0]))
    y = int(round(point[1]))

    # sets dimension of Sobel derivative kernel
    k_size = 5
    # obtain eigenvalues and corresponding eigenvectors of image structure tensor
    eigen = cv2.cornerEigenValsAndVecs(img, block_size, ksize = k_size)

    # extract eigenvalues
    lambda_one = get_image_value(x, y, eigen)[0]
    lambda_two = get_image_value(x, y, eigen)[1]

    # return Shi-Tomasi corner score (min of eigenvalues)
    return min(lambda_one, lambda_two)


def filter_points(window_size, pts, eps, filter_type, img, percent):

    # select image filter, determined by filterType argument
    filter = get_filter_from_num(filter_type)

    # apply filter
    filtered_img = filter(img)

    # convert pts from np array to list for convenience, create dict for sorting
    pts = list(pts)
    map = dict()
    filtered_pts = []
    for i in range(len(pts)):
        point = pts[i]
        corner_score = shi_tomasi_corner_score(point, 7, filtered_img)
        map[i] = corner_score
        if (corner_score >= eps):
            filtered_pts.append(point)

    filtered_points = []

    # converts map to a list of 2-tuples (key, value), which are in sorted order by value
    # key is index of point in the pts list
    sorted_mapping = sorted(map.items(), key=lambda x: x[1], reverse=True)

    # get top 60% of points
    for i in range(0, round(percent * len(sorted_mapping))):
        filtered_points.append(pts[sorted_mapping[i][0]])

    return np.array(filtered_points)


def csrt_tracking(prev_img, curr_img, points, window_size):

    new_points = []

    for point in points:
        x = point[0][0]
        y = point[0][1]
        tracker = cv2.TrackerCSRT_create()
        bounding_box = (x - window_size[0]//2, y - window_size[1]//2, window_size[0], window_size[1])
        tracker.init(prev_img, bounding_box)

        ret, new_bounding_box = tracker.update(curr_img)
        lower_x = new_bounding_box[0]
        lower_y = new_bounding_box[1]
        translation_x = (x - window_size[0]//2) - lower_x
        translation_y = (y - window_size[1]//2) - lower_y

        new_x = x + translation_x
        new_y = y + translation_y

        new_points.append(np.array([np.array([new_x, new_y])]))

    color_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
    # visualize
    for point in new_points:
        x = int(np.rint(point[0][0]))
        y = int(np.rint(point[0][1]))
        cv2.circle(color_curr_img, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Frame2', color_curr_img)
    key = cv2.waitKey(1)
    time.sleep(0.01)

    new_points = np.array(new_points)

    return new_points.reshape(points.shape)



def get_image_value(x, y, img):
    return img[y][x]


def get_filter_from_num(filter_type):
    filter = None
    if filter_type == 1:
        filter = median_filter
    elif filter_type == 2:
        filter = fine_bilateral_filter
    elif filter_type == 3:
        filter = course_bilateral_filter
    elif filter_type == 4:
        filter = anisotropic_diffuse
    else:
        filter = no_filter
    return filter

# image filtering
def no_filter(color_image):
    return color_image

def median_filter(color_image):
    # hyperparameter
    kernelSize = 5
    return cv2.medianBlur(color_image, kernelSize)

def fine_bilateral_filter(color_image):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    diam = 35
    sigmaColor = 80
    sigmaSpace = 80
    bilateralColor = cv2.bilateralFilter(color_image, diam, sigmaColor, sigmaSpace)
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)

def course_bilateral_filter(color_image):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
    # hyperparameters
    diam = 20
    sigmaColor = 10
    sigmaSpace = 10
    bilateralColor = cv2.bilateralFilter(color_image, diam, sigmaColor, sigmaSpace)
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)


def anisotropic_diffuse(color_image):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
    # hyperparameters
    alphaVar = 0.1
    KVar = 5
    nitersVar = 5
    diffusedColor = cv2.ximgproc.anisotropicDiffusion(src = color_image, alpha = alphaVar, K = KVar, niters = nitersVar)
    return cv2.cvtColor(diffusedColor, cv2.COLOR_RGB2GRAY)

def otsu_binarization(grayImage):
    ret2,th2 = cv2.threshold(grayImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def canny(grayImage):
    edges = cv2.Canny(grayImage,180,200)
    return edges
