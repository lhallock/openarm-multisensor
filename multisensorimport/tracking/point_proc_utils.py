#!/usr/bin/env python3
"""Utility functions to filter image points.

This module contains functions to extract, process, and filter contour points
based on various metrics (e.g., Shi-Tomasi corner score).
"""
import os

import cv2
import numpy as np
import scipy

from multisensorimport.tracking.image_proc_utils import *


def extract_contour_pts_png(filename):
    """Extract points from largest contour in PNG image.

    This function is used to extract ordered points along the largest detected
    contour in the provided PNG image and format them for use by OpenCV image
    tracking. In particular, this function is used to extract the fascial
    border of the brachioradialis muscle in a mask manually segmented from a
    given ultrasound frame. Typically used to initialize points to track.

    Args:
        filename (str): full path to PNG file

    Returns:
        numpy.ndarray of contour points
    """
    # convert PNG to OpenCV mask
    img = cv2.imread(filename, -1)
    alpha_channel = img[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 254, 255,
                            cv2.THRESH_BINARY)  # binarize mask
    color = img[:, :, :3]
    new_img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    new_img = (255 - new_img)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)

    # extract contours from processed contour mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # convert largest contour to tracking-compatible numpy array
    points = []
    for i in range(len(contours[0])):
        points.append(np.array(contours[0][i], dtype=np.float32))
    np_points = np.array(points)

    return np_points


def extract_contour_pts_pgm(filename):
    """Extract points from largest contour in PGM image.

    This function is used to extract ordered points along the largest detected
    contour in the provided PGM image and format them for use by OpenCV image
    tracking. In particular, this function is used to extract the fascial
    border of the brachioradialis muscle in a mask manually segmented from a
    given ultrasound frame. Typically used to initialize points to track.

    Args:
        filename (str): full path to PNG file

    Returns:
        numpy.ndarray of contour points
    """
    # read in image
    img = cv2.imread(filename, -1)
    # convert image to grayscale if it is color

    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_level = 127

    # binarize image
    _, binarized = cv2.threshold(img, threshold_level, 255, cv2.THRESH_BINARY)

    # flip image (need a white object on black background)
    flipped = cv2.bitwise_not(binarized)
    contours, _ = cv2.findContours(flipped, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # convert largest contour to tracking-compatible numpy array
    points = []
    for i in range(len(contours[0])):
        points.append(np.array(contours[0][i], dtype=np.float32))

    np_points = np.array(points)

    return np_points


def shi_tomasi_corner_score(point, block_size, img):
    """Calculates Shi-Tomasi corner score for point in the given image.

    Args:
        point (numpy.ndarray): single-element array whose element is a
            numpy.ndarray of x-y pixel coordinates
        block_size (TODO type): block size determining neighborhood around
            point to consider
        img (TODO type): image in which corner score is being calculated

    Returns:
        float Shi-Tomasi corner score for given point in given image
    """
    point = point[0]

    # get x-y coordinates
    x = int(round(point[0]))
    y = int(round(point[1]))

    # set dimension of Sobel derivative kernel
    k_size = 5

    # obtain eigenvalues and corresponding eigenvectors of image structure tensor
    eigen = cv2.cornerEigenValsAndVecs(img, block_size, ksize=k_size)

    # extract eigenvalues
    lambda_one = get_image_value(x, y, eigen)[0]
    lambda_two = get_image_value(x, y, eigen)[1]

    # return Shi-Tomasi corner score (min of eigenvalues)
    return min(lambda_one, lambda_two)


def filter_points(run_params,
                  window_size,
                  pts,
                  filter_type,
                  img,
                  percent,
                  keep_bottom=False):
    """Filter given contour points by Shi-Tomasi corner score.

    This method extracts the top specified percentage of contour points based
    on their Shi-Tomasi corner score and is used in FRLK, BFLK, and SBLK
    algorithms.

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking
        window_size (TODO type): size of neighborhood around point to consider
            when calculating corner score
        pts (TODO type): TODO description
        img (TODO type): image used to calculate corner scores
        percent (TODO type): percent of points to keep based on corner score
        keep_bottom (bool): whether points along bottom of contour should be
            kept regardless of score (used to ensure contour contains bottom of
            fascia)

    Returns:
        numpy.ndarray of filtered points
        numpy.ndarray of their corresponding indices in the original contour
    """
    # select image filter, determined by filterType argument
    filter = get_filter_from_num(filter_type)

    # apply filter
    filtered_img = filter(img, run_params)
    x = (len(pts))

    # convert pts from np array to list for convenience, create dict for sorting
    pts = list(pts)
    ind_to_score_map = dict()
    ind_to_y_map = dict()
    for i in range(len(pts)):
        point = pts[i]
        corner_score = shi_tomasi_corner_score(point, window_size, filtered_img)
        ind_to_score_map[i] = corner_score
        ind_to_y_map[i] = pts[i][0][1]

    filtered_points = []
    filtered_points_ind = []

    # convert map to  list of 2-tuples (key, value), sorted in descending order
    # by value; key is index of point in pts list
    sorted_corner_mapping = sorted(ind_to_score_map.items(),
                                   key=lambda x: x[1],
                                   reverse=True)
    sorted_y_mapping = sorted(ind_to_y_map.items(),
                              key=lambda x: x[1],
                              reverse=True)

    # get top percent of points
    for i in range(0, int(np.rint(percent * len(sorted_corner_mapping)))):
        points_ind = sorted_corner_mapping[i][0]
        filtered_points.append(pts[points_ind])
        filtered_points_ind.append(points_ind)

    # keep bottom-most points if specified
    if keep_bottom:
        for i in range(run_params.num_bottom):
            points_ind = sorted_y_mapping[i][0]
            filtered_points.append(pts[points_ind])
            filtered_points_ind.append(points_ind)

    return np.array(filtered_points), np.array(filtered_points_ind)


def separate_points(run_params, img, pts):
    """Separate points into two subsets for use by different filters.

    This method separates a given set of contour points into two sets of points
    to be tracked by separate filters (i.e., a "coarse" and "aggressive"
    bilateral filter). Each subset contains the top percentage (as specified in
    the input parameter object) of contour points based on Shi-Tomasi corner
    score, as evaluated on the correspondingly filtered image. If points are
    considered high quality in both filtered images, they are tracked in the
    more aggressively filtered frames.

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking
        img (TODO type): image used to calculate corner scores
        pts (numpy.ndarray): array of points to be filtered and separated

    Returns:
        numpy.ndarray of points to be tracked by fine/aggressive filter
        numpy.ndarray of their corresponding indices in the original contour
        numpy.ndarray of points to be tracked by coarse/less-aggressive filter
        numpy.ndarray of their corresponding indices in the original contour
    """
    # determine image filters to use (bilateral filters)
    fine_filter_type = 2
    course_filter_type = 3

    # TODO add explanation for magic number
    corner_window_size = 7

    # separate points into two potentially overlapping subsets of pts
    fine_pts, fine_pts_inds = filter_points(run_params, corner_window_size, pts,
                                            fine_filter_type, img,
                                            run_params.percent_fine)
    course_pts, course_pts_inds = filter_points(run_params, corner_window_size,
                                                pts, course_filter_type, img,
                                                run_params.percent_course)

    # remove overlap between two subsets; a point in both sets will be removed
    # from course_pts and kept in fine_pts
    course_kept_indeces = set()
    for i in range(len(course_pts)):
        course_pt = course_pts[i]
        add = True
        for j in range(len(fine_pts)):
            fine_pt = fine_pts[j]
            if np.linalg.norm(course_pt - fine_pt) < 0.001:
                add = False
        if add:
            course_kept_indeces.add(i)

    course_to_keep = []
    course_to_keep_inds = []
    for index in course_kept_indeces:
        course_to_keep.append(course_pts[index])
        course_to_keep_inds.append(course_pts_inds[index])

    # convert to numpy arrays and return
    course_pts = np.array(course_to_keep)
    course_pts_inds = np.array(course_to_keep_inds)

    return fine_pts, fine_pts_inds, course_pts, course_pts_inds


def order_points(points_one, points_one_inds, points_two, points_two_inds):
    """Generate single contour from two sets of points.

    This method combines two subsets of contour points into a single contour
    while maintaining their original counter-clockwise order, and is used when
    recombining points that have been tracked using different algorithms.

    Args:
        points_one (numpy.ndarray): first subset of contour points
        points_one_inds (TODO type): indices of first subset of points in
            original contour
        points_two (numpy.ndarray): second subset of contour points
        points_two_inds (TODO type): indices of second subset of points in
            original contour

    Returns:
        numpy.ndarray of points_one and points_two, ordered as in original
            contour
    """
    # init dictionary mapping index to point
    point_dict = dict()

    # populate dictionary
    for i in range(len(points_one)):
        point = points_one[i]
        point_ind = points_one_inds[i]
        point_dict[point_ind] = point
    for i in range(len(points_two)):
        point = points_two[i]
        point_ind = points_two_inds[i]
        point_dict[point_ind] = point

    # order dictionary by key and append points
    pts = []
    for key in sorted(point_dict.keys()):
        pts.append(point_dict[key])

    return np.array(pts)


def thickness(points):
    """Find contour thickness in x and y dimensions.

    This method determines contour thickness (i.e., maximal extent/width) in x
    and y dimensions given a set of contour points. Thickness is defined as the
    maximal difference between two points along that dimension.

    Args:
        points (numpy.ndarray): array of contour points

    Returns:
        float thickness in x dimension
        float thickness in y dimension
    """
    # initialize min and max values
    min_x = float("inf")
    max_x = -1 * float("inf")

    min_y = float("inf")
    max_y = -1 * float("inf")

    # find max and min of x and y
    for point in points:
        x = point[0]
        y = point[1]
        min_x = min(x, min_x)
        max_x = max(x, max_x)

        min_y = min(y, min_y)
        max_y = max(y, max_y)

    # return difference
    return (max_x - min_x), (max_y - min_y)


def get_image_value(x, y, img):
    """Get pixel value at specified x-y coordinate of image.

    Args:
        x (int): horizontal pixel coordinate
        y (int): vertical pixel coordinate
        img (TODO type): TODO description

    Returns:
        TODO type pixel value at specified coordinate
    """
    return img[y][x]
