#!/usr/bin/env python3
"""Methods to initialize supporter points and use them to infer contour points.

This module contains functions to initialize supporter points (i.e.,
non-contour points that are easy to track) and to use them to infer non-tracked
points along the desired contour. These methods are employed when tracking
using the supporter-based Lucas-Kanade tracking (SBLK) algorithm. TODO: a
citation on what this work is based on would be helpful here.
"""
import numpy as np
import scipy
from scipy import stats

from multisensorimport.tracking.image_proc_utils import *
from multisensorimport.tracking.point_proc_utils import *


def initialize_supporters(run_params, READ_PATH, keyframe_path, init_img,
                          feature_params, lk_params, which_contour):
    """Initialize all point groups and parameters for SBLK tracking.

    This method reads in a contour from the specified image, separates contour
    points into two groups (tracked via Lucas-Kanade and tracked via
    supporters), determines good supporter points, and initializes their
    parameters.

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking
        READ_PATH (str): path to raw ultrasound frames
        key_frame_path (str): path to ground truth hand-segmented frames
        init_img (TODO type): first frame in ultrasound image sequence
        feature_params (TODO type): parameters to find good features to track
        lk_params (TODO type): parameters for Lucas-Kanade tracking
        which_contour (int): integer indicating if image containing contour is
            a PNG (1) or PGM (2)

    Returns:
        numpy.ndarray of points to be tracked via supporters
        numpy.ndarray of their corresponding indices in the original contour
        numpy.ndarray of points to be tracked via Lucas-Kanade
        numpy.ndarray of their corresponding indices in the original contour
        TODO type list of parameters for each supporter point
    """
    # extract contour
    if which_contour == 1:
        pts = extract_contour_pts_png(keyframe_path)
    else:
        pts = extract_contour_pts_pgm(keyframe_path)

    mean_x_pts = 0
    mean_y_pts = 0

    # find mean of contour point coordinates
    for contour_point in pts:
        x = contour_point[0][0]
        y = contour_point[0][1]
        mean_x_pts += x
        mean_y_pts += y

    mean_x_pts = mean_x_pts / len(pts)
    mean_y_pts = mean_y_pts / len(pts)

    # filter to be used:
    #    1: median filter
    #    2: fine bilateral filter
    #    3: coarse bilateral filter
    #    4: anisotropic diffusion
    #    other: no filter
    fineFilterNum = 2
    courseFilterNum = 3

    course_filter = get_filter_from_num(courseFilterNum)
    filtered_init_img = course_filter(init_img, run_params)

    # remove points w/ low Shi-Tomasi corner score (kept for LK tracking)
    # TODO not sure I understand this description; is "keep points w/ high
    # Shi-Tomasi corner score for LK tracking" accurate/better?
    lucas_kanade_points, lucas_kanade_points_indeces = filter_points(
        run_params,
        7,
        pts,
        fineFilterNum,
        init_img,
        run_params.fine_threshold,
        keep_bottom=True)

    # add first point to LK tracking (top left)
    lucas_kanade_points = np.append(lucas_kanade_points,
                                    np.array([pts[0]]),
                                    axis=0)
    lucas_kanade_points_indeces = np.append(lucas_kanade_points_indeces, 0)

    # initialize list of supporter-tracked points
    supporter_tracked_points = pts.copy()
    supporter_tracked_points_indeces = np.arange(0, len(pts))

    # restrict supporter-tracked points to those in desired region (i.e., top
    # right image quadrant, x>[mean x], y<[mean y])
    supporter_kept_indeces = set()
    for i in range(len(supporter_tracked_points)):
        supporter_tracked_point = supporter_tracked_points[i]
        add = (supporter_tracked_point[0][0] > mean_x_pts and
               supporter_tracked_point[0][1] < mean_y_pts)
        if add:
            supporter_kept_indeces.add(i)

    # list only top-right-quadrant points as determined above
    supporter_tracked_to_keep = []
    supporter_tracked_to_keep_inds = []
    for index in supporter_kept_indeces:
        supporter_tracked_to_keep.append(supporter_tracked_points[index])
        supporter_tracked_to_keep_inds.append(
            supporter_tracked_points_indeces[index])

    # reset supporter-tracked points list to right-top-quadrant points only
    supporter_tracked_points = np.array(supporter_tracked_to_keep)
    supporter_tracked_points_indeces = np.array(supporter_tracked_to_keep_inds)

    # remove points from LK tracking list if they're tracked via supporters
    LK_kept_indeces = set()
    for i in range(len(lucas_kanade_points)):
        lucas_kanade_point = lucas_kanade_points[i]
        add = True
        # loop through supporter points
        for j in range(len(supporter_tracked_points)):
            supporter_point = supporter_tracked_points[j]
            # if LK point is same as a supporter point or is in top right
            # quadrant, don't track it
            if ((np.linalg.norm(lucas_kanade_point - supporter_point) < 0.001)
                    or (lucas_kanade_point[0][0] > mean_x_pts and
                        lucas_kanade_point[0][1] < mean_y_pts)):
                add = False
        if add:
            LK_kept_indeces.add(i)

    # keep only non-supporter points for LK tracking as determined above
    LK_to_keep = []
    LK_to_keep_inds = []
    for index in LK_kept_indeces:
        LK_to_keep.append(lucas_kanade_points[index])
        LK_to_keep_inds.append(lucas_kanade_points_indeces[index])

    # reset LK-tracked points list
    lucas_kanade_points = np.array(LK_to_keep)
    lucas_kanade_points_indeces = np.array(LK_to_keep_inds)

    # find supporters based on good points
    supporters = cv2.goodFeaturesToTrack(filtered_init_img,
                                         mask=None,
                                         **feature_params)

    # initialize supporters
    supporter_params = []
    for i in range(len(supporter_tracked_to_keep)):
        supporter_tracked_point = supporter_tracked_to_keep[i][0]
        initial_variance = 10
        _, run_params = initialize_supporters_for_point(
            supporters, supporter_tracked_point, initial_variance)
        supporter_params.append(run_params)

    return supporter_tracked_points, supporter_tracked_points_indeces, lucas_kanade_points, lucas_kanade_points_indeces, supporters, supporter_params


def initialize_supporters_for_point(supporter_points, target_point, variance):
    """Format supporter point list and initialize parameters for
    supporter-tracked point.

    This method reformats (TODO: sharpen; what does this mean?) the list of
    supporter points and initializes their corresponding parameters
    (displacement and convariance) for a given supporter-tracked target point.

    Args:
        supporter_points (numpy.ndarray): array of 1-element arrays, where each
            element is a 2-element array containing supporter point locations
        target point (numpy.ndarray): x-y coordinates of target point to track
        variance (float): initial variance for each element of the displacement
            (TODO: what is "element of the displacement"?)

    Returns:
        list of 2-element arrays containing supporter point locations
        TODO type supporter parameters
    """
    # initialize empty lists
    supporters = []
    supporter_params = []

    for i in range(len(supporter_points)):
        # extract numpy.ndarray of the supporter location
        supporter_point = supporter_points[i][0]
        supporters.append(supporter_point)
        # initialize displacement average w/ initial displacement, diagonal
        # covariance
        supporter_params.append(
            (target_point - supporter_point, variance * np.eye(2)))

    return supporters, supporter_params


def apply_supporters_model(run_params, predicted_target_point,
                           prev_feature_points, feature_points, feature_params,
                           use_tracking, alpha):
    """Execute model learning or prediction based on conditions of image
    tracking.

    TODO: I don't understand this one-line explanation; add a short paragraph
    explanation here.

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking
        predicted_target_point (numpy.ndarray): 2-element array of x-y
            coordinates of LK tracking prediction of target point
        prev_feature_points (TODO type): list of x-y coordinates of feature
            (supporter) points in previous frame
        feature_points (TODO type): list of x-y coordinates of feature
            (supporter) points in current frame
        feature_params (list): list of 2-tuples of (displacement vector
            average, covariance matrix average) for each feature point
        use_tracking (bool): whether to return pure Lucas-Kanade prediction or
            supporters-based prediction (former used in first n frames for
            training, where n is determined a priori)
        alpha (TODO type): learning rate for exponential forgetting principle

    Returns:
        TODO type predicted location of target point
        TODO type updated supporter point parameters
    """
    # reformat feature points for easier processing
    feature_points = format_supporters(feature_points)

    # round to integer so that the prediction lands on pixel coordinates
    predicted_target_point = np.round(predicted_target_point).astype(int)

    # initialize value to return
    target_point_final = None

    # initialize new target param tuple array
    new_feature_params = []

    # if specified, use LK tracking to track point directly and update
    # supporter parameters
    # TODO: make sure I'm understanding these variables correctly
    if use_tracking:

        target_point_final = predicted_target_point

        # update supporter feature parameters
        for i in range(len(feature_points)):
            curr_feature_point = feature_points[i]
            curr_feature_point = np.round(curr_feature_point).astype(int)
            # displacement vector between the current feature and the target point
            curr_displacement = target_point_final - curr_feature_point
            # previous average for displacement
            prev_displacement_average = feature_params[i][0]
            # update displacement average using exponential forgetting principle
            new_displacement_average = alpha * prev_displacement_average + (
                1 - alpha) * curr_displacement
            displacement_mean_diff = curr_displacement - new_displacement_average
            # compute current covariance matrix
            curr_covariance_matrix = displacement_mean_diff.reshape(
                2, 1) @ displacement_mean_diff.reshape(1, 2)
            # update covariance matrix average using exponential forgetting principle
            prev_covariance_matrix = feature_params[i][1]
            new_covariance_matrix = alpha * prev_covariance_matrix + (
                1 - alpha) * curr_covariance_matrix

            new_feature_params.append(
                (new_displacement_average, new_covariance_matrix))

    # otherwise, track point as a weighted average of mean supporter point
    # displacements, weighted by probability of supporter and prediction (TODO:
    # what does "probability of supporter and prediction" mean?)
    else:
        # quantities used in calculation TODO: sharpen
        numerator = 0
        denominator = 0
        displacements = []

        for i in range(len(feature_points)):
            feature_point = feature_points[i]
            prev_feature_point = prev_feature_points[i]
            displacement_norm = np.linalg.norm(feature_point -
                                               prev_feature_point)
            # determine weight to assign to point (function of displacement)
            weight = weight_function(run_params, displacement_norm)
            covariance = feature_params[i][1]
            displacement = feature_params[i][0]

            numerator += (
                weight *
                (displacement + feature_point)) / np.linalg.det(covariance)
            denominator += weight / np.linalg.det(covariance)

        # return weighted average
        target_point_final = numerator / denominator

    # if supporter-based tracking was used, return old feature parameters
    if new_feature_params == []:
        return target_point_final, feature_params

    # otherwise, return updated feature parameters
    else:
        return target_point_final, new_feature_params


def weight_function(run_params, displacement_norm):
    """Determine prediction weight of given supporter point.

    This method determines the weight to apply to each supporter point when
    using it for prediction of target point based on the norm of its
    displacement vector. TODO: add sentence on a) the specific calculation it
    performs, and b) why that's a good idea for weighting

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking, including scalar alpha
        displacement_norm (float): L2 norm of relevant supporter point's
            displacement vector

    Returns:
        float weighting to apply to supporter point when tracking target point

    """
    alpha = run_params.displacement_weight

    return 1 + (alpha * displacement_norm)


def format_supporters(supporter_points):
    """Reformat array of supporter point locations into list of arrays.

    Args:
        supporter_points (numpy.ndarray): array of 1-element arrays, where each
            element is a 2-element array containing supporter point locations

    Returns:
        list of 2-element arrays containing supporter point locations
    """
    supporters = []
    for i in range(len(supporter_points)):
        supporters.append(supporter_points[i][0])
    return supporters
