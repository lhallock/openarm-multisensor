#!/usr/bin/env python3
"""Methods to initialize supporter points and use them to infer contour points.

This module contains functions to initialize supporter points (i.e.,
non-contour points that are easy to track) and use them to infer non-tracked
points along the desired contour. These methods are employed when tracking
using the supporter-based Lucas-Kanade tracking (SBLK) algorithm.

The supporters algorithm is based on the following work:
H. Grabner, J. Matas, L. Van Gool, and P. Cattin. "Tracking the Invisible:
    Learning Where the Object Might Be." Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), 2010.
"""
import numpy as np

import cv2

from multisensorimport.tracking.image_proc_utils import *
from multisensorimport.tracking.point_proc_utils import *


def initialize_supporters(run_params, read_path, keyframe_path, init_img,
                          feature_params, lk_params, which_contour):
    """Initialize all point groups and parameters for SBLK tracking.

    This method reads in a contour from the specified image, separates contour
    points into two groups (tracked via Lucas-Kanade and tracked via
    supporters), determines good supporter points, and initializes their
    parameters.

    Args:
        run_params (ParamValues): values of parameters used in tracking
        read_path (str): path to raw ultrasound frames
        keyframe_path (str): path to ground truth hand-segmented frames
        init_img (numpy.ndarray): first frame in ultrasound image sequence
        feature_params (dict): parameters to find good features to track
        lk_params (dict): parameters for Lucas-Kanade tracking
        which_contour (int): integer indicating if image containing contour is
            a PNG (1) or PGM (2)

    Returns:
        numpy.ndarray of points to be tracked via supporters
        numpy.ndarray of the corresponding indices of the supporter-tracked
            points in the original contour
        numpy.ndarray of points to be tracked via Lucas-Kanade
        numpy.ndarray of the corresponding indices of the Lucas-Kanade-tracked
            points in the original contour
        numpy.ndarray of supporter points
        list of parameters used for each supporter point
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
    fine_filter_num = 2
    coarse_filter_num = 3

    coarse_filter = get_filter_from_num(coarse_filter_num)
    filtered_init_img = coarse_filter(init_img, run_params)

    # keep points with high Shi-Tomasi corner score for LK tracking
    lucas_kanade_points, lucas_kanade_points_indices = filter_points(
        run_params,
        7,
        pts,
        fine_filter_num,
        init_img,
        run_params.fine_threshold,
        keep_bottom=True)

    # add first point to LK tracking (top left)
    lucas_kanade_points = np.append(lucas_kanade_points,
                                    np.array([pts[0]]),
                                    axis=0)
    lucas_kanade_points_indices = np.append(lucas_kanade_points_indices, 0)

    # initialize list of supporter-tracked points
    supporter_tracked_points = pts.copy()
    supporter_tracked_points_indices = np.arange(0, len(pts))

    # restrict supporter-tracked points to those in desired region (i.e., top
    # right image quadrant, x>[mean x], y<[mean y])
    supporter_kept_indices = set()
    for i in range(len(supporter_tracked_points)):
        supporter_tracked_point = supporter_tracked_points[i]
        add = (supporter_tracked_point[0][0] > mean_x_pts and
               supporter_tracked_point[0][1] < mean_y_pts)
        if add:
            supporter_kept_indices.add(i)

    # list only top-right-quadrant points as determined above
    supporter_tracked_to_keep = []
    supporter_tracked_to_keep_inds = []
    for index in supporter_kept_indices:
        supporter_tracked_to_keep.append(supporter_tracked_points[index])
        supporter_tracked_to_keep_inds.append(
            supporter_tracked_points_indices[index])

    # reset supporter-tracked points list to right-top-quadrant points only
    supporter_tracked_points = np.array(supporter_tracked_to_keep)
    supporter_tracked_points_indices = np.array(supporter_tracked_to_keep_inds)

    # remove points from LK tracking list if they're tracked via supporters
    lk_kept_indices = set()
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
            lk_kept_indices.add(i)

    # keep only non-supporter points for LK tracking as determined above
    lk_to_keep = []
    lk_to_keep_inds = []
    for index in lk_kept_indices:
        lk_to_keep.append(lucas_kanade_points[index])
        lk_to_keep_inds.append(lucas_kanade_points_indices[index])

    # reset LK-tracked points list
    lucas_kanade_points = np.array(lk_to_keep)
    lucas_kanade_points_indices = np.array(lk_to_keep_inds)

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

    return (supporter_tracked_points, supporter_tracked_points_indices,
            lucas_kanade_points, lucas_kanade_points_indices, supporters,
            supporter_params)


def initialize_supporters_for_point(supporter_points, target_point, variance):
    """Format single point supporter point list and supporter point parameters.

    This method reformats an input array of supporter points into a list of
    supporter points for easier use in tracking algorithms. It also initializes
    the supporter points and parameters used for supporter-based tracking.

    Args:
        supporter_points (numpy.ndarray): array of 1-element arrays, where each
            element is a 2-element array containing supporter point locations
        target_point (numpy.ndarray): x-y coordinates of target point to track
        variance (float): initial variance used for supporter based tracking
            (see paper cited above for algorithmic details)

    Returns:
        list of 2-element arrays containing supporter point locations
        list of 2-element tuples containing initial displacement and covariance
            matrices for each supporter point
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
    """Perform single supporters-based tracking update.

    This method performs a supporters-based tracking update to find the final
    predicted location of the target point. Depending on the value of input
    argument use_tracking, the method will execute the following:

        1. If use_tracking, then the new target point location predicted by
            Lucas-Kanade (also provided as an argument) is trusted, and this
            method returns this value as the final target location. Parameters
            for each supporter are updated based on this predicted target.

        2. If not use_tracking, then the new target point location is predicted
            based on supporter points, and no parameter updates are performed.

    Args:
        run_params (ParamValues): values of parameters used in tracking
        predicted_target_point (numpy.ndarray): 2-element array of x-y
            coordinates of LK tracking prediction of target point
        prev_feature_points (list): list of x-y coordinates of feature
            (supporter) points in previous frame
        feature_points (list): list of x-y coordinates of feature
            (supporter) points in current frame
        feature_params (list): list of 2-tuples of (displacement vector
            average, covariance matrix average) for each feature point
        use_tracking (bool): whether to return pure Lucas-Kanade prediction or
            supporters-based prediction (former used in first n frames for
            training, where n is determined a priori)
        alpha (float): update rate for exponential forgetting principle

    Returns:
        numpy.ndarray containing predicted location of target point
        list of updated supporter point parameters
    """
    # reformat feature points for easier processing
    feature_points = format_supporters(feature_points)

    # round to integer so that the prediction lands on pixel coordinates
    predicted_target_point = np.round(predicted_target_point).astype(int)

    # initialize value to return
    target_point_final = None

    # initialize new target param tuple array
    new_feature_params = []

    # if use_tracking, use the LK-tracked point as the final prediction and
    # update supporter parameters
    if use_tracking:

        target_point_final = predicted_target_point

        # update supporter feature parameters
        for i in range(len(feature_points)):
            curr_feature_point = feature_points[i]
            curr_feature_point = np.round(curr_feature_point).astype(int)
            # displacement vector between the current feature and target point
            curr_displacement = target_point_final - curr_feature_point
            # previous average for displacement
            prev_displacement_average = feature_params[i][0]
            # update displacement average using exponential forgetting
            # principle
            new_displacement_average = alpha * prev_displacement_average + (
                1 - alpha) * curr_displacement
            displacement_mean_diff = curr_displacement - new_displacement_average
            # compute current covariance matrix
            curr_covariance_matrix = displacement_mean_diff.reshape(
                2, 1) @ displacement_mean_diff.reshape(1, 2)
            # update covariance matrix average using exponential forgetting
            # principle
            prev_covariance_matrix = feature_params[i][1]
            new_covariance_matrix = alpha * prev_covariance_matrix + (
                1 - alpha) * curr_covariance_matrix

            new_feature_params.append(
                (new_displacement_average, new_covariance_matrix))

    # otherwise, track point as a weighted average of mean supporter point
    # displacements, where the weights are determined based on the covariance
    # matrix associated with each supporter point (the higher the determinant
    # of the covariance matrix, the lower the weight; weights are also affected
    # by the amount of movement of the supporter points between consecutive
    # frames)
    else:

        # initialize intermediate values used for weighted average calculation;
        # numerator holds the unnormalized final point calculation, and
        # denominator contains the normalization constant to ensure weights add
        # to one
        numerator = 0
        denominator = 0

        for i in range(len(feature_points)):
            feature_point = feature_points[i]
            prev_feature_point = prev_feature_points[i]
            displacement_norm = np.linalg.norm(feature_point -
                                               prev_feature_point)
            # determine weight to assign to point (function of displacement)
            displacement_weight = weight_function(run_params, displacement_norm)
            covariance = feature_params[i][1]
            displacement = feature_params[i][0]

            numerator += (
                displacement_weight *
                (displacement + feature_point)) / np.linalg.det(covariance)
            denominator += displacement_weight / np.linalg.det(covariance)

        # return weighted average
        target_point_final = numerator / denominator

    # if supporter-based tracking was used, return old feature parameters
    if new_feature_params == []:
        return target_point_final, feature_params

    # otherwise, return updated feature parameters
    else:
        return target_point_final, new_feature_params


def weight_function(run_params, displacement_norm):
    """Determine motion-dependent prediction weight of given supporter point.

    This method determines the weight to apply to each supporter point when
    using it for prediction of a target point based on the norm of its
    displacement vector. The larger the displacement, the higher weight the
    supporter point receives; the weight is a linear function of the
    displacement norm, with an offset of 1.

    Displacement-based weighting is used to give less importance to supporter
    points that are part of the "background" and have little correlation with
    the movement of the muscle fascia.

    Args:
        run_params (ParamValues): values of parameters used in tracking,
            including scalar alpha
        displacement_norm (float): L2 norm of relevant supporter point's
            displacement vector

    Returns:
        float weighting applied to supporter point when tracking target point
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
