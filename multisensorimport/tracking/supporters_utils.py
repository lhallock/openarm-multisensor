"""
Methods to initialize supporter points, and use supporter points to infer contour points in SBLK.
"""

import numpy as np
import scipy
from scipy import stats

from multisensorimport.tracking.image_proc_utils import *
from multisensorimport.tracking.point_proc_utils import *


def initialize_supporters(run_params, READ_PATH, keyframe_path, init_img,
                          feature_params, lk_params, which_contour):
    """
    Separates contour points into those to be tracked via lucas kucas tracking, and those to be tracked via supporters, and also determine good supporter points and initialize their parameters.

    Args:
        run_params: instance of ParamValues class, contains values of parameters used in tracking
        READ_PATH: path to raw ultrasound frames
        key_frame_path: path to ground truth hand segmented frames
        init_img: first frame in the video sequence
        feature_params: parameters to find good features to track
        lk_params: parameters for lucas kanade tracking
        which_contour: integer determining whether image containing contour is a png (1) or a pgm (0)

    Returns:
        numpy array of points to be tracked via supporters and their corresponding indeces in the contour, numpy array of points to be tracked via Lucas Kanade and their corresponding indeces in the contour, list of supporter point locations, list of the parameters for each supporter points
    """

    # extract contour
    if which_contour == 1:
        pts = extract_contour_pts_png(keyframe_path)
    else:
        pts = extract_contour_pts_pgm(keyframe_path)

    mean_x_pts = 0
    mean_y_pts = 0

    # find mean of the coordinates of the contour points
    for contour_point in pts:
        x = contour_point[0][0]
        y = contour_point[0][1]
        mean_x_pts += x
        mean_y_pts += y

    mean_x_pts = mean_x_pts / len(pts)
    mean_y_pts = mean_y_pts / len(pts)

    # filter to be used (1: median filter, 2: bilateral filter, 3: course bilateral, 4: anisotropicDiffuse anything else no filter )
    fineFilterNum = 2
    courseFilterNum = 3

    course_filter = get_filter_from_num(courseFilterNum)
    filtered_init_img = course_filter(init_img, run_params)

    # remove points that have low corner scores (Shi Tomasi Corner scoring): these points will be kept for LK tracking
    lucas_kanade_points, lucas_kanade_points_indeces = filter_points(
        run_params,
        7,
        pts,
        fineFilterNum,
        init_img,
        run_params.fine_threshold,
        keep_bottom=True)
    # add the first point to LK tracking (top left)
    lucas_kanade_points = np.append(lucas_kanade_points,
                                    np.array([pts[0]]),
                                    axis=0)
    lucas_kanade_points_indeces = np.append(lucas_kanade_points_indeces, 0)

    # obtain points which need supporters to be tracked
    supporter_tracked_points = pts.copy()
    supporter_tracked_points_indeces = np.arange(0, len(pts))

    # filter supporter tracked to be in desired region: should be in top right "quadrant" (greater than mean x, less than mean y)
    supporter_kept_indeces = set()
    for i in range(len(supporter_tracked_points)):
        supporter_tracked_point = supporter_tracked_points[i]
        add = (supporter_tracked_point[0][0] > mean_x_pts and
               supporter_tracked_point[0][1] < mean_y_pts)
        if add:
            supporter_kept_indeces.add(i)

    # only add the supporter_tracked points that we determined should be added
    supporter_tracked_to_keep = []
    supporter_tracked_to_keep_inds = []
    for index in supporter_kept_indeces:
        supporter_tracked_to_keep.append(supporter_tracked_points[index])
        supporter_tracked_to_keep_inds.append(
            supporter_tracked_points_indeces[index])

    # reset the points to get tracked using supporters
    supporter_tracked_points = np.array(supporter_tracked_to_keep)
    supporter_tracked_points_indeces = np.array(supporter_tracked_to_keep_inds)

    # find points which differ between supporter tracked and LK points: remove the LK points which match supporter points
    LK_kept_indeces = set()
    for i in range(len(lucas_kanade_points)):
        lucas_kanade_point = lucas_kanade_points[i]
        add = True
        # go through the supporter points
        for j in range(len(supporter_tracked_points)):
            supporter_point = supporter_tracked_points[j]
            # go through supporter points: if the LK point is the same as a supporter point or it is in the top right zone, do not add
            if ((np.linalg.norm(lucas_kanade_point - supporter_point) < 0.001)
                    or (lucas_kanade_point[0][0] > mean_x_pts and
                        lucas_kanade_point[0][1] < mean_y_pts)):
                add = False
        if add:
            LK_kept_indeces.add(i)

    # only add the lucas kanade points that we determined should be added
    LK_to_keep = []
    LK_to_keep_inds = []
    for index in LK_kept_indeces:
        LK_to_keep.append(lucas_kanade_points[index])
        LK_to_keep_inds.append(lucas_kanade_points_indeces[index])

    # reset the points to be tracked using supporters
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
    """
    Reformats list of given supporter points, and initializes parameters (displacement, covariance) for each supporter point, for a given target point

    supporter_points: numpy array of 1 element numpy arrays, where the 1 element is a 2-element numpy array containing supporter point locations
    target_point: numpy array containing x,y coordinates for the target point being tracked
    variance: scalar value, indicates the initial variance for each element of the displacement

    Returns: list of 2-element numpy arrays containing supporter point locations
    """

    # initialize empty lists
    supporters = []
    supporter_params = []
    for i in range(len(supporter_points)):
        # extract numpy array of the supporter location
        supporter_point = supporter_points[i][0]
        supporters.append(supporter_point)
        # initialize displacement average with initial displacement and a diagonal covariance matrix
        supporter_params.append(
            (target_point - supporter_point, variance * np.eye(2)))

    return supporters, supporter_params


def apply_supporters_model(run_params, predicted_target_point,
                           prev_feature_points, feature_points, feature_params,
                           use_tracking, alpha):
    """
    Do model learning or prediction based on learned model, based on conditions of image tracking

    run_params: instance of ParamValues class holding relevent parameters
    predicted_target_point: numpy array (2-element) of x, y coord of tracking prediction of current target point
    prev_feature_points: list of (x,y) coordinates of the feature (supporter) points in previous frame
    feature_points: list of [x,y] coordinate array of the feature (supporter) points in current frame
    feature_params: list of 2-tuples of (displacement vector average, covariance matrix aveage) and  for each feature point
    use_tracking: boolean determining whether to return the pure Lucas Kanade prediction or the supporters based prediction
    alpha: learning rate for exponential forgetting principle

    Returns: predicted location of target point, updated parameters corresponding for the supporter points
    """

    # reformat feature points for easier processing
    feature_points = format_supporters(feature_points)

    # round to integer so that the prediction lands on pixel coordinates
    predicted_target_point = np.round(predicted_target_point).astype(int)

    # initialize value to return
    target_point_final = None
    # initialize new target param tuple array
    new_feature_params = []

    # tracking is to be used (first x amount of frames, x determined a priori)
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

    # Use supporter prediction: take a weighted average of the mean displacements + supporter positions, weighted by probability of supporter and prediction
    else:
        # quantities used in calculation
        numerator = 0
        denominator = 0
        displacements = []

        for i in range(len(feature_points)):
            feature_point = feature_points[i]
            prev_feature_point = prev_feature_points[i]
            displacement_norm = np.linalg.norm(feature_point -
                                               prev_feature_point)
            # determine the weight to assign to that point, as a function of displacement
            weight = weight_function(run_params, displacement_norm)
            covariance = feature_params[i][1]
            displacement = feature_params[i][0]

            numerator += (
                weight *
                (displacement + feature_point)) / np.linalg.det(covariance)
            denominator += weight / np.linalg.det(covariance)

        # return weighted average
        target_point_final = numerator / denominator

    # if Supporters was used, return the old feature_params; else return the updated params
    if new_feature_params == []:
        return target_point_final, feature_params
    else:
        return target_point_final, new_feature_params


def weight_function(run_params, displacement_norm):
    """
    Determines the weight to apply to each supporter point, as a function of the norm of the displacement vector for that point.

    run_params: instance of ParamValues class holding relevent parameters
    displacement_norm: L2 norm of the displacement vector of the supporter point being considered

    Returns: weight to place for the supporter point being considered

    """
    alpha = run_params.displacement_weight

    return 1 + (alpha * displacement_norm)


def format_supporters(supporter_points):
    """
    Reformats list of given supporter points into a list of numpy arrays containing the supporter point locations

    Args:
        supporter_points: numpy array of 1 element numpy arrays, where the 1 element is a 2-element numpy array containing supporter point locations

    Returns: list of 2-element numpy arrays containing supporter point locations
    """
    supporters = []
    for i in range(len(supporter_points)):
        supporters.append(supporter_points[i][0])
    return supporters
