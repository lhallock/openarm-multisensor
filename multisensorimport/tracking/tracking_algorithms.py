#!/usr/bin/env python3
"""Methods implementing LK, FRLK, BFLK, and SBLK tracking algorithms.

This module contains functions to execute Lucas-Kanade (LK), feature-refined
Lucas-Kanade (FRLK), bilaterally-filtered Lucas-Kanade (BFLK), and
supporter-based Lucas-Kanade (SBLK) tracking algorithms, including processing
and filtering of ultrasound images, extraction of contour points, and tracking
of these points and their properties through series of scans.
"""
import csv
import os
import time

import cv2
import numpy as np
import scipy

from multisensorimport.tracking import supporters_utils as supporters_utils
from multisensorimport.tracking.image_proc_utils import *
from multisensorimport.tracking.point_proc_utils import *


def track_LK(run_params,
             seg_filedir,
             filedir,
             pts,
             lk_params,
             viz=True,
             filter_type=0,
             filtered_LK_run=False):
    """Unmodified Lucas-Kanade point tracking w/ optional feature refinement.

    This method implements unmodified, iterative tracking of a list of
    pre-determined keypoints in a sequence of images via Lucas-Kanade optical
    flow tracking (LK), with the option to filter points based on tracking
    quality and track only the best ones (FRLK), returning time series ground
    truth and tracking error data and including (optional) visualization. This
    method also supports periodic re-setting of tracking to ground-truth
    contour values for drift evaluation.

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking
        seg_fildir (str): path to directory of ground truth (hand-segmented)
            contour images
        filedir (str): path to directory of raw (ultrasound) images
        pts (numpy.ndarray): array of points to be tracked
        lk_params (dict): dictionary of tracking parameters for use by OpenCV's
            Lucas-Kanade tracking method
        viz (bool): whether tracking video should be displayed
        filter_type (int): number specifying type of image filter to apply to
            frames before executing tracking. Filter methods located in image_proc_utils.py
        filtered_LK_run (bool): whether contour points should be filtered based
            on Shi-Tomasi corner score (for FRLK run)

    Returns:
        TODO type predicted contour area time series
        TODO type ground truth contour area time series
        TODO type ground truth thickness time series
        TODO type ground truth thickness/aspect ratio time series
        TODO type predicted thickness time series
        TODO type predicted thickness/aspect ratio time series
        TODO type normalized intersection-over-union (IoU) error time series
            (TODO: sharpen)
        TODO type IoU accuracy time series (TODO: sharpen)
        TODO: this return order seems weird
    """
    # obtain image filter function
    # TODO: filter is a built in python method/keyword, consider renaming
    filter = get_filter_from_num(filter_type)

    # keep track of contour areas that are being tracked
    predicted_contour_areas = []

    # keep track of ground truth contour areas
    ground_truth_contour_areas = []

    # keep track of ground truth thickness
    ground_truth_thickness = []

    # keep track of ground truth thickness ratio (x to y)
    ground_truth_thickness_ratio = []

    # keep track of tracked thickness
    predicted_thickness = []

    # keep track of tracked thickness ratio (x to y)
    predicted_thickness_ratio = []

    # keep track of IoU accuracy over time
    iou_accuracy_series = []

    # add first contour area
    predicted_contour_areas.append(cv2.contourArea(pts))
    ground_truth_contour_areas.append(cv2.contourArea(pts))

    # add first thickness and thickness/aspect ratio
    first_thickness_x, first_thickness_y = thickness(
        supporters_utils.format_supporters(pts))
    ground_truth_thickness.append(first_thickness_x)
    ground_truth_thickness_ratio.append(first_thickness_x / first_thickness_y)

    # get the filenames for the images
    image_filenames = os.listdir(filedir)
    segmented_filenames = os.listdir(seg_filedir)

    # obtain correct frames to track, and sort them into proper order
    filtered_image_filenames = []
    filtered_segmented_filenames = []

    # obtain the right image files and sort
    for image_filename in image_filenames:
        if (image_filename.endswith('.pgm')):
            filtered_image_filenames.append(image_filename)
    for segmented_filename in segmented_filenames:
        if (segmented_filename.endswith('.pgm')):
            filtered_segmented_filenames.append(segmented_filename)

    # sort by the image number
    sorted_image_filenames = sorted(filtered_image_filenames,
                                    key=lambda s: int(s[0:len(s) - 4]))
    sorted_segmented_filenames = sorted(filtered_segmented_filenames,
                                        key=lambda s: int(s[0:len(s) - 4]))

    predicted_thickness.append(first_thickness_x)
    predicted_thickness_ratio.append(first_thickness_x / first_thickness_y)

    iou_accuracy_series.append(1)

    # track and display specified points through images
    first_loop = True

    frame_num = 0

    # cumulative intersection-over-union (IoU) error (sum over all frames)
    cumulative_iou_error = 0

    for num in range(len(sorted_image_filenames)):
        frame_num += 1
        image_filename = sorted_image_filenames[num]
        segmented_filename = sorted_segmented_filenames[num]

        if image_filename.endswith('.pgm') and segmented_filename.endswith(
                '.pgm'):

            # make sure segmented and image filenames are the same
            assert (segmented_filename == image_filename)

            # filepath to the image
            filepath = filedir + image_filename

            # if it's the first image, we already have the contour area
            if first_loop:

                old_frame = cv2.imread(filepath, -1)

                # apply filters to frame
                old_frame_filtered = filter(old_frame, run_params)

                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)
                frame_filtered = filter(frame, run_params)

                # obtain key frame for re-initializing points and/or IoU
                # computation
                key_frame_path = seg_filedir + segmented_filename

                # if resetting the tracked contour to a ground truth frame
                if frame_num % run_params.reset_frequency == 0:
                    seg_contour = extract_contour_pts_pgm(key_frame_path)

                    # if tracking via FRLK, filter points and order them
                    # counter-clockwise
                    if filtered_LK_run:
                        filtered_contour, indeces = filter_points(
                            run_params, run_params.block_size, seg_contour, 0,
                            frame, run_params.point_frac)
                        filtered_contour = order_points(filtered_contour,
                                                        indeces, np.array([]),
                                                        np.array([]))
                        tracked_contour = filtered_contour.copy()
                    else:
                        tracked_contour = seg_contour.copy()

                # if not resetting, use tracking
                else:
                    tracked_contour, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_filtered, frame_filtered, pts, None,
                        **lk_params)

                # obtain ground truth contour for current frame
                segmented_contour = extract_contour_pts_pgm(key_frame_path)

                # draw the tracked contour
                # TODO: shouldn't this be encapsulated in "if viz" logic?
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB).copy()
                for i in range(len(tracked_contour)):
                    x, y = tracked_contour[i].ravel()
                    cv2.circle(frame_color, (x, y), 3, (0, 0, 255), -1)

                # update for next iteration
                old_frame_filtered = frame_filtered.copy()
                pts = tracked_contour.copy()

                # add ground truth and tracked thickness
                segmented_thickness_x, segmented_thickness_y = thickness(
                    supporters_utils.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(
                    supporters_utils.format_supporters(tracked_contour))

                ground_truth_thickness.append(segmented_thickness_x)

                # add ground truth and tracked aspect ratio
                if segmented_thickness_x == 0 or segmented_thickness_y == 0:
                    ground_truth_thickness_ratio.append(0)
                else:
                    ground_truth_thickness_ratio.append(segmented_thickness_x /
                                                        segmented_thickness_y)

                predicted_thickness.append(predicted_thickness_x)
                predicted_thickness_ratio.append(predicted_thickness_x /
                                                 predicted_thickness_y)

                # add ground truth and tracked contour area
                predicted_contour_areas.append(cv2.contourArea(tracked_contour))
                ground_truth_contour_areas.append(
                    cv2.contourArea(segmented_contour))

                # calculate IoU accuracy:
                # initialize matrices of zeros corresponding to image area
                mat_predicted = np.zeros(
                    cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(
                    cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                # fill matrices with nonzero numbers inside contour area
                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)],
                             255)

                intersection = np.sum(
                    np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                iou_error = intersection / union

                cumulative_iou_error += iou_error

                iou_accuracy_series.append(iou_error)

                # visualize if specified
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27:  # stop on escape key
                        break
                    time.sleep(0.01)

    # divide cumulative error by # frames to get average error per frame
    normalized_iou_error = cumulative_iou_error / frame_num

    return (predicted_contour_areas, ground_truth_contour_areas,
            ground_truth_thickness, ground_truth_thickness_ratio,
            predicted_thickness, predicted_thickness_ratio,
            normalized_iou_error, iou_accuracy_series)


def track_BFLK(run_params,
               seg_filedir,
               filedir,
               fine_pts,
               fine_pts_inds,
               course_pts,
               course_pts_inds,
               lk_params,
               viz=True):
    """Bilaterally-filtered Lucas-Kanade point tracking.

    This method implements bilaterally-filtered Lucas-Kanade (BFLK) optical
    flow tracking of a list of pre-determined keypoints in a series of images,
    returning time series ground truth and tracking error data and including
    (optional) visualization. This method also supports periodic re-setting of
    tracking to ground-truth contour values for drift evaluation and can be
    used with other non-bilateral image filters.

    TODO: It actually looks like this one doesn't support periodic re-setting,
    even though track_LK and track_SBLK do. Is there a reason for that?

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking
        seg_fildir (str): path to directory of ground truth (hand-segmented)
            contour images
        filedir (str): path to directory of raw (ultrasound) images
        fine_pts (numpy.ndarray): array of points to be tracked using more
            aggressive bilateral filter
        fine_pts_inds (numpy.ndarray): array of indices of fine_pts in the
            overall contour; used for ordering the contour and visualizing
        course_pts (numpy.ndarray): array of points to be tracked using less
            aggressive bilateral filter
        course_pts_inds (numpy.ndarray): array of indices of fine_pts in the
            overall contours; used for ordering the contour and visualizing
        lk_params (dict): dictionary of tracking parameters for use by OpenCV's
            Lucas-Kanade tracking method
        viz (bool): whether tracking video should be displayed

    Returns:
        TODO type predicted contour area time series
        TODO type ground truth contour area time series
        TODO type ground truth thickness time series
        TODO type ground truth thickness/aspect ratio time series
        TODO type predicted thickness time series
        TODO type predicted thickness/aspect ratio time series
        TODO type normalized intersection-over-union (IoU) error time series
            (TODO: sharpen)
        TODO type IoU accuracy time series (TODO: sharpen)
        TODO: this return order seems weird
    """
    # set filters (coarse_filter is less aggressive, fine_filter is more
    # aggressive)
    course_filter = course_bilateral_filter
    fine_filter = fine_bilateral_filter

    # keep track of contour areas that are being tracked
    predicted_contour_areas = []

    # keep track of ground truth contour areas
    ground_truth_contour_areas = []

    # keep track of ground truth thickness
    ground_truth_thickness = []

    # keep track of ground truth thickness ratio (x to y)
    ground_truth_thickness_ratio = []

    # keep track of tracked thickness
    predicted_thickness = []

    # keep track of tracked thickness ratio (x to y)
    predicted_thickness_ratio = []

    # keep track of IoU accuracy over time
    iou_accuracy_series = []
    iou_accuracy_series.append(1)

    # combine points to form contour
    tracked_contour = order_points(fine_pts, fine_pts_inds, course_pts,
                                   course_pts_inds)

    # add first contour area
    predicted_contour_areas.append(cv2.contourArea(tracked_contour))
    ground_truth_contour_areas.append(cv2.contourArea(tracked_contour))

    # add first thickness and thickness/aspect ratio
    first_thickness_x, first_thickness_y = thickness(
        supporters_utils.format_supporters(tracked_contour))
    ground_truth_thickness.append(first_thickness_x)
    ground_truth_thickness_ratio.append(first_thickness_x / first_thickness_y)

    predicted_thickness.append(first_thickness_x)
    predicted_thickness_ratio.append(first_thickness_x / first_thickness_y)

    # get the filenames for the images
    image_filenames = os.listdir(filedir)
    segmented_filenames = os.listdir(seg_filedir)

    # obtain correct frames to track, and sort them into proper order
    filtered_image_filenames = []
    filtered_segmented_filenames = []

    # obtain the right image files and sort
    for image_filename in image_filenames:
        if (image_filename.endswith('.pgm')):
            filtered_image_filenames.append(image_filename)
    for segmented_filename in segmented_filenames:
        if (segmented_filename.endswith('.pgm')):
            filtered_segmented_filenames.append(segmented_filename)

    # sort by the image number
    sorted_image_filenames = sorted(filtered_image_filenames,
                                    key=lambda s: int(s[0:len(s) - 4]))
    sorted_segmented_filenames = sorted(filtered_segmented_filenames,
                                        key=lambda s: int(s[0:len(s) - 4]))

    # track and display specified points through images
    first_loop = True

    frame_num = 0

    # cumulative intersection-over-union (IoU) error (sum over all frames)
    cumulative_iou_error = 0

    for num in range(len(sorted_image_filenames)):
        frame_num += 1
        image_filename = sorted_image_filenames[num]
        segmented_filename = sorted_segmented_filenames[num]

        if image_filename.endswith('.pgm') and segmented_filename.endswith(
                '.pgm'):

            # make sure segmented and image filenames are the same
            assert (segmented_filename == image_filename)

            # filepath to the image
            filepath = filedir + image_filename

            # if it's the first image, we already have the contour area
            if first_loop:

                old_frame = cv2.imread(filepath, -1)

                # apply filters to frame
                old_frame_course_filtered = course_filter(old_frame, run_params)
                old_frame_fine_filtered = fine_filter(old_frame, run_params)

                first_loop = False

                # TODO: this frame color and the viz logic below aren't in
                # track_LK or track_SBLK, make consistent
                old_frame_color = cv2.cvtColor(old_frame,
                                               cv2.COLOR_GRAY2RGB).copy()

                # visualize if specified
                if viz:
                    cv2.imshow('Frame', old_frame_color)
                    key = cv2.waitKey(1)
                    if key == 27:  # stop on escape key
                        break
                    time.sleep(0.01)

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)

                # apply filters to frames
                frame_course_filtered = course_filter(frame, run_params)
                frame_fine_filtered = fine_filter(frame, run_params)

                # obtain key frame for re-initializing points and/or IoU
                # computation
                key_frame_path = seg_filedir + segmented_filename

                # find tracked locations of points (both fine- and
                # coarse-filtered) via Lucas-Kanade and update for next
                # iteration
                fine_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame_fine_filtered, frame_fine_filtered, fine_pts,
                    None, **lk_params)
                course_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame_course_filtered, frame_course_filtered,
                    course_pts, None, **lk_params)

                # combine fine- and coarse-filtered points into full contour in
                # proper counter-clockwise order
                tracked_contour = order_points(fine_pts, fine_pts_inds,
                                               course_pts, course_pts_inds)

                # obtain ground truth contour for current frame
                segmented_contour = extract_contour_pts_pgm(key_frame_path)

                # draw the tracked contour
                # TODO: shouldn't this be encapsulated in "if viz" logic?
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                for i in range(len(fine_pts)):
                    x, y = fine_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 3, (0, 0, 255), -1)
                for i in range(len(course_pts)):
                    x, y = course_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 3, (0, 255, 255), -1)

                # add ground truth and tracked thickness and aspect ratio
                segmented_thickness_x, segmented_thickness_y = thickness(
                    supporters_utils.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(
                    supporters_utils.format_supporters(tracked_contour))

                ground_truth_thickness.append(segmented_thickness_x)

                if segmented_thickness_x == 0 or segmented_thickness_y == 0:
                    ground_truth_thickness_ratio.append(0)
                else:
                    ground_truth_thickness_ratio.append(segmented_thickness_x /
                                                        segmented_thickness_y)

                predicted_thickness.append(predicted_thickness_x)
                predicted_thickness_ratio.append(predicted_thickness_x /
                                                 predicted_thickness_y)

                # add ground truth and tracked contour area
                predicted_contour_areas.append(cv2.contourArea(tracked_contour))
                ground_truth_contour_areas.append(
                    cv2.contourArea(segmented_contour))

                # calculate IoU accuracy:
                # initialize matrices of zeros corresponding to image area
                mat_predicted = np.zeros(
                    cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(
                    cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                # fill matrices with nonzero numbers inside contour area
                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)],
                             255)

                intersection = np.sum(
                    np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                iou_error = intersection / union
                cumulative_iou_error += iou_error
                iou_accuracy_series.append(iou_error)

                # update frames for next iteration
                old_frame_fine_filtered = frame_fine_filtered.copy()
                old_frame_course_filtered = frame_course_filtered

                # visualize if specified
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27:  # stop on escape key
                        break
                    time.sleep(0.01)

    # divide cumulative error by # frames to get average error per frame
    normalized_iou_error = cumulative_iou_error / frame_num

    return (predicted_contour_areas, ground_truth_contour_areas,
            ground_truth_thickness, ground_truth_thickness_ratio,
            predicted_thickness, predicted_thickness_ratio,
            normalized_iou_error, iou_accuracy_series)


def track_SBLK(run_params,
               seg_filedir,
               filedir,
               fine_pts,
               fine_pts_inds,
               course_pts,
               course_pts_inds,
               supporter_pts,
               supporter_params,
               lk_params,
               reset_supporters,
               feature_params,
               viz=False,
               fine_filter_type=0,
               course_filter_type=0):
    """Supporter-based Lucas-Kanade point tracking.

    This method implements supporter-based Lucas-Kanade (SBLK) optical flow
    tracking of a list of pre-determined keypoints in a series of images,
    returning time series ground truth and tracking error data and including
    (optional) visualization. This method also supports periodic re-setting of
    tracking to ground-truth contour values for drift evaluation.

    Args:
        run_params (ParamValues): class containing values of parameters used in
            tracking
        seg_fildir (str): path to directory of ground truth (hand-segmented)
            contour images
        filedir (str): path to directory of raw (ultrasound) images
        fine_pts (numpy.ndarray): array of points to be tracked using more
            aggressive bilateral filter
        fine_pts_inds (numpy.ndarray): array of indices of fine_pts in the
            overall contour; used for ordering the contour and visualizing
        course_pts (numpy.ndarray): array of points to be tracked using less
            aggressive bilateral filter
        course_pts_inds (numpy.ndarray): array of indices of fine_pts in the
            overall contours; used for ordering the contour and visualizing
        supporter_pts (TODO type): TODO description
        supporter_params (TODO type): TODO description
        lk_params (dict): dictionary of tracking parameters for use by OpenCV's
            Lucas-Kanade tracking method
        reset_supporters (TODO type): TODO description
        feature_params (TODO type): TODO description
        viz (bool): whether tracking video should be displayed
        fine_filter_type (int): TODO description
        course_filter_type (int): TODO description

    Returns:
        TODO type predicted contour area time series
        TODO type ground truth contour area time series
        TODO type ground truth thickness time series
        TODO type ground truth thickness/aspect ratio time series
        TODO type predicted thickness time series
        TODO type predicted thickness/aspect ratio time series
        TODO type normalized intersection-over-union (IoU) error time series
            (TODO: sharpen)
        TODO type IoU accuracy time series (TODO: sharpen)
        TODO: this return order seems weird
    """
    # combine coarse and fine points, maintaining clockwise ordering so OpenCV
    # can interpret contours
    # TODO: isn't it counter-clockwise? that's what it says in other places
    pts = order_points(fine_pts, fine_pts_inds, course_pts, course_pts_inds)

    # keep track of ground truth contour areas
    ground_truth_contour_areas = []

    # keep track of ground truth thickness
    ground_truth_thickness = []

    # keep track of ground truth thickness ratio (x to y)
    ground_truth_thickness_ratio = []

    # keep track of contour areas that are being tracked
    predicted_contour_areas = []

    # keep track of tracked thickness
    predicted_thickness = []

    # keep track of tracked thickness ratio (x to y)
    predicted_thickness_ratio = []

    # keep track of IoU accuracy over time
    iou_accuracy_series = []
    iou_accuracy_series.append(1)

    # add first contour area
    predicted_contour_areas.append(cv2.contourArea(pts))
    ground_truth_contour_areas.append(cv2.contourArea(pts))

    # add first thickness and thickness/aspect ratio
    first_thickness_x, first_thickness_y = thickness(
        supporters_utils.format_supporters(pts))
    ground_truth_thickness.append(first_thickness_x)
    ground_truth_thickness_ratio.append(first_thickness_x / first_thickness_y)

    predicted_thickness.append(first_thickness_x)
    predicted_thickness_ratio.append(first_thickness_x / first_thickness_y)

    # track and display specified points through images
    first_loop = True
    frame_num = 0

    # cumulative intersection-over-union (IoU) error (sum over all frames)
    cumulative_iou_error = 0

    # number of frames to use for updating supporters model (0 for 1-shot
    # learning)
    num_training_frames = 0

    # get the filenames for the images
    image_filenames = os.listdir(filedir)
    segmented_filenames = os.listdir(seg_filedir)

    # obtain correct frames to track, and sort them into proper order
    filtered_image_filenames = []
    filtered_segmented_filenames = []

    # obtain the right image files and sort
    for image_filename in image_filenames:
        if (image_filename.endswith('.pgm')):
            filtered_image_filenames.append(image_filename)
    for segmented_filename in segmented_filenames:
        if (segmented_filename.endswith('.pgm')):
            filtered_segmented_filenames.append(segmented_filename)

    # sort by the image number
    sorted_image_filenames = sorted(filtered_image_filenames,
                                    key=lambda s: int(s[0:len(s) - 4]))
    sorted_segmented_filenames = sorted(filtered_segmented_filenames,
                                        key=lambda s: int(s[0:len(s) - 4]))

    # create OpenCV window (if visualization is desired)
    # TODO: this is missing from track_LK and track_BFLK, make consistent
    if viz:
        cv2.namedWindow('Frame')

    # set filters (coarse_filter is less aggressive, fine_filter is more
    # aggressive)
    # TODO: these are defined much earlier in track_BFLK, make consistent
    course_filter = get_filter_from_num(course_filter_type)
    fine_filter = get_filter_from_num(fine_filter_type)

    # how often we reset contour to a pre-segmented frame (set to super high if
    # no reset)
    # TODO: you just use this directly in track_LK, be consistent
    reset_freq = run_params.reset_frequency

    # track and display specified points through images
    first_loop = True
    frame_num = 0

    # cumulative intersection over union error (sum over all frames)
    # TODO I think you can delete this line, it's a repeat of 607 above, but
    # check to make sure
    cumulative_iou_error = 0

    # number of frames to use for updating supporters model (0 for 1-shot
    # learning)
    # TODO I think you can delete this line, it's a repeat of 610 above, but
    # check to make sure
    num_training_frames = 0

    for num in range(len(sorted_image_filenames)):
        image_filename = sorted_image_filenames[num]
        segmented_filename = sorted_segmented_filenames[num]

        if image_filename.endswith('.pgm') and segmented_filename.endswith(
                '.pgm'):

            # make sure segmented and image filenames are the same
            assert (segmented_filename == image_filename)

            # filepath to the image
            filepath = filedir + image_filename

            # if it's the first image, we already have the contour area
            if first_loop:

                old_frame = cv2.imread(filepath, -1)

                # apply filters to frame
                old_frame_course = course_filter(old_frame, run_params)
                old_frame_fine = fine_filter(old_frame, run_params)

                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)

                # obtain key frame for re-initializing points and/or IoU
                # computation
                key_frame_path = seg_filedir + segmented_filename

                # apply filters to frame
                frame_course = course_filter(frame, run_params)
                frame_fine = fine_filter(frame, run_params)
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # reset tracked contour to ground truth contour
                if frame_num % reset_freq == 0:

                    if reset_supporters:
                        # reset all points to contour and re-initialize a new
                        # set of supporters based on good corner features
                        (course_pts, course_pts_inds, fine_pts, fine_pts_inds,
                         supporter_pts,
                         supporter_params) = supporters_utils.initialize_supporters(
                             run_params, filedir, key_frame_path, frame,
                             feature_params, lk_params, 2)
                    else:
                        # reset tracking points to contour, but do not set new
                        # supporter points
                        (course_pts, course_pts_inds, fine_pts, fine_pts_inds,
                         _, _) = supporters_utils.initialize_supporters(run_params, filedir,
                                                       key_frame_path, frame,
                                                       feature_params,
                                                       lk_params, 2)
                        # re-initialize parameters for supporters
                        supporter_params = []
                        for i in range(len(course_pts)):
                            point = course_pts[i][0]
                            (_, supporter_param
                            ) = supporters_utils.initialize_supporters_for_point(
                                supporter_pts, point,
                                run_params.supporter_variance)
                            supporter_params.append(supporter_param)
                else:
                    # calculate new point locations for fine_points using frame
                    # filtered by fine filter
                    new_fine_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_fine, frame_fine, fine_pts, None, **lk_params)

                    # calculate new point locations for course_points using
                    # frame filtered by coarse filter (predictions, might be
                    # updated by supporters)
                    predicted_course_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_course, frame_course, course_pts, None,
                        **lk_params)

                    # calculate new supporter locations in coarsely filtered
                    # frame
                    new_supporter_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_course, frame_course, supporter_pts, None,
                        **lk_params)

                    # re-format predicted points TODO: sharpen
                    predicted_course_pts = supporters_utils.format_supporters(
                        predicted_course_pts)

                    # initialize new feature parameters
                    updated_feature_params = []
                    new_course_pts = []

                    # whether to trust LK tracking or not
                    use_tracking = ((frame_num % reset_freq) <=
                                    num_training_frames)

                    # get supporter predictions (returns LK predictions if
                    # use_tracking is True)
                    for i in range(len(predicted_course_pts)):
                        predicted_point = predicted_course_pts[i]
                        param_list = supporter_params[i]

                        # pass in both supporter_pts (old values) and
                        # new_supporter_pts (new values) so the displacement
                        # can be calculated TODO: what lines does this comment
                        # cover?

                        # obtain point predictions and updated parameters for
                        # target point
                        (point_location,
                         new_params) = supporters_utils.apply_supporters_model(
                             run_params, predicted_point, supporter_pts,
                             new_supporter_pts, param_list, use_tracking,
                             run_params.update_rate)
                        updated_feature_params.append(new_params)
                        new_course_pts.append(
                            np.array([[point_location[0], point_location[1]]],
                                     dtype=np.float32))

                    new_course_pts = np.array(new_course_pts)

                    # update point locations for next iteration
                    fine_pts = new_fine_pts
                    course_pts = new_course_pts
                    supporter_pts = new_supporter_pts

                    # update supporter params
                    supporter_params = updated_feature_params

                    # save old frame for optical flow calculation in next
                    # iteration
                    old_frame_course = frame_course.copy()
                    old_frame_fine = frame_fine.copy()

                # draw the tracked contour and supporter points
                # TODO: appears to be missing frame color logic from track_LK
                # and track_BFLK
                for i in range(len(fine_pts)):
                    x, y = fine_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 4, (0, 0, 255), -1)
                for i in range(len(course_pts)):
                    x, y = course_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 4, (0, 255, 0), -1)
                for i in range(len(supporter_pts)):
                    x, y = supporter_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 4, (0, 255, 0), 1)

                # combine fine- and coarse-filtered points into full contour in
                # proper counter-clockwise order
                tracked_contour = order_points(fine_pts, fine_pts_inds,
                                               course_pts, course_pts_inds)

                # set y coordinate of first and last point to 0 to fix downward
                # boundary drift, if specified
                # TODO: why is this functionality not supported in track_LK or
                # track_BFLK? if this is logical, add a short explanation
                # (maybe in docstring), if not logical, add support in other
                # functions for consistency
                if run_params.fix_top:
                    first_contour_point = tracked_contour[0]
                    first_contour_point[0][1] = 0

                    last_contour_point = tracked_contour[len(tracked_contour) -
                                                         1]
                    last_contour_point[0][1] = 0

                # obtain ground truth contour for current frame
                segmented_contour = extract_contour_pts_pgm(key_frame_path)

                # visualize
                # TODO: bring this visualization logic in line with track_LK
                # and track_BFLK
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27:  # stop on escape key
                        break
                    time.sleep(0.01)

                # add ground truth and tracked thickness
                segmented_thickness_x, segmented_thickness_y = thickness(
                    supporters_utils.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(
                    supporters_utils.format_supporters(tracked_contour))

                ground_truth_thickness.append(segmented_thickness_x)

                # add ground truth and tracked thickness/aspect ratio
                if segmented_thickness_x == 0 or segmented_thickness_y == 0:
                    ground_truth_thickness_ratio.append(0)
                else:
                    ground_truth_thickness_ratio.append(segmented_thickness_x /
                                                        segmented_thickness_y)

                predicted_thickness.append(predicted_thickness_x)
                predicted_thickness_ratio.append(predicted_thickness_x /
                                                 predicted_thickness_y)

                # add ground truth and tracked contour area
                predicted_contour_areas.append(cv2.contourArea(tracked_contour))
                ground_truth_contour_areas.append(
                    cv2.contourArea(segmented_contour))

                # calculate IoU accuracy:
                # initialize matrices of zeros corresponding to image area
                mat_predicted = np.zeros(
                    cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(
                    cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                # fill matrices with nonzero numbers inside contour area
                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)],
                             255)

                intersection = np.sum(
                    np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                iou_error = intersection / union
                cumulative_iou_error += iou_error

                iou_accuracy_series.append(iou_error)

        frame_num += 1

    # TODO: why isn't this in track_LK and track_BFLK? make consistent
    if viz:
        cv2.destroyAllWindows()

    # divide cumulative error by # frames to get average error per frame
    normalized_iou_error = cumulative_iou_error / frame_num

    return (predicted_contour_areas, ground_truth_contour_areas,
            ground_truth_thickness, ground_truth_thickness_ratio,
            predicted_thickness, predicted_thickness_ratio,
            normalized_iou_error, iou_accuracy_series)
