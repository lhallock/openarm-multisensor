#!/usr/bin/env python3
"""Methods implementing different tracking algorithms: LK, FRLK, BFLK, and SBLK.

This module contains algorithms to extract desired contour points for tracking, process and filter ultrasound images,  and track these points and their properties through ultrasound scans.
"""

import time
import os

import csv

import cv2
import numpy as np
import scipy

from multisensorimport.tracking import supporters_simple as supporters_simple
from multisensorimport.tracking.image_proc_utils import *
from multisensorimport.tracking.point_proc_utils import *


def track_LK(run_params, seg_filedir, filedir, pts, lk_params, viz = True, filter_type=0, filtered_LK_run=False):
    """
    Implements unmodified, iterative lucas kanade (LK), and feature refined lucas kanade (FRLK) algorithm to track a list of pre-determined keypoints in a sequence of images, and obtain values of interest.

    Args:
        run_params: instance of ParamValues class, contains values of parameters used in tracking
        seg_fildir: path to directory containing hand segmented (ground truth) images
        filedir: path to directory containing raw images
        pts: numpy array of pts to be tracked
        lk_params: dictionary of parameters to pass into OpenCV's Lucas Kanade method
        viz: boolean value specifying if the video should be displayed
        filter_type: number specifiying the type of image filter to apply on frames, before they are passed into Lucas Kanade
        filtered_LK_run: if contour points are filtered based on corner score (for FRLK)

    Returns: contour time series, thickness time series, aspect ratio time series, for both tracking and ground truth, and IoU accuracy measures

    """

    # obtain image filter function
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

    # keep track of tracked thickness ratio (x to y
    predicted_thickness_ratio = []

    # keep track of iou accuracy over time
    iou_accuracy_series = []

    # add first contour area
    predicted_contour_areas.append(cv2.contourArea(pts))
    ground_truth_contour_areas.append(cv2.contourArea(pts))

    # add first thickness
    first_thickness_x, first_thickness_y = thickness(supporters_simple.format_supporters(pts))
    ground_truth_thickness.append(first_thickness_x)
    ground_truth_thickness_ratio.append(first_thickness_x / first_thickness_y)

    image_filenames = os.listdir(filedir)
    segmented_filenames = os.listdir(seg_filedir)

    filtered_image_filenames = []
    filtered_segmented_filenames = []

    # obtain the right image files and sort
    for image_filename in image_filenames:
        if (image_filename.endswith('.pgm')):
            filtered_image_filenames.append(image_filename)
    for segmented_filename in segmented_filenames:
        if (segmented_filename.endswith('.pgm')):
            filtered_segmented_filenames.append(segmented_filename)

    sorted_image_filenames = sorted(filtered_image_filenames, key=lambda s: int(s[0:len(s)-4]))
    sorted_segmented_filenames = sorted(filtered_segmented_filenames, key=lambda s: int(s[0:len(s)-4]))

    predicted_thickness.append(first_thickness_x)
    predicted_thickness_ratio.append(first_thickness_x / first_thickness_y)

    iou_accuracy_series.append(1)


    # track and display specified points through images
    first_loop = True

    frame_num = 0

    # cumulative intersection over union error (sum over all frames)
    cumulative_iou_error = 0

    for num in range(len(sorted_image_filenames)):
        frame_num += 1
        image_filename = sorted_image_filenames[num]
        segmented_filename = sorted_segmented_filenames[num]

        if image_filename.endswith('.pgm') and segmented_filename.endswith('.pgm'):
            assert(segmented_filename == image_filename)
            filepath = filedir + image_filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                old_frame_filtered = filter(old_frame, run_params)
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)
                frame_filtered = filter(frame, run_params)

                # obtain key frame, for re-initializing points and/or intersection/union computation
                key_frame_path = seg_filedir + segmented_filename

                # if we are resetting the tracked contour to a ground truth frame
                if frame_num % run_params.reset_frequency == 0:
                    seg_contour = extract_contour_pts_pgm(key_frame_path)

                    # if tracking is done via a FRLK, filter the points and order them counter-clockwise
                    if filtered_LK_run:
                        filtered_contour, indeces = filter_points(run_params, 7, seg_contour, 0,
                                                                                frame, 0.7)
                        filtered_contour = order_points(filtered_contour, indeces, np.array([]),
                                                                      np.array([]))
                        tracked_contour = filtered_contour.copy()
                    else:
                        tracked_contour = seg_contour.copy()
                # use tracking
                else:
                    tracked_contour, status, error = cv2.calcOpticalFlowPyrLK(old_frame_filtered, frame_filtered, pts, None, **lk_params)

                # obtain ground truth contour for current frame
                segmented_contour = extract_contour_pts_pgm(key_frame_path)

                # draw the tracked contour
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB).copy()
                for i in range(len(tracked_contour)):
                    x, y = tracked_contour[i].ravel()
                    cv2.circle(frame_color, (x, y), 3, (0, 0, 255), -1)

                # update for next iteration
                old_frame_filtered = frame_filtered.copy()
                pts = tracked_contour.copy()

                # add ground truth and tracked thickness
                segmented_thickness_x, segmented_thickness_y = thickness(supporters_simple.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(supporters_simple.format_supporters(tracked_contour))

                ground_truth_thickness.append(segmented_thickness_x)

                # add ground truth and tracked aspect ratio
                if segmented_thickness_x == 0 or segmented_thickness_y == 0:
                    ground_truth_thickness_ratio.append(0)
                else:
                    ground_truth_thickness_ratio.append(segmented_thickness_x / segmented_thickness_y)

                predicted_thickness.append(predicted_thickness_x)
                predicted_thickness_ratio.append(predicted_thickness_x / predicted_thickness_y)

                # add ground truth and tracked contour area
                predicted_contour_areas.append(cv2.contourArea(tracked_contour))
                ground_truth_contour_areas.append(cv2.contourArea(segmented_contour))

                # calculate intersection over union accuracy:

                # initialize matrices of zeros
                mat_predicted = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                # fill the initialized matrices with nonzero numbers in the area of the contour
                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)], 255)

                intersection = np.sum(np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                iou_error = intersection / union

                cumulative_iou_error += iou_error

                iou_accuracy_series.append(iou_error)

                # visualize if flag is set:
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27: # stop on escape key
                        break
                    time.sleep(0.01)

    # final average iou
    normalized_iou_error = cumulative_iou_error / frame_num

    return predicted_contour_areas, ground_truth_contour_areas, ground_truth_thickness, ground_truth_thickness_ratio, predicted_thickness, predicted_thickness_ratio, normalized_iou_error, iou_accuracy_series





def track_BFLK(run_params, seg_filedir, filedir, fine_pts, fine_pts_inds, course_pts, course_pts_inds, lk_params, viz = True):
    """
    Bilaterally filtered lucas kanade (BFLK) algorithm to track a list of pre-determined keypoints in a sequence of images, and obtain values of interest. *This can also be used with other image filters*

    Args:
        run_params: instance of ParamValues class, contains values of parameters used in tracking
        seg_fildir: path to directory containing hand segmented (ground truth) images
        filedir: path to directory containing raw images
        fine_pts: numpy array of pts to be tracked using the more aggressive filter
        fine_pts_inds: numpy array containing the indeces of the fine_pts in the overall contour; used for ordering the contour and visualizing
        course_pts: numpy array of pts to be tracked using the less aggressive filter
        course_pts_inds: numpy array containing the indeces of the course_pts in the overall contour; used for ordering the contour and visualizing
        lk_params: dictionary of parameters to pass into OpenCV's Lucas Kanade method
        viz: boolean value specifying if the video should be displayed

    Returns: contour time series, thickness time series, aspect ratio time series, for both tracking and ground truth, and IoU accuracy measures

    """
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

    # keep track of tracked thickness ratio (x to y
    predicted_thickness_ratio = []

    iou_accuracy_series = []

    iou_accuracy_series.append(1)


    # combine pts to form contour
    tracked_contour = order_points(fine_pts, fine_pts_inds, course_pts, course_pts_inds)


    # add first contour area
    predicted_contour_areas.append(cv2.contourArea(tracked_contour))
    ground_truth_contour_areas.append(cv2.contourArea(tracked_contour))

    # add first aspect ratio
    first_thickness_x, first_thickness_y = thickness(supporters_simple.format_supporters(tracked_contour))
    ground_truth_thickness.append(first_thickness_x)
    ground_truth_thickness_ratio.append(first_thickness_x / first_thickness_y)

    # add first thickness
    predicted_thickness.append(first_thickness_x)
    predicted_thickness_ratio.append(first_thickness_x / first_thickness_y)

    image_filenames = os.listdir(filedir)
    segmented_filenames = os.listdir(seg_filedir)

    # obtain correct frames to track, and sort them into proper order
    filtered_image_filenames = []
    filtered_segmented_filenames = []

    for image_filename in image_filenames:
        if (image_filename.endswith('.pgm')):
            filtered_image_filenames.append(image_filename)
    for segmented_filename in segmented_filenames:
        if (segmented_filename.endswith('.pgm')):
            filtered_segmented_filenames.append(segmented_filename)

    sorted_image_filenames = sorted(filtered_image_filenames, key=lambda s: int(s[0:len(s)-4]))
    sorted_segmented_filenames = sorted(filtered_segmented_filenames, key=lambda s: int(s[0:len(s)-4]))

    # track and display specified points through images
    first_loop = True

    frame_num = 0

    # cumulative intersection over union error (sum over all frames)
    cumulative_iou_error = 0

    for num in range(len(sorted_image_filenames)):
        frame_num += 1
        image_filename = sorted_image_filenames[num]
        segmented_filename = sorted_segmented_filenames[num]

        if image_filename.endswith('.pgm') and segmented_filename.endswith('.pgm'):
            assert(segmented_filename == image_filename)
            filepath = filedir + image_filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                old_frame_course_filtered = course_filter(old_frame, run_params)
                old_frame_fine_filtered = fine_filter(old_frame, run_params)
                first_loop = False
                old_frame_color = cv2.cvtColor(old_frame, cv2.COLOR_GRAY2RGB).copy()

                # visualize if needed:
                if viz:
                    cv2.imshow('Frame', old_frame_color)
                    key = cv2.waitKey(1)
                    if key == 27:  # stop on escape key
                        break
                    time.sleep(0.01)

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)

                # apply image filters to the frames
                frame_course_filtered = course_filter(frame, run_params)
                frame_fine_filtered = fine_filter(frame, run_params)

                # obtain key frame, for re-initializing points and/or intersection/union computation
                key_frame_path = seg_filedir + segmented_filename

                # find tracked locations of the points (for both fine and course) using lucas kanade, and update for next iteration
                fine_pts, status, error = cv2.calcOpticalFlowPyrLK(old_frame_fine_filtered, frame_fine_filtered, fine_pts, None, **lk_params)
                course_pts, status, error = cv2.calcOpticalFlowPyrLK(old_frame_course_filtered, frame_course_filtered, course_pts, None, **lk_params)

                # combine fine and course points into full contour, in proper counter-clockwise order
                tracked_contour = order_points(fine_pts, fine_pts_inds, course_pts, course_pts_inds)

                # obtain ground truth contour
                segmented_contour = extract_contour_pts_pgm(key_frame_path)

                # draw contour points
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                for i in range(len(fine_pts)):
                    x, y = fine_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 3, (0, 0, 255), -1)
                for i in range(len(course_pts)):
                    x, y = course_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 3, (0, 255, 255), -1)


                # calculate values of interest (thickness, CSA, AR) for ground truth and tracking
                segmented_thickness_x, segmented_thickness_y = thickness(
                    supporters_simple.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(
                    supporters_simple.format_supporters(tracked_contour))

                ground_truth_thickness.append(segmented_thickness_x)

                if segmented_thickness_x == 0 or segmented_thickness_y == 0:
                    ground_truth_thickness_ratio.append(0)
                else:
                    ground_truth_thickness_ratio.append(segmented_thickness_x / segmented_thickness_y)

                predicted_thickness.append(predicted_thickness_x)
                predicted_thickness_ratio.append(predicted_thickness_x / predicted_thickness_y)

                # append new predicted contour area
                predicted_contour_areas.append(cv2.contourArea(tracked_contour))
                ground_truth_contour_areas.append(cv2.contourArea(segmented_contour))

                # calculate intersection over union accuracy:
                # initialize matrices of zeros
                mat_predicted = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                # fill the initialized matrices with nonzero numbers in the area of the contour
                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)], 255)

                intersection = np.sum(np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                iou_error = intersection / union
                # ("intersection over union error: ", iou_error)
                cumulative_iou_error += iou_error
                iou_accuracy_series.append(iou_error)

                # update frames for next iteration
                old_frame_fine_filtered = frame_fine_filtered.copy()
                old_frame_course_filtered = frame_course_filtered

                # visualize if flag is set:
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27:  # stop on escape key
                        break
                    time.sleep(0.01)

    # obtain averaged iou
    normalized_iou_error = cumulative_iou_error / frame_num

    return predicted_contour_areas, ground_truth_contour_areas, ground_truth_thickness, ground_truth_thickness_ratio, predicted_thickness, predicted_thickness_ratio, normalized_iou_error, iou_accuracy_series


def track_SBLK(run_params, seg_filedir, filedir, fine_pts, fine_pts_inds, course_pts, course_pts_inds, supporter_pts, supporter_params, lk_params, reset_supporters, feature_params, viz=False, fine_filter_type=0, course_filter_type=0):
    """Supporters based lucas kanade (SBLK) algorithm to track a list of pre-determined keypoints in a sequence of images, and obtain values of interest.


    Args:
        run_params: instance of ParamValues class, contains values of parameters used in tracking
        seg_fildir: path to directory containing hand segmented (ground truth) images
        filedir: path to directory containing raw images
        fine_pts: numpy array of pts to be tracked using the more aggressive filter
        fine_pts_inds: numpy array containing the indeces of the fine_pts in the overall contour; used for ordering the contour and visualizing
        course_pts: numpy array of pts to be tracked using the less aggressive filter
        course_pts_inds: numpy array containing the indeces of the course_pts in the overall contour; used for ordering the contour and visualizing
        lk_params: dictionary of parameters to pass into OpenCV's Lucas Kanade method
        viz: boolean value specifying if the video should be displayed


    Returns:
        list of contour areas of each frame
    """

    # combine course and fine points (maintaining clockwise ordering so OpenCV can interprate contours
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

    # keep track of tracked thickness ratio (x to y
    predicted_thickness_ratio = []

    iou_accuracy_series = []

    iou_accuracy_series.append(1)

    # add first contour area
    predicted_contour_areas.append(cv2.contourArea(pts))
    ground_truth_contour_areas.append(cv2.contourArea(pts))

    # add first thickness
    first_thickness_x, first_thickness_y = thickness(supporters_simple.format_supporters(pts))
    ground_truth_thickness.append(first_thickness_x)
    ground_truth_thickness_ratio.append(first_thickness_x / first_thickness_y)

    predicted_thickness.append(first_thickness_x)
    predicted_thickness_ratio.append(first_thickness_x / first_thickness_y)

    # track and display specified points through images
    first_loop = True
    frame_num = 0

    # cumulative intersection over union error (sum over all frames)
    cumulative_iou_error = 0

    num_training_frames = 0

    # get the filenames for the images
    image_filenames = os.listdir(filedir)
    segmented_filenames = os.listdir(seg_filedir)

    filtered_image_filenames = []
    filtered_segmented_filenames = []

    # keep only the pgm images
    for image_filename in image_filenames:
        if (image_filename.endswith('.pgm')):
            filtered_image_filenames.append(image_filename)
    for segmented_filename in segmented_filenames:
        if (segmented_filename.endswith('.pgm')):
            filtered_segmented_filenames.append(segmented_filename)

    # sort by the image number
    sorted_image_filenames = sorted(filtered_image_filenames, key=lambda s: int(s[0:len(s)-4]))
    sorted_segmented_filenames = sorted(filtered_segmented_filenames, key=lambda s: int(s[0:len(s)-4]))

    # create OpenCV window (if visualization is desired)
    if viz:
        cv2.namedWindow('Frame')

    # set filters (course filter is a less aggressive filter, fine_filter more aggressive)
    course_filter = get_filter_from_num(course_filter_type)
    fine_filter = get_filter_from_num(fine_filter_type)

    # how often we reset contour to a pre-segmented frame (set to super high if no reset)
    reset_freq = run_params.reset_frequency

    # track and display specified points through images
    first_loop = True
    frame_num = 0

    # cumulative intersection over union error (sum over all frames)
    cumulative_iou_error = 0

    # number of frames to use for updating supporters model (0 for 1-shot learning)
    num_training_frames = 0

    for num in range(len(sorted_image_filenames)):
        image_filename = sorted_image_filenames[num]
        segmented_filename = sorted_segmented_filenames[num]

        if image_filename.endswith('.pgm') and segmented_filename.endswith('.pgm'):
            # make sure segmented and image filenames are the same
            assert(segmented_filename == image_filename)

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

                # obtain key frame, for re-initializing points and/or iou computation
                key_frame_path = seg_filedir + segmented_filename

                # apply filters to frame
                frame_course = course_filter(frame, run_params)
                frame_fine = fine_filter(frame, run_params)
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # reset tracked contour to ground truth contour
                if frame_num % reset_freq == 0:

                    if reset_supporters:
                        # reset all points to contour, and re-initialize a new set of supporters based on good corner features
                        course_pts, course_pts_inds, fine_pts, fine_pts_inds, supporter_pts, supporter_params = initialize_supporters(run_params, filedir, key_frame_path, frame, feature_params, lk_params, 2)
                    else:
                        # reset tracking points to contour, but do not set new supporter points.
                        course_pts, course_pts_inds, fine_pts, fine_pts_inds, _, _ = initialize_supporters(run_params, filedir, key_frame_path, frame, feature_params, lk_params, 2)
                        # re-initialize parameters for supporters
                        supporter_params = []
                        for i in range(len(course_pts)):
                            point = course_pts[i][0]
                            _, run_params = supporters_simple.initialize_supporters_for_point(supporter_pts, point, 10)
                            supporter_params.append(run_params)
                else:
                    # calculate new point locations for fine_points using frame filtered by the fine filter
                    new_fine_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_fine, frame_fine, fine_pts, None, **lk_params)

                    # calculate new point locations for course_points using frame filtered by the course filter: predictions, might be updated by supporters
                    predicted_course_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_course, frame_course, course_pts, None, **lk_params)

                    # calculate new supporter locations in coursely filtered frame
                    new_supporter_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_course, frame_course, supporter_pts, None, **lk_params
                    )

                    # reformat predicted points (using a function in supporters_simple)
                    predicted_course_pts = supporters_simple.format_supporters(predicted_course_pts)

                    # initialize new params
                    updated_feature_params = []
                    new_course_pts = []

                    # whether to trust LK tracking or not
                    use_tracking = ((frame_num % reset_freq) <= num_training_frames)

                    # get supporters predictions (will return the LK predictions if use_tracking is True)
                    for i in range(len(predicted_course_pts)):
                        predicted_point = predicted_course_pts[i]
                        param_list = supporter_params[i]

                        # pass in both supporter_pts (the old values) and new_supporter_pts (old values) so that the displacement can be calculated
                        learning_rate = 0.7
                        # obtain point predictions and updated params for target point
                        point_location, new_params = supporters_simple.apply_supporters_model(run_params, predicted_point, supporter_pts, new_supporter_pts, param_list, use_tracking, learning_rate)
                        updated_feature_params.append(new_params)
                        new_course_pts.append(np.array([[point_location[0], point_location[1]]], dtype=np.float32))

                    new_course_pts = np.array(new_course_pts)


                    # update point locations for next iteration
                    fine_pts = new_fine_pts
                    course_pts = new_course_pts
                    supporter_pts = new_supporter_pts

                    # update supporter params
                    supporter_params = updated_feature_params

                    # save old frame for optical flow calculation in next iteration
                    old_frame_course = frame_course.copy()
                    old_frame_fine = frame_fine.copy()


                # draw the contour
                for i in range(len(fine_pts)):
                    x, y = fine_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 4, (0, 0, 255), -1)
                for i in range(len(course_pts)):
                    x, y = course_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 4, (0, 255, 0), -1)
                for i in range(len(supporter_pts)):
                    x, y = supporter_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 4, (0, 255, 0), 1)

                # combine and order the fine and course points into one contour
                tracked_contour = order_points(fine_pts, fine_pts_inds, course_pts, course_pts_inds)


                # set the y coordinate of first and last point to 0 to fix downward boundary drift (if this flag is set)
                if run_params.fix_top:
                    first_contour_point = tracked_contour[0]
                    first_contour_point[0][1] = 0

                    last_contour_point = tracked_contour[len(tracked_contour) - 1]
                    last_contour_point[0][1] = 0

                # contour from ground truth segmentaton
                segmented_contour = extract_contour_pts_pgm(key_frame_path)

                # visualize
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27: # stop on escape key
                        break
                    time.sleep(0.01)

                # calculate thickness for ground truth and tracking
                segmented_thickness_x, segmented_thickness_y = thickness(supporters_simple.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(supporters_simple.format_supporters(tracked_contour))

                ground_truth_thickness.append(segmented_thickness_x)

                # calculate AR for ground truth and tracking
                if segmented_thickness_x == 0 or segmented_thickness_y == 0:
                    ground_truth_thickness_ratio.append(0)
                else:
                    ground_truth_thickness_ratio.append(segmented_thickness_x / segmented_thickness_y)

                predicted_thickness.append(predicted_thickness_x)
                predicted_thickness_ratio.append(predicted_thickness_x / predicted_thickness_y)

                # calculate CSA for ground truth and tracking
                predicted_contour_areas.append(cv2.contourArea(tracked_contour))
                ground_truth_contour_areas.append(cv2.contourArea(segmented_contour))

                # calculate intersection over union

                # initialize matrices of zeros
                mat_predicted = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                # fill the initialized matrices with nonzero numbers in the area of the contour
                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)], 255)

                intersection = np.sum(np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                iou_error = intersection / union
                cumulative_iou_error += iou_error

                iou_accuracy_series.append(iou_error)


        frame_num += 1

    if viz:
        cv2.destroyAllWindows()

    # divide cumulative error by # frames to get average error per frame
    normalized_iou_error = cumulative_iou_error / frame_num

    return predicted_contour_areas, ground_truth_contour_areas, ground_truth_thickness, ground_truth_thickness_ratio, predicted_thickness, predicted_thickness_ratio, normalized_iou_error, iou_accuracy_series
