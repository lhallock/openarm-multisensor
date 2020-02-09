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
import scipy

from multisensorimport.tracking import supporters_simple as supporters_simple
from multisensorimport.tracking import params as parameters

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

def extract_contour_pts_two(filename):
    # function to extract contour points from a pgm file; similar to function above but for different file type

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

    # convert largest contour to tracking-compatible array
    points = []
    for i in range(len(contours[0])):
        points.append(np.array(contours[0][i], dtype=np.float32))
        #cv2.circle(frame_color, (x, y), 5, (0, 255, 0), -1)
    np_points = np.array(points)


    return np_points


def track_pts_lucas_kanade(seg_filedir, filedir, pts, lk_params, viz = True, filter_type=0):

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


    # track and display specified points through images
    first_loop = True
    frame_num = 0

    # cumulative intersection over union error (sum over all frames)
    cumulative_iou_error = 0

    for num in range(len(sorted_image_filenames)):
        image_filename = sorted_image_filenames[num]
        segmented_filename = sorted_segmented_filenames[num]
        print("image_filename: ", image_filename)
        print("segmented_filename: ", segmented_filename)

        print("FRAME: ", frame_num)
        if image_filename.endswith('.pgm') and segmented_filename.endswith('.pgm'):
            assert(segmented_filename == image_filename)
            filepath = filedir + image_filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)

                # obtain key frame, for re-initializing points and/or intersection/union computation
                key_frame_path = seg_filedir + segmented_filename

                tracked_contour, status, error = cv2.calcOpticalFlowPyrLK(old_frame, frame, pts, None, **lk_params)

                segmented_contour = extract_contour_pts_two(key_frame_path)

                frame_color = frame.copy()
                # draw the predicted contour
                #cv2.drawContours(frame_color, [segmented_contour.astype(int)], 0, (0, 255, 0), 3)
                #cv2.drawContours(frame_color, [tracked_contour.astype(int)], 0, (255, 0, 0), 3)

                segmented_area = cv2.contourArea(segmented_contour)

                segmented_thickness_x, segmented_thickness_y = thickness(supporters_simple.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(supporters_simple.format_supporters(tracked_contour))

                ground_truth_thickness.append(segmented_thickness_x)
                ground_truth_thickness_ratio.append(segmented_thickness_x / segmented_thickness_y)

                predicted_thickness.append(predicted_thickness_x)
                predicted_thickness_ratio.append(predicted_thickness_x / predicted_thickness_y)



                # append new predicted contour area
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
                print("intersection over union error: ", iou_error)
                cumulative_iou_error += iou_error








def track_pts(seg_filedir, filedir, fine_pts, fine_pts_inds, course_pts, course_pts_inds, supporter_pts, supporter_params, lk_params, reset_supporters, feature_params,viz=True, fine_filter_type=0, course_filter_type=0):
    """Track specified points through all (ordered) images in directory.

    This function is used to track, record, and visualize points through a full
    directory of images, recording values of interest at each frame. In
    particular, this function is used to calculate the cross-sectional area of
    the brachioradialis muscle at each frame.

    Args:
        filedir (str): directory in which files are stored, including final '/'
        fine_pts (numpy.ndarray): list of points with higher corner scores, from an aggressively filtered image, to track; assumed to correspond to
            the first image file in filedir, and to be in counter-clockwise
            order (TODO: check this)
        course_pts (numpy.ndarray): list of points with higher corner scores, from a less filtered image, to track; assumed to correspond to
            the first image file in filedir, and to be in counter-clockwise.
            order (TODO: check this)
        lk_params (dict): parameters for Lucas-Kanade image tracking in OpenCV
            TODO: set appropriate default values
        viz (bool): whether to visualize the tracking process
        fine_filter_type (int): what kind of filter to use to "finely" filter points (more aggressive filter) (0 for none, 1 for median, 2 for fine bilateral, 3 for course bilateral, 4 for anisotropic diffusion)
        course_filter_type (int): same as fine_filter_type, but less aggressive, used to "coursely" filter points (keeps more top right points)

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
    reset_freq = 100000

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
        print("image_filename: ", image_filename)
        print("segmented_filename: ", segmented_filename)

        print("FRAME: ", frame_num)
        if image_filename.endswith('.pgm') and segmented_filename.endswith('.pgm'):
            # make sure segmented and image filenames are the same
            assert(segmented_filename == image_filename)

            # filepath to the image
            filepath = filedir + image_filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                # apply filters to frame
                old_frame_course = course_filter(old_frame)
                old_frame_fine = fine_filter(old_frame)
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)

                # obtain key frame, for re-initializing points and/or iou computation
                key_frame_path = seg_filedir + segmented_filename

                # apply filters to frame
                frame_course = course_filter(frame)
                frame_fine = fine_filter(frame)
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # reset condition
                if frame_num % reset_freq == 0:
                    print("RESETTING POINTS")
                    if reset_supporters:
                        # reset all points to contour, and re-initialize a new set of supporters based on good corner features
                        course_pts, course_pts_inds, fine_pts, fine_pts_inds, supporter_pts, supporter_params = initialize_points(filedir, key_frame_path, frame, feature_params, lk_params, 2)
                    else:
                        # reset tracking points to contour, but do not set new supporter points.
                        course_pts, course_pts_inds, fine_pts, fine_pts_inds, _, _ = initialize_points(filedir, key_frame_path, frame, feature_params, lk_params, 2)
                        # re-initialize parameters for supporters
                        supporter_params = []
                        for i in range(len(course_pts)):
                            point = course_pts[i][0]
                            _, params = supporters_simple.initialize_supporters(supporter_pts, point, 10)
                            supporter_params.append(params)
                else:
                    # calculate new point locations for fine_points using frame filtered by the fine filter
                    new_fine_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_fine, frame_fine, fine_pts, None, **lk_params)

                    # calculate new point locations for course_points using frame filtered by the course filter: predictions, might be updated by supporters
                    predicted_course_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_course, frame_course, course_pts, None, **lk_params)

                    # calculate new supporter locations in course filter frame
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


                    # TODO: check this
                    for i in range(len(predicted_course_pts)):
                        predicted_point = predicted_course_pts[i]
                        param_list = supporter_params[i]

                        # pass in both supporter_pts (the old values) and new_supporter_pts (old values) so that the displacement can be calculated
                        # 0.7 is alpha (learning rate)
                        point_location, new_params = supporters_simple.apply_supporters_model(predicted_point, supporter_pts, new_supporter_pts, param_list, use_tracking, 0.7)
                        updated_feature_params.append(new_params)
                        new_course_pts.append(np.array([[point_location[0], point_location[1]]], dtype=np.float32))

                    new_course_pts = np.array(new_course_pts)

                    # save old frame for optical flow calculation
                    old_frame_course = frame_course.copy()
                    old_frame_fine = frame_fine.copy()

                    # reset point locations
                    fine_pts = new_fine_pts
                    course_pts = new_course_pts
                    supporter_pts = new_supporter_pts

                    # reset supporter params
                    supporter_params = updated_feature_params


                for i in range(len(fine_pts)):
                    x, y = fine_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 5, (255, 0, 0), -1)
                for i in range(len(course_pts)):
                    x, y = course_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 5, (0, 255, 0), -1)
                for i in range(len(supporter_pts)):
                    x, y = supporter_pts[i].ravel()
                    cv2.circle(frame_color, (x, y), 5, (0, 0, 255), -1)

                tracked_contour = order_points(fine_pts, fine_pts_inds, course_pts, course_pts_inds)


                # set the y coordinate of first and last point to 0 to fix downward boundary drift

                if parameters.fix_top:
                    first_contour_point = tracked_contour[0]
                    first_contour_point[0][1] = 0

                    last_contour_point = tracked_contour[len(tracked_contour) - 1]
                    last_contour_point[0][1] = 0

                # contour from ground truth segmentaton
                segmented_contour = extract_contour_pts_two(key_frame_path)

                # draw the predicted contour
                # cv2.drawContours(frame_color, [segmented_contour.astype(int)], 0, (0, 255, 0), 3)
                # cv2.drawContours(frame_color, [tracked_contour.astype(int)], 0, (255, 0, 0), 3)


                # display to frame
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27: # stop on escape key
                        break
                    time.sleep(0.01)

                segmented_area = cv2.contourArea(segmented_contour)

                segmented_thickness_x, segmented_thickness_y = thickness(supporters_simple.format_supporters(segmented_contour))

                predicted_thickness_x, predicted_thickness_y = thickness(supporters_simple.format_supporters(tracked_contour))

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

                # calculate intersection over union

                # initialize matrices of zeros
                mat_predicted = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                # fill the initialized matrices with nonzero numbers in the area of the contour
                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)], 255)

                # cv2.imshow("SEG", mat_segmented)
                # cv2.imshow("PRED", mat_predicted)
                # cv2.waitKey()

                intersection = np.sum(np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                iou_error = intersection / union
                print("intersection over union error: ", iou_error)
                cumulative_iou_error += iou_error


        frame_num += 1

    if viz:
        cv2.destroyAllWindows()

    # divide cumulative error by # frames to get average error per frame
    normalized_iou_error = cumulative_iou_error / frame_num

    return predicted_contour_areas, ground_truth_contour_areas, ground_truth_thickness, ground_truth_thickness_ratio, predicted_thickness, predicted_thickness_ratio, normalized_iou_error

def filter_supporters(supporter_points, filedir, lk_params):
    """
    Takes in supporter points, goes through frames, and only keeps the top x percentile of points, ranked by how much they moved.
    Useful to remove supporter points that are part of background
    """
    # print("FILTERING")
    # track and display specified points through images
    first_loop = True
    frame_num = 0

    # array to keep track of net movement of each supporter
    movement = []

    # store old frame for LK tracking
    old_frame_course = None

    # initialize movements to 0
    for i in range(len(supporter_points)):
        movement.append(0)

    for filename in sorted(os.listdir(filedir)):
        # print("FRAME: ", frame_num)
        if filename.endswith('.pgm'):
            filepath = filedir + filename
            # print(filepath)

            # if it's the first image, initialize old_frame
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                # apply filter to frame
                old_frame_course = course_bilateral_filter(old_frame)
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)

                # apply filter to frame
                frame_course = course_bilateral_filter(frame)
                # track points
                new_supporter_points, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame_course, frame_course, supporter_points, None, **lk_params)

                # reformat for easier processing
                new_supporter_points_formated = supporters_simple.format_supporters(new_supporter_points)

                prev_supporter_points_formated = supporters_simple.format_supporters(supporter_points)

                # update the movements
                for i in range(len(new_supporter_points_formated)):
                    new_supporter_point = new_supporter_points_formated[i]
                    prev_supporter_point = prev_supporter_points_formated[i]
                    movement[i] += np.linalg.norm(new_supporter_point - prev_supporter_point)

                old_frame_course = frame_course.copy()
                supporter_points = new_supporter_points.copy()


        frame_num += 1

    movement = np.array(movement)
    movement / frame_num
    moved_points = movement >= scipy.percentile(movement, 10)

    # print("FINISHED FILTERING")
    # return the indeces of the points to keep
    return moved_points



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
                    time.sleep(0.0001)

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


def filter_points(window_size, pts, filter_type, img, percent, keep_bottom=False):

    # select image filter, determined by filterType argument
    filter = get_filter_from_num(filter_type)

    # apply filter
    filtered_img = filter(img)
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

    # converts map to a list of 2-tuples (key, value), which are in sorted order by value
    # key is index of point in the pts list
    sorted_corner_mapping = sorted(ind_to_score_map.items(), key=lambda x: x[1], reverse=True)
    sorted_y_mapping = sorted(ind_to_y_map.items(), key=lambda x: x[1], reverse=True)
    # get top percent% of points
    for i in range(0, round(percent * len(sorted_corner_mapping))):
        points_ind = sorted_corner_mapping[i][0]
        filtered_points.append(pts[points_ind])
        filtered_points_ind.append(points_ind)


    if keep_bottom:
        for i in range(parameters.num_bottom):
            points_ind = sorted_y_mapping[i][0]
            filtered_points.append(pts[points_ind])
            filtered_points_ind.append(points_ind)

    return np.array(filtered_points), np.array(filtered_points_ind)


def initialize_points(READ_PATH, keyframe_path, init_img, feature_params, lk_params, which_contour):

    # determines which points will get tracked by Lucas Kanade solely, and which will need supporter assistance
    # also determines supporter points to use based on good corner features (Shi Tomasi)

    if which_contour == 1:
        pts = extract_contour_pts(keyframe_path)
    else:
        pts= extract_contour_pts_two(keyframe_path)

    mean_x_pts = 0
    mean_y_pts = 0

    for supporter_tracked_point in pts:
        x = supporter_tracked_point[0][0]
        y = supporter_tracked_point[0][1]
        mean_x_pts += x
        mean_y_pts += y
    mean_x_pts = mean_x_pts / len(pts)
    mean_y_pts = mean_y_pts / len(pts)

    # filter to be used (1: median filter, 2: bilateral filter, 3: course bilateral, 4: anisotropicDiffuse anything else no filter )
    fineFilterNum = 2
    courseFilterNum = 3

    course_filter = get_filter_from_num(courseFilterNum)
    filtered_init_img = course_filter(init_img)

    # remove points that have low corner scores (Shi Tomasi Corner scoring): these points will be kept for LK tracking
    lucas_kanade_points, lucas_kanade_points_indeces = filter_points(7, pts, fineFilterNum, init_img, parameters.fine_threshold, keep_bottom=True)
    # add the first point to LK tracking (top left)
    lucas_kanade_points = np.append(lucas_kanade_points, np.array([pts[0]]), axis=0)
    lucas_kanade_points_indeces = np.append(lucas_kanade_points_indeces, 0)

    # obtain points which need supporters to be tracked
    supporter_tracked_points = pts.copy()
    supporter_tracked_points_indeces = np.arange(0, len(pts))

    # filter supporter tracked to be in desired region: should be in top right "quadrant" (greater than mean x, less than mean y)
    supporter_kept_indeces = set()
    for i in range(len(supporter_tracked_points)):
        supporter_tracked_point = supporter_tracked_points[i]
        add = (supporter_tracked_point[0][0] > mean_x_pts and supporter_tracked_point[0][1] < mean_y_pts)
        if add:
            supporter_kept_indeces.add(i)

    # only add the supporter_tracked points that we determined should be added
    supporter_tracked_to_keep = []
    supporter_tracked_to_keep_inds = []
    for index in supporter_kept_indeces:
        supporter_tracked_to_keep.append(supporter_tracked_points[index])
        supporter_tracked_to_keep_inds.append(supporter_tracked_points_indeces[index])

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
            if ((np.linalg.norm(lucas_kanade_point - supporter_point) < 0.001) or (lucas_kanade_point[0][0] > mean_x_pts and lucas_kanade_point[0][1] < mean_y_pts)):
                add = False
        if add:
            LK_kept_indeces.add(i)


    # only add the supporter_tracked points that we determined should be added
    LK_to_keep = []
    LK_to_keep_inds = []
    for index in LK_kept_indeces:
        LK_to_keep.append(lucas_kanade_points[index])
        LK_to_keep_inds.append(lucas_kanade_points_indeces[index])

    # reset the points to get tracked using supporters
    lucas_kanade_points = np.array(LK_to_keep)
    lucas_kanade_points_indeces = np.array(LK_to_keep_inds)

    # find supporters based on good points
    supporters = cv2.goodFeaturesToTrack(filtered_init_img, mask=None, **feature_params)
    # ind = filter_supporters(supporters_tracking, READ_PATH, lk_params)
    # supporters_tracking = supporters_tracking[ind]

    supporter_params = []
    for i in range(len(supporter_tracked_to_keep)):
        supporter_tracked_point = supporter_tracked_to_keep[i][0]
        # 10 is variance
        _, params = supporters_simple.initialize_supporters(supporters, supporter_tracked_point, 10)
        supporter_params.append(params)

    return supporter_tracked_points, supporter_tracked_points_indeces, lucas_kanade_points, lucas_kanade_points_indeces, supporters, supporter_params


def order_points(fine_pts, fine_pts_inds, course_pts, course_pts_inds):
    point_dict = dict()
    for i in range(len(fine_pts)):
        fine_pt = fine_pts[i]
        fine_pt_ind = fine_pts_inds[i]
        point_dict[fine_pt_ind] = fine_pt
    for i in range(len(course_pts)):
        course_pt = course_pts[i]
        course_pt_ind = course_pts_inds[i]
        point_dict[course_pt_ind] = course_pt

    pts = []
    for key in sorted(point_dict.keys()):
        pts.append(point_dict[key])

    return np.array(pts)


def thickness(points):
    min_x = float("inf")
    max_x = -1 * float("inf")

    min_y = float("inf")
    max_y = -1 * float("inf")

    for point in points:
        x = point[0]
        y = point[1]
        min_x = min(x, min_x)
        max_x = max(x, max_x)

        min_y = min(y, min_y)
        max_y = max(y, max_y)


    return (max_x - min_x), (max_y - min_y)




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
def no_filter(img):
    return img


def median_filter(img):
    # hyperparameter
    kernelSize = 5
    return cv2.medianBlur(img, kernelSize)


def fine_bilateral_filter(img):
    # convert to color (what bilateral filter expects)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    diam = parameters.fine_diam
    sigmaColor = parameters.fine_sigma_color
    sigmaSpace = parameters.fine_sigma_space
    bilateralColor = cv2.bilateralFilter(img, diam, sigmaColor, sigmaSpace)

    # convert back to grayscale and return
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)


def course_bilateral_filter(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # hyperparameters
    diam = parameters.course_diam
    sigmaColor = parameters.course_sigma_color
    sigmaSpace = parameters.course_sigma_space
    bilateralColor = cv2.bilateralFilter(img, diam, sigmaColor, sigmaSpace)
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)


def anisotropic_diffuse(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # hyperparameters
    alphaVar = 0.1
    KVar = 5
    nitersVar = 5
    diffusedColor = cv2.ximgproc.anisotropicDiffusion(src = img, alpha = alphaVar, K = KVar, niters = nitersVar)
    return cv2.cvtColor(diffusedColor, cv2.COLOR_RGB2GRAY)


def otsu_binarization(gray_image):
    ret2, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def canny(gray_image):
    edges = cv2.Canny(gray_image, 180, 200)
    return edges
