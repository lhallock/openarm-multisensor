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
from scipy import signal
from scipy.spatial import distance as dist
from multisensorimport.tracking import supporters_simple as supporters_simple


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
    img = cv2.imread(filename, -1)
    # convert image to grayscale if it iscolor
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_level = 127
    # binarize image
    _, binarized = cv2.threshold(img, threshold_level, 255, cv2.THRESH_BINARY)
    # flip image
    flipped = cv2.bitwise_not(binarized)
    contours, _ = cv2.findContours(flipped, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # convert largest contour to tracking-compatible array
    points = []
    frame_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(len(contours[0])):
        points.append(np.array(contours[0][i], dtype=np.float32))
        #cv2.circle(frame_color, (x, y), 5, (0, 255, 0), -1)
    np_points = np.array(points)
    # cvx_hull = cv2.convexHull(np_points, clockwise=True)
    # print(len(cvx_hull))
    # for i in range(len(cvx_hull)):
    #     x = cvx_hull[i][0][0]
    #     y = cvx_hull[i][0][1]
    #     cv2.circle(frame_color, (x, y), 5, (0, 255, 0), -1)


    # cv2.imshow('SEGMENTED', frame_color)
    # cv2.waitKey()

    return np_points



def track_pts_csrt(filedir, pts, viz=True):
    # keep track of contour areas
    contour_areas = []
    # add first contour area
    contour_areas.append(cv2.contourArea(pts))

    # create OpenCV window (if visualization is desired)
    if viz:
        cv2.namedWindow('Frame')

    # track and display specified points through images
    first_loop = True
    old_frame = None
    i = 0
    skip_num = 200
    for filename in sorted(os.listdir(filedir)):
        if filename.endswith('.pgm') and i % skip_num == 0:
            filepath = filedir + filename

            # if it's the first image, we already have the contour area
            if first_loop:
                old_frame = cv2.imread(filepath, -1)
                # apply filter to frame
                first_loop = False

            else:
                # read in new frame
                frame = cv2.imread(filepath, -1)
                # print("SHAPE", frame.shape)
                # apply filter to frame
                try:
                    shape = frame.shape
                except cv2.error as e:
                    print('Invalid frame!')
                    continue
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


                # # CSRT tracking
                pts = csrt_tracking(old_frame, frame, pts, (7, 7))
                old_frame = frame
                # print('POINTS: ', pts)

                for i in range(len(pts)):
                    x, y = pts[i].ravel()
                    x = int(np.rint(x))
                    y = int(np.rint(y))
                    cv2.circle(frame_color, (x, y), 5, (0, 255, 0), -1)

                # display to frame
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27: # stop on escape key
                        break
                    time.sleep(0.01)

                # append new contour area
               # contour_areas.append(cv2.contourArea(pts))
        i += 1

    if viz:
        cv2.destroyAllWindows()

    return contour_areas

def track_pts(seg_filedir, filedir, fine_pts, fine_pts_inds, course_pts, course_pts_inds, supporter_pts, supporter_params, lk_params, feature_params,viz=True, fine_filter_type=0, course_filter_type=0):
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
    # print(course_pts)
    # combine course and fine points
    pts = np.concatenate((fine_pts, course_pts), axis=0)

    # keep track of contour areas
    contour_areas = []
    # add first contour area
    contour_areas.append(cv2.contourArea(pts))

    # create OpenCV window (if visualization is desired)
    if viz:
        cv2.namedWindow('Frame')

    # set filters (course filter is a less aggressive filter, fine_filter more aggressive)
    course_filter = get_filter_from_num(course_filter_type)
    fine_filter = get_filter_from_num(fine_filter_type)

    # track and display specified points through images
    first_loop = True
    frame_num = 0
    for filename in sorted(os.listdir(filedir)):
        print(frame_num)
        if filename.endswith('.pgm'):
            filepath = filedir + filename
            # print(filepath)

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

                # obtain key frame, for re-initializing points and/or intersection/union computation
                key_frame_path = seg_filedir + filename
                segmented_contour = extract_contour_pts_two(key_frame_path)

                # apply filters to frame
                frame_course = course_filter(frame)
                frame_fine = fine_filter(frame)
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # how often we reset contour to a pre-segmented frame
                reset_freq = 110

                # reset condition
                if frame_num % reset_freq == 0 and frame_num < 300:
                    print("RESETTING POINTS")

                    course_pts, course_pts_inds, fine_pts, fine_pts_inds, supporter_pts, supporter_params = initialize_points(filedir, key_frame_path, frame, feature_params, lk_params, 2)

                else:
                    # calculate new point locations for fine_points using frame filtered by the fine filter
                    new_fine_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_fine, frame_fine, fine_pts, None, **lk_params)

                    # calculate new point locations for course_points using frame filtered by the course filter
                    predicted_course_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_course, frame_course, course_pts, None, **lk_params)

                    # calculate new supporter locations in course filter frame
                    new_supporter_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        old_frame_course, frame_course, supporter_pts, None, **lk_params
                    )

                    # reformat for supporters
                    predicted_course_pts = supporters_simple.format_supporters(predicted_course_pts)

                    new_feature_params = []
                    new_course_pts = []
                    use_tracking = ((frame_num % reset_freq) <= -1)
                    for i in range(len(predicted_course_pts)):
                        predicted_point = predicted_course_pts[i]
                        param_list = supporter_params[i]
                        point_location, new_params = supporters_simple.apply_supporters_model(predicted_point, supporter_pts, new_supporter_pts, param_list, use_tracking, 0.7)
                        # print("POINT: ", predicted_point)
                        new_feature_params.append(new_params)
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
                    supporter_params = new_feature_params


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
                cv2.drawContours(frame_color, [tracked_contour.astype(int)], 0, (0, 255, 0), 3)
                cv2.fillPoly(frame_color, [tracked_contour.astype(int)], 255)
                # display to frame
                if viz:
                    cv2.imshow('Frame', frame_color)
                    key = cv2.waitKey(1)
                    if key == 27: # stop on escape key
                        break
                    time.sleep(0.01)

                # append new contour area
                contour_areas.append(cv2.contourArea(tracked_contour))

                # calculate intersection over union
                mat_predicted = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)
                mat_segmented = np.zeros(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY).shape)

                cv2.fillPoly(mat_predicted, [tracked_contour.astype(int)], 255)
                cv2.fillPoly(mat_segmented, [segmented_contour.astype(int)], 255)

                # cv2.imshow("SEG", mat_segmented)
                # cv2.imshow("PRED", mat_predicted)
                # cv2.waitKey()

                intersection = np.sum(np.logical_and(mat_predicted, mat_segmented))
                union = np.sum(np.logical_or(mat_predicted, mat_segmented))

                jaccard_index = intersection / union
                print("JACCARD: ", jaccard_index)

        frame_num += 1

    if viz:
        cv2.destroyAllWindows()

    return contour_areas

def filter_supporters(supporter_points, filedir, lk_params):
    print("FILTERING")
    # track and display specified points through images
    first_loop = True
    frame_num = 0
    movement = []
    old_frame_course = None
    supporter_points_formated = supporters_simple.format_supporters(supporter_points)
    for i in range(len(supporter_points)):
        movement.append(0)

    for filename in sorted(os.listdir(filedir)):
        print(frame_num)
        if filename.endswith('.pgm'):
            filepath = filedir + filename
            # print(filepath)

            # if it's the first image, we already have the contour area
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
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                supporter_points, status, error = cv2.calcOpticalFlowPyrLK(
                    old_frame_course, frame_course, supporter_points, None, **lk_params)

                # reformat for supporters
                new_supporter_points = supporters_simple.format_supporters(supporter_points)
                for i in range(len(new_supporter_points)):
                    new_supporter_point = new_supporter_points[i]
                    prev_supporter_point = supporter_points_formated[i]
                    movement[i] += np.linalg.norm(new_supporter_point - prev_supporter_point)

        frame_num += 1

    movement = np.array(movement)
    movement / frame_num
    moved_points = movement >= scipy.percentile(movement, 50)
    print("FINISHED FILTERING")
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


def filter_points(window_size, pts, filter_type, img, percent):

    # select image filter, determined by filterType argument
    filter = get_filter_from_num(filter_type)

    # apply filter
    filtered_img = filter(img)

    # convert pts from np array to list for convenience, create dict for sorting
    pts = list(pts)
    ind_to_score_map = dict()
    for i in range(len(pts)):
        point = pts[i]
        corner_score = shi_tomasi_corner_score(point, window_size, filtered_img)
        ind_to_score_map[i] = corner_score

    filtered_points = []
    filtered_points_ind = []

    # converts map to a list of 2-tuples (key, value), which are in sorted order by value
    # key is index of point in the pts list
    sorted_mapping = sorted(ind_to_score_map.items(), key=lambda x: x[1], reverse=True)
    print("MAPPING: ", sorted_mapping)

    # get top percent% of points
    for i in range(0, round(percent * len(sorted_mapping))):
        points_ind = sorted_mapping[i][0]
        filtered_points.append(pts[points_ind])
        filtered_points_ind.append(points_ind)

    return np.array(filtered_points), np.array(filtered_points_ind)


def csrt_tracking(prev_img, curr_img, points, window_size):

    new_points = []

    for point in points:
        x = point[0][0]
        y = point[0][1]
        tracker = cv2.TrackerCSRT_create()
        x_lower = max(0, x - window_size[0]//2)
        y_lower = max(0, y - window_size[1]//2)
        bounding_box = (x_lower, y_lower, window_size[0], window_size[1])
        print('prev img shape: ', prev_img.shape)
        print('bounding box: ', bounding_box)
        tracker.init(prev_img, bounding_box)

        ret, new_bounding_box = tracker.update(curr_img)
        if not ret:
            continue
        lower_x = new_bounding_box[0]
        lower_y = new_bounding_box[1]
        translation_x = (x - window_size[0]//2) - lower_x
        translation_y = (y - window_size[1]//2) - lower_y

        new_x = x + translation_x
        new_y = y + translation_y

        if (new_x >= 0 and new_x < curr_img.shape[1]) and (new_y >= 0 and new_y < curr_img.shape[0]):
            new_points.append(np.array([np.array([new_x, new_y])]))

    color_curr_img = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2RGB)
    # visualize
    for point in new_points:
        int_point = point.astype(np.int64)
        x = int_point[0][0]
        y = int_point[0][1]
        if x <= 0 or y <= 0:
            print("OUT OF BOUNDS!")
            return
        cv2.circle(color_curr_img, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Frame2', color_curr_img)
    time.sleep(0.01)

    new_points = np.array(new_points)

    return new_points.reshape(len(new_points), 1, 2)

def initialize_points(READ_PATH, keyframe_path, init_img, feature_params, lk_params, which_contour):
    if which_contour == 1:
        pts = extract_contour_pts(keyframe_path)
    else:
        pts= extract_contour_pts_two(keyframe_path)

    # filter to be used (1: median filter, 2: bilateral filter, 3: course bilateral, 4: anisotropicDiffuse anything else no filter )
    fineFilterNum = 2
    courseFilterNum = 3

    filter = get_filter_from_num(courseFilterNum)
    filtered_init_img = filter(init_img)
    # remove points that have low corner scores (Shi Tomasi Corner scoring)
    fine_filtered_points, fine_filtered_indeces = filter_points(7, pts, fineFilterNum, init_img, .45)
    course_filtered_points, course_filtered_indeces = filter_points(7, pts, courseFilterNum, init_img, 1)
    print(pts[0])
    print(fine_filtered_points.shape)
    fine_filtered_points = np.append(fine_filtered_points, np.array([pts[0]]), axis=0)
    fine_filtered_indeces = np.append(fine_filtered_indeces, 0)



    # find points which differ
    course_points_indeces = set()
    for i in range(len(course_filtered_points)):
        coursePoint = course_filtered_points[i]
        add = True
        for j in range(len(fine_filtered_points)):
            finePoint = fine_filtered_points[j]
            if (np.linalg.norm(finePoint - coursePoint) < 0.001) or (not (coursePoint[0][0] > 90 and coursePoint[0][1] < 120)):
                add = False
        if add:
            course_points_indeces.add(i)


    # only add the course points that are already not being tracked by fine points
    coursePoints = []
    courseInds = []
    for index in course_points_indeces:
        coursePoints.append(course_filtered_points[index])
        courseInds.append(course_filtered_indeces[index])

    course_filtered_points = np.array(coursePoints)
    course_filtered_indeces = np.array(courseInds)

    # find supporters, and filter based on those that move (static supporters are less useful)
    supporters_tracking = cv2.goodFeaturesToTrack(filtered_init_img, mask=None, **feature_params)
    # ind = filter_supporters(supporters_tracking, READ_PATH, lk_params)
    # supporters_tracking = supporters_tracking[ind]

    supporter_params = []
    for i in range(len(coursePoints)):
        point = coursePoints[i][0]
        _, params = supporters_simple.initialize_supporters(supporters_tracking, point, 10)
        supporter_params.append(params)

    return course_filtered_points, course_filtered_indeces, fine_filtered_points, fine_filtered_indeces, supporters_tracking, supporter_params


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
    diam = 35
    sigmaColor = 80
    sigmaSpace = 80
    bilateralColor = cv2.bilateralFilter(img, diam, sigmaColor, sigmaSpace)

    # convert back to grayscale and return
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)


def course_bilateral_filter(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # hyperparameters
    diam = 9
    sigmaColor = 100
    sigmaSpace = 100
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
