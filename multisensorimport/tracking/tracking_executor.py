#!/usr/bin/env python3
"""Executes and visualizes contour tracking of ultrasound frame series using
specified algorithm.
"""
import cv2
import numpy as np

from multisensorimport.tracking import point_proc_utils as point_proc
from multisensorimport.tracking import supporters_utils as supporters_utils
from multisensorimport.tracking import tracking_algorithms as track


def tracking_run(arg_params, run_params):
    """Execute ultrasound image tracking, tracking evaluation, and tracking
    visualization.

    Args:
        arg_params (dict): dictionary specifying algorithm to use, read path
            for raw and segmented files, and initial image name
        run_params (ParamValues): instance of ParamValues class containing
            parameter values for each optical flow algorithm
        run_type (int): integer determining which tracking algorithm to run
            (mappings documented in run_tracking.py)
    """
    read_path = arg_params['img_path']
    seg_path = arg_params['seg_path']
    out_path = arg_params['out_path']
    init_img_name = arg_params['init_img']
    run_type = arg_params['run_type']

    # set Lucas-Kanade optical flow parameters
    window_size = run_params.LK_window
    lk_params = dict(winSize=(window_size, window_size),
                     maxLevel=run_params.pyr_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # set parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=run_params.max_corners,
                          qualityLevel=run_params.quality_level,
                          minDistance=run_params.min_distance,
                          blockSize=run_params.block_size)

    # set initial image path
    init_path = read_path + init_img_name
    init_img = cv2.imread(init_path, -1)

    # extract initial contour from keyframe path
    keyframe_path = seg_path + init_img_name
    initial_contour_pts = track.extract_contour_pts_pgm(keyframe_path)

    # track points
    if run_type == 1:
        print("LK tracking")
        # obtain results from tracking
        tracking_contour_areas, ground_truth_contour_areas, tracking_thickness, ground_truth_thickness, tracking_thickness_ratio, ground_truth_thickness_ratio, iou_series, iou_error = track.track_LK(
            run_params,
            seg_path,
            read_path,
            initial_contour_pts,
            lk_params,
            viz=True,
            filter_type=0)

    elif run_type == 2:
        print("FRLK tracking")
        # 7 is window size, 0.7 is fraction of points to keep
        shi_tomasi_window = run_params.block_size
        fraction_points = run_params.point_frac

        # do not apply a filter for determining corner score for contour points
        filter_type_tomasi = 0

        # filter the contour points to track based on their corner scores
        filtered_initial_contour, indices = track.filter_points(
            run_params, shi_tomasi_window, initial_contour_pts,
            filter_type_tomasi, init_img, fraction_points)
        # order the contour points in counter-clockwise order for easier OpenCV contour analysis
        filtered_initial_contour = track.order_points(filtered_initial_contour,
                                                      indices, np.array([]),
                                                      np.array([]))
        # obtain results from tracking
        tracking_contour_areas, ground_truth_contour_areas, tracking_thickness, ground_truth_thickness, tracking_thickness_ratio, ground_truth_thickness_ratio, iou_series, iou_error = track.track_LK(
            run_params,
            seg_path,
            read_path,
            filtered_initial_contour,
            lk_params,
            viz=True,
            filter_type=0,
            filtered_LK_run=True)

    elif run_type == 3:
        print("BFLK tracking")
        # separate points into those to be tracked with the less aggressive bilateral filter, and those to be tracked with the more aggressive bilateral filter
        fine_filtered_points, fine_pts_inds, coarse_filtered_points, coarse_pts_inds = point_proc.separate_points(
            run_params, init_img, initial_contour_pts)

        # obtain results from tracking
        tracking_contour_areas, ground_truth_contour_areas, tracking_thickness, ground_truth_thickness, tracking_thickness_ratio, ground_truth_thickness_ratio, iou_series, iou_error = track.track_BFLK(
            run_params, seg_path, read_path, fine_filtered_points,
            fine_pts_inds, coarse_filtered_points, coarse_pts_inds, lk_params)

    elif run_type == 4:
        print("SBLK tracking")

        # initialize contours and supporters
        coarse_filtered_points, coarse_pts_inds, fine_filtered_points, fine_pts_inds, supporters_tracking, _ = supporters_utils.initialize_supporters(
            run_params, read_path, keyframe_path, init_img, feature_params,
            lk_params, 2)

        # initialize supporters
        supporter_params = []
        for i in range(len(coarse_filtered_points)):
            point = coarse_filtered_points[i][0]
            _, sup_params = supporters_utils.initialize_supporters_for_point(
                supporters_tracking, point, 10)
            supporter_params.append(sup_params)

        # determine image filters to apply on frames
        fineFilterNum = 2
        coarseFilterNum = 3

        # obtain results from tracking

        tracking_contour_areas, ground_truth_contour_areas, tracking_thickness, ground_truth_thickness, tracking_thickness_ratio, ground_truth_thickness_ratio, iou_series, iou_error = track.track_SBLK(
            run_params,
            seg_path,
            read_path,
            fine_filtered_points,
            fine_pts_inds,
            coarse_filtered_points,
            coarse_pts_inds,
            supporters_tracking,
            supporter_params,
            lk_params,
            True,
            feature_params,
            True,
            fine_filter_type=fineFilterNum,
            coarse_filter_type=coarseFilterNum)

    # Errors/Accuracy measures from tracking
    thickness_error = np.linalg.norm(
        np.array([ground_truth_thickness]) -
        np.array([tracking_thickness])) / len(tracking_thickness)

    thickness_ratio_error = np.linalg.norm(
        np.array([ground_truth_thickness_ratio]) -
        np.array([tracking_thickness_ratio])) / len(tracking_thickness_ratio)

    csa_error = np.linalg.norm(
        np.array([ground_truth_contour_areas]) -
        np.array([tracking_contour_areas])) / len(tracking_contour_areas)

    print("THICKNESS ERROR: ", thickness_error)
    print("THICKNESS RATIO ERROR: ", thickness_ratio_error)
    print("CSA ERROR: ", csa_error)
    print("IOU ACCURACY: ", iou_error)

    # write contour areas to csv file

    # change to True if ground truth needs to be written
    write_ground_truth = True

    write_tracking = True

    if write_ground_truth:
        out_path_csa_ground_truth = out_path + 'ground_truth_csa.csv'
        with open(out_path_csa_ground_truth, 'w') as outfile:
            for ctr in ground_truth_contour_areas:
                outfile.write(str(ctr))
                outfile.write('\n')

        out_path_thickness_ground_truth = out_path + 'ground_truth_thickness.csv'
        with open(out_path_thickness_ground_truth, 'w') as outfile:
            for thickness in ground_truth_thickness:
                outfile.write(str(thickness))
                outfile.write('\n')

        out_path_thickness_ratio_ground_truth = out_path + 'ground_truth_thickness_ratio.csv'
        with open(out_path_thickness_ratio_ground_truth, 'w') as outfile:
            for thickness_ratio in ground_truth_thickness_ratio:
                outfile.write(str(thickness_ratio))
                outfile.write('\n')

    if write_tracking:
        out_path_tracking_csa = out_path + 'tracking_csa.csv'
        with open(out_path_tracking_csa, 'w') as outfile:
            for ctr in tracking_contour_areas:
                outfile.write(str(ctr))
                outfile.write('\n')

        out_path_tracking_thickness = out_path + 'tracking_thickness.csv'
        with open(out_path_tracking_thickness, 'w') as outfile:
            for ctr in tracking_thickness:
                outfile.write(str(ctr))
                outfile.write('\n')

        out_path_tracking_thickness_ratio = out_path + 'tracking_thickness_ratio.csv'
        with open(out_path_tracking_thickness_ratio, 'w') as outfile:
            for ctr in tracking_thickness_ratio:
                outfile.write(str(ctr))
                outfile.write('\n')

        out_path_tracking_iou_series = out_path + 'iou_series.csv'
        with open(out_path_tracking_iou_series, 'w') as outfile:
            for ctr in iou_series:
                outfile.write(str(ctr))
                outfile.write('\n')

    return iou_error, thickness_error, thickness_ratio_error, csa_error
