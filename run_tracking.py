#!/usr/bin/env python3
"""Script to track ultrasound muscle contours via optical flow.

Example:
    Run this function via

        $ python run_tracking.py --run_type <alg_int> --img_path <filepath_us>\
        --seg_path <filepath_seg> --out_path <filepath_out> --init_img\
        <filename_init>

    for
        alg_int: integer value corresponding to desired algorithm
            1: LK
            2: FRLK
            3: BFLK
            4: SBLK
        filepath_us: file path to raw ultrasound PGM frames
        filepath_seg: file path to segmented PGM images
        filepath_out: file path to which [TODO: spec. files] will be written
        filename_init: file name of first image in ultrasound frame series
"""

import numpy as np

import ultrasoundviz as viz

class ParamValues():
    """Class containing tracking algorithm parameter values.

    This class contains all parameters used in optical flow tracking of
    ultrasound images as instance variables, as well as a method to modify
    these variables. It is used to easily modify parameters for tuning.
    """
    ###########################################################################
    ## LK PARAMETERS
    ###########################################################################

    # window size for Lucas Kanade
    LK_window = 35

    # pyramiding level for Lucas Kanade
    pyr_level = 3

    ###########################################################################
    ## FRLK PARAMETERS
    ###########################################################################

    # quality level of corners chosen via Shi-Tomasi corner detection
    quality_level = 0.4

    # minimum distance between corners chosen via Shi-Tomasi corner detection
    min_distance = 0

    # maximum number of good corner points chosen
    max_corners = 300

    # block size used for derivative kernel in Shi-Tomasi corner scoring
    block_size = 7

    # fraction of top points (based on corner score) to keep in FRLK
    point_frac = 0.7

    ###########################################################################
    ## BFLK PARAMETERS
    ###########################################################################

    # bilateral filter parameters - coarse/less aggressive
    course_diam = 5
    course_sigma_color = 100
    course_sigma_space = 100

    # bilateral filter parameters - fine/more aggressive
    fine_diam = 20
    fine_sigma_color = 80
    fine_sigma_space = 80

    # fraction of points (ordered by corner score) to track using fine/course filters
    percent_fine = 0.2
    percent_course = 0.8

    ###########################################################################
    ## SBLK PARAMETERS
    ###########################################################################

    # offset (alpha) used in weighting function for supporter points
    displacement_weight = 40

    # fraction of points to track without supporters
    fine_threshold = 0.45

    # diagonal entries to initialize covariance matrix, for supporters prediction
    supporter_variance = 10

    # update rate for exponential moving average
    update_rate = 0.7

    ###########################################################################
    ## SHARED/COMMON PARAMETERS
    ###########################################################################

    # number of lowermost contour points to keep (used to ensure point along
    # entire contour are kept)
    num_bottom = 0

    # flag to maintain top edge of contour across tracking (mitigates downward
    # drift)
    fix_top = False

    # how often to reset contour to ground truth (used to analyze when and how
    # often drift occurs)
    # set to high number (i.e., > # total frames) for no reset
    reset_frequency = 100000

    ###########################################################################
    ## GETTERS/SETTERS
    ###########################################################################

    def change_values(self, disp_weight, qual_level, min_dist, course_d,
                      course_sigma_c, course_sigma_s, fine_d, fine_sigma_c,
                      fine_sigma_s, window, pyr, fine_thresh, num_bot,
                      perc_fine, perc_course, reset_freq):
        """Modify parameter instance variables to given arguments.

        This method is used for parameter tuning, and only changes arguments
        that are not None.
        """
        if disp_weight is not None:
            self.displacement_weight = disp_weight
        if qual_level is not None:
            self.quality_level = qual_level
        if min_dist is not None:
            self.min_distance = min_dist
        if course_d is not None:
            self.course_diam = course_d
        if course_sigma_c is not None:
            self.course_sigma_color = course_sigma_c
        if course_sigma_s is not None:
            self.course_sigma_space = course_sigma_s
        if fine_d is not None:
            self.fine_diam = fine_d
        if fine_sigma_c is not None:
            self.fine_sigma_color = fine_sigma_c
        if fine_sigma_s is not None:
            self.fine_sigma_space = fine_sigma_s
        if window is not None:
            self.LK_window = window
        if pyr is not None:
            self.pyr_level = pyr
        if fine_thresh is not None:
            self.fine_threshold = fine_thresh
        if num_bot is not None:
            self.num_bottom = num_bot
        if perc_fine is not None:
            self.percent_fine = perc_fine
        if perc_course is not None:
            self.percent_course = perc_course
        if reset_freq is not None:
            self.reset_frequency = reset_freq

    def get_displacement_weight(self):
        """Get SBLK object displacement weight parameter.

        Returns:
            float displacement weight
        """
        return self.displacement_weight


global parameter_values
parameter_values = ParamValues()


def write_run():
    """Execute tracking run based on command line arguments given.
    """
    import argparse

    # set up command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_type',
                        type=int,
                        help='denotes which algorithm to use')
    parser.add_argument('--img_path',
                        type=str,
                        help='file path to raw ultrasound images to track')
    parser.add_argument('--seg_path',
                        type=str,
                        help='file path to segmented ground truth images')
    parser.add_argument('--out_path',
                        type=str,
        help='file path to folder to which CSV data should be written')
    parser.add_argument('--init_img',
                        type=str,
                        help='file name for first frame in ultrasound sequence')

    args = parser.parse_args()
    arg_params = vars(args)

    viz.tracking_run(arg_params, parameter_values)


if __name__ == "__main__":
    write_run()
