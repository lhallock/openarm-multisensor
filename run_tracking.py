"""
Runner script containing parameters and main method to do tracking. Specify which algorithm to use via run_type.

Run this via:
    $ python run_tracking.py --run_type <algorithm integer> --img_path <filepath to US frames> --seg_path <filepath to segmented images> --out_path <filepath to where output will be written> --init_img <name of first image in US frame series>

    algorithm integer to algorithm mapping:
        1: LK
        2: FRLK
        3: BFLK
        4: SBLK
"""

import numpy as np

import ultrasoundviz as viz


class ParamValues():
    """
    Class containing instance variables which are parameters used in various tracking algorithms and image filtering techniques, and a method to modify these variables. Used to easily change parameters for tuning.
    """

    # LK params:
    # window size for Lucas Kanade
    LK_window = 35
    # Pyramiding level for Lucas Kanade
    pyr_level = 3

    # FRLK params:
    # Quality level of corners chosen via Shi-Tomasi corner detection
    quality_level = 0.4

    # Minimum distance between corners chosen via Shi-Tomasi corner detection
    min_distance = 0

    # Maximum number of good corner points chosen
    max_corners = 300
    # Block size used for derivative kernel in ShiTomasi corner scoring
    block_size = 7

    # Fraction of top points (based on corner score) to keep in FRLK
    point_frac = 0.7

    # BFLK params:
    # Bilateral filter parameters - 'course/less agressive bilateral filter':
    course_diam = 5
    course_sigma_color = 100
    course_sigma_space = 100

    # Bilateral filter parameters - 'Fine/more agressive bilateral filter':
    fine_diam = 20
    fine_sigma_color = 80
    fine_sigma_space = 80

    # fraction of points (ordered by corner score) to track using fine/course filters
    percent_fine = 0.2
    percent_course = 0.8

    # SBLK params:
    # offset (alpha) used in the weighting function for supporters points
    displacement_weight = 40

    # fraction of points to track without supporters
    fine_threshold = 0.45

    # diagonal entries to initialize covariance matrix, for supporters prediction
    supporter_variance = 10

    # update rate for exponential moving average
    update_rate = 0.7

    # common parameters
    # number of lowermost contour points to keep (used to ensure that points along entire contour were kept)
    num_bottom = 0

    # flag to maintain the top edge of the contour across tracking (mitigates against downward drift)
    fix_top = False

    # How often to reset contour to ground truth (used to analyze how often/when drift occurs)
    # set to high number (i.e. > # frames) for no reset
    reset_frequency = 100000

    def change_values(self, disp_weight, qual_level, min_dist, course_d,
                      course_sigma_c, course_sigma_s, fine_d, fine_sigma_c,
                      fine_sigma_s, window, pyr, fine_thresh, num_bot,
                      perc_fine, perc_course, reset_freq):
        """
        Method to modify the parameter instance variables to the given arguments. Only changes the arguments which are not None. Used for parameter tuning.
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
        return self.displacement_weight


global parameter_values
parameter_values = ParamValues()


def write_run():
    """
    Execute a run of tracking, based on command line arguments given.
    """
    import argparse

    # set up command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_type',
                        type=int,
                        help='denotes which algorithm to use')
    parser.add_argument('--img_path',
                        type=str,
                        help='filepath to the raw ultrasound images to track')
    parser.add_argument('--seg_path',
                        type=str,
                        help='filepath to the segmented ground truth images')
    parser.add_argument(
        '--out_path',
        type=str,
        help='filepath to the folder to which csv data should be written')
    parser.add_argument('--init_img',
                        type=str,
                        help='filename for first frame in ultrasound video')

    args = parser.parse_args()
    arg_params = vars(args)

    viz.tracking_run(arg_params, parameter_values)


if __name__ == "__main__":
    write_run()
