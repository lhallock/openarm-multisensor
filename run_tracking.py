#!/usr/bin/env python3
"""Script to track ultrasound muscle contours via optical flow.

Example:
    Run this function via

        $ python run_tracking.py --run_type <alg_int> --img_path <filepath_us>
        --seg_path <filepath_seg> --out_path <filepath_out> --init_img
        <filename_init>

    for
        alg_int: integer value corresponding to desired algorithm
            1: LK
            2: FRLK
            3: BFLK
            4: SBLK
        filepath_us: file path to raw ultrasound PGM frames
        filepath_seg: file path to ground truth segmented PGM images
        filepath_out: file path to which ground_truth_csa.csv,
            ground_truth_thickness.csv, ground_truth_thickness_ratio.csv,
            tracking_csa.csv, tracking_thickness.csv,
            tracking_thickness_ratio.csv, and iou_series.csv will be written
        filename_init: file name of first image in ultrasound frame series
"""
import argparse

from multisensorimport.tracking import tracking_executor
from multisensorimport.tracking.paramvalues import ParamValues

# set tracking parameters; for full description, see
# multisensorimport.tracking.ParamValues class
LK_WINDOW = 35
PYR_LEVEL = 3
QUALITY_LEVEL = 0.4
MIN_DISTANCE = 0
MAX_CORNERS = 300
BLOCK_SIZE = 7
POINT_FRAC = 0.7
COARSE_DIAM = 5
COARSE_SIGMA_COLOR = 100
COARSE_SIGMA_SPACE = 100
FINE_DIAM = 20
FINE_SIGMA_COLOR = 80
FINE_SIGMA_SPACE = 80
PERCENT_FINE = 0.2
PERCENT_COARSE = 0.8
DISPLACEMENT_WEIGHT = 40
FINE_THRESHOLD = 0.45
UPDATE_RATE = 0.7
NUM_BOTTOM = 0
FIX_TOP = False
RESET_FREQUENCY = 10000


def main():
    """Execute tracking run based on above and command line parameters."""
    # set up command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_type',
                        type=int,
                        help='integer value corresponding to desired algorithm')
    parser.add_argument('--img_path',
                        type=str,
                        help='file path to raw ultrasound PGM frames')
    parser.add_argument('--seg_path',
                        type=str,
                        help='file path to ground truth segmented PGM images')
    parser.add_argument('--out_path',
                        type=str,
                        help='file path to which CSV data will be written')
    parser.add_argument(
        '--init_img',
        type=str,
        help='file name of first image in ultrasound frame series')

    args = parser.parse_args()
    arg_params = vars(args)

    # initialize parameters for running tracking
    parameter_values = ParamValues(
        LK_WINDOW, PYR_LEVEL, QUALITY_LEVEL, MIN_DISTANCE, MAX_CORNERS,
        BLOCK_SIZE, POINT_FRAC, COARSE_DIAM, COARSE_SIGMA_COLOR,
        COARSE_SIGMA_SPACE, FINE_DIAM, FINE_SIGMA_COLOR, FINE_SIGMA_SPACE,
        PERCENT_FINE, PERCENT_COARSE, DISPLACEMENT_WEIGHT, FINE_THRESHOLD,
        UPDATE_RATE, NUM_BOTTOM, FIX_TOP, RESET_FREQUENCY)

    tracking_executor.tracking_run(arg_params, parameter_values)


if __name__ == "__main__":
    main()
