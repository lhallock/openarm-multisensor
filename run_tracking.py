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
import argparse
from multisensorimport.tracking import tracking_executor as viz
from multisensorimport.tracking.paramvalues import ParamValues

def main():
    """Execute tracking run based on command line arguments given.
    """
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
    parser.add_argument(
        '--out_path',
        type=str,
        help='file path to folder to which CSV data should be written')
    parser.add_argument('--init_img',
                        type=str,
                        help='file name for first frame in ultrasound sequence')

    args = parser.parse_args()
    arg_params = vars(args)

    # initialize parameters for running tracking. If non-default parameter values are to be used, set them here.
    parameter_values = ParamValues()

    viz.tracking_run(arg_params, parameter_values)


if __name__ == "__main__":
    main()
