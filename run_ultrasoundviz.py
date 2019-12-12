#!/usr/bin/env python3
"""Example display of ultrasound image tracking.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_ultrasoundviz.py

Todo:
    implement all the things!
"""

import time
import os

import cv2
import numpy as np

from multisensorimport.tracking import us_tracking_utils as track

READ_PATH = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/ultrasound_t5w1_expanded/'


def main():
    """Execute ultrasound image tracking visualization."""
    # set Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(25, 25),
                     maxLevel=8,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    # extract initial contour from keyframe
    keyframe_path = READ_PATH + '0.png'
    pts = track.extract_contour_pts(keyframe_path)

    # track points
    contour_areas = track.track_pts(READ_PATH, pts, lk_params)

    # write contour areas to csv file
    out_path = READ_PATH + 'csa.csv'
    with open(out_path, 'w') as outfile:
        for ctr in contour_areas:
            outfile.write(str(ctr))
            outfile.write('\n')


if __name__ == "__main__":
    main()
