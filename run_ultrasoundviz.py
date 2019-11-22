#!/usr/bin/env python3
"""Example display of ultrasound image tracking.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_ultrasoundviz.py

Todo:
    implement all the things!
"""

#import matplotlib.pyplot as plt
#import seaborn as sns
import cv2
import time


#from multisensorimport.dataobj import trialdata as td

READ_PATH = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/ultrasound_t5w1/'

def main():
    """Execute ultrasound image tracking visualization."""
    # set frame numbers of desired data
    ind_low = 1495
    ind_high = 2282

    for i in range(ind_low, ind_high+1):
        # execute image tracking
        filename = READ_PATH + str(i) + '.pgm'
        #print('reading from ' + filename)
        frame = cv2.imread(filename, -1)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
