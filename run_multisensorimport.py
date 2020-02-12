#!/usr/bin/env python3
"""Example import of time series muscle data.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_multisensorimport.py

Todo:
    implement all the things!
"""

import matplotlib.pyplot as plt
import seaborn as sns

from multisensorimport.dataobj import trialdata as td
from multisensorimport.dataobj import data_utils as utils

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/'

READ_PATH_MAT = DATA_DIR + 'sub1/seg_data.mat'
READ_PATH_MAT_28 = DATA_DIR + 'sub2/seg_data.mat'
READ_PATH_MAT_33 = DATA_DIR + 'sub3/seg_data.mat'
READ_PATH_MAT_34 = DATA_DIR + 'sub4/seg_data.mat'
READ_PATH_MAT_37 = DATA_DIR + 'sub5/seg_data.mat'

READ_PATH_US_1 = DATA_DIR + 'sub1/wp1t5'
READ_PATH_US_2 = DATA_DIR + 'sub1/wp2t6'
READ_PATH_US_5 = DATA_DIR + 'sub1/wp5t11'
READ_PATH_US_8 = DATA_DIR + 'sub1/wp8t15'
READ_PATH_US_10 = DATA_DIR + 'sub1/wp10t25'
READ_PATH_US_28 = DATA_DIR + 'sub2/wp5t28'
READ_PATH_US_33 = DATA_DIR + 'sub3/wp5t33'
READ_PATH_US_34 = DATA_DIR + 'sub4/wp5t34'
READ_PATH_US_37 = DATA_DIR + 'sub5/wp5t37'

CORR_OUT_PATH = DATA_DIR + 'correlations.csv'

PLOT = True

def main():
    """Execute all EMBC 2020 data analysis."""

    # TODO: AMG peaks are inaccurate

    # s1wp1
    data1 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_1, 'sub1', 0,
                                                    emg_peak=5500,
                                                    amg_peak=13290,
                                                    force_peak=3721, us_peak=51)

    print(utils.calculate_pvalues(data1.df))
    data1.plot()
    raise ValueError("poor man's breakpoint")

    # s1wp2
    data2 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_2, 'sub1', 1,
                                                    emg_peak=5500,
                                                    amg_peak=13290,
                                                    force_peak=5800, us_peak=46)

#   raise ValueError("poor man's breakpoint")


    # s1wp5
    data5 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_5, 'sub1', 4,
                                                    emg_peak=5800,
                                                    amg_peak=13290,
                                                    force_peak=4476, us_peak=50)


    # s1wp8
    data8 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_8, 'sub1', 7,
                                                    emg_peak=5700,
                                                    amg_peak=13290,
                                                    force_peak=2469, us_peak=49)


    # s1wp10
    data10 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_10, 'sub1', 9,
                                                    emg_peak=6000,
                                                    amg_peak=13290,
                                                    force_peak=4222, us_peak=48)

    # s2wp5
    data28 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_28,
                                                    READ_PATH_US_28, 'sub2', 0,
                                                    emg_peak=None,
                                                    amg_peak=None,
                                                    force_peak=4033,
                                                    us_peak=53,
                                                    force_only=True)


    # s3wp5
    data33 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_33,
                                                    READ_PATH_US_33, 'sub3', 0,
                                                    emg_peak=None,
                                                    amg_peak=None,
                                                    force_peak=7113,
                                                    us_peak=63,
                                                    force_only=True)


    # s4wp5
    data34 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_34,
                                                    READ_PATH_US_34, 'sub4', 0,
                                                    emg_peak=None,
                                                    amg_peak=None,
                                                    force_peak=10810,
                                                    us_peak=72, force_only=True)


    # s5wp5
    data37 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_37,
                                                    READ_PATH_US_37, 'sub5', 0,
                                                    emg_peak=None,
                                                    amg_peak=None,
                                                    force_peak=0, us_peak=36,
                                                    force_only=True)


    utils.build_corr_table([data1, data2, data5, data8, data10, data28, data33,
                           data34, data37], CORR_OUT_PATH)

    if PLOT:
        data1.plot()
        data2.plot()
        data5.plot()
        data8.plot()
        data10.plot()
        data28.plot()
        data33.plot()
        data34.plot()
        data37.plot()

#    sns.set()
#
#    fig, axs = plt.subplots(4)
#    fig.suptitle('test plot')
#    axs[0].plot(data1.data_emg.data)
#    axs[1].plot(data1.data_amg.data)
#    axs[2].plot(data1.data_force.data)
#    axs[3].plot(data1.data_ultrasound.data)
#
#    plt.show()


if __name__ == "__main__":
    main()
