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

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/sub1/'

READ_PATH_MAT = DATA_DIR + 'seg_data.mat'

READ_PATH_US_1 = DATA_DIR + 'wp1t5'
READ_PATH_US_2 = DATA_DIR + 'wp2t6'
READ_PATH_US_5 = DATA_DIR + 'wp5t11'
READ_PATH_US_8 = DATA_DIR + 'wp8t15'
READ_PATH_US_10 = DATA_DIR + 'wp10t25'


def main():
    """Execute all EMBC 2020 data analysis."""

    # TODO: amg peaks

    # wp1
    data1 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_1, 'Sub1', 0,
                                                    emg_peak=5500,
                                                    amg_peak=13290,
                                                    force_peak=3721, us_peak=51)
#    raise ValueError("poor man's breakpoint")

    # wp2
    data2 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_2, 'Sub1', 1,
                                                    emg_peak=5500,
                                                    amg_peak=13290,
                                                    force_peak=5800, us_peak=46)
#    raise ValueError("poor man's breakpoint")


    # wp5
    data5 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_5, 'Sub1', 4,
                                                    emg_peak=5800,
                                                    amg_peak=13290,
                                                    force_peak=4476, us_peak=50)

    # wp8
    data8 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_8, 'Sub1', 7,
                                                    emg_peak=5700,
                                                    amg_peak=13290,
                                                    force_peak=2469, us_peak=49)

    # wp10
    data10 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_10, 'Sub1', 9,
                                                    emg_peak=6000,
                                                    amg_peak=13290,
                                                    force_peak=1460, us_peak=48)


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
