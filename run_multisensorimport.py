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

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/'

READ_PATH_MAT = DATA_DIR + 'seg_data_US.mat'

READ_PATH_US = DATA_DIR + 'wp1t5'


def main():
    """Execute all EMBC 2020 data analysis."""
    data1 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US, 'Sub1', 1,
                                                    emg_peak=5500,
                                                    amg_peak=13290,
                                                    force_peak=3721, us_peak=51)
    print(data1.data_emg.data.shape)

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
