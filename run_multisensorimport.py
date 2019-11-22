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

READ_PATH = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/seg_data_US.mat'


def main():
    """Execute all EMBC 2020 data analysis."""
    data1 = td.TrialData.from_preprocessed_mat_file(READ_PATH, 'Sub1', 1)
    print(data1.data_emg.data.shape)

    sns.set()

    fig, axs = plt.subplots(3)
    fig.suptitle('test plot')
    axs[0].plot(data1.data_emg.data)
    axs[1].plot(data1.data_amg.data)
    axs[2].plot(data1.data_force.data)

    plt.show()


if __name__ == "__main__":
    main()
