#!/usr/bin/env python3
"""Example import of time series muscle data.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_multisensorimport.py

Todo:
    implement all the things!
"""

from multisensorimport.dataobj import trialdata as td

READ_PATH = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/seg_data_US.mat'


def main():
    """Execute all EMBC 2020 data analysis."""
    data1 = td.TrialData.from_preprocessed_mat_file(READ_PATH, 'Sub1', 1)
    print(data1.data_emg.data.shape)


if __name__ == "__main__":
    main()
