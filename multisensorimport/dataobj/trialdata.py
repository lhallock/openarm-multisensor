#!/usr/bin/env python3
"""Class containing all muscle data for a single experimental trial.

This module contains the TrialData class, which contains all time series data
associated with a particular trial and associated metadata. Specifically, this
class is used to aggregate associated surface electromyography (sEMG), acoustic
myography (AMG), force, and quantitative features associated with time series
ultrasound (e.g., muscle cross-sectional area, or CSA), alongside subject
metadata.

"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #TODO: won't need these after refactor
import seaborn as sns
from scipy.io import loadmat

from multisensorimport.dataobj.timeseriesdata import TimeSeriesData


class TrialData():
    """Class containing muscle time series data and associated metadata.

    Attributes:
        subj (str): subject identifier
        trial_no (int): trial number
        wp (int): kinematic waypoint (corresponds directly to elbow angle)
        ang (int): elbow angle in degrees of flexion (i.e., from "full
            flexion", which is technically impossible)
        data_emg (TimeSeriesData): sEMG data, pre-normalized and filtered with
            60Hz notch filter
        data_amg (TimeSeriesData): AMG data
        data_force (TimeSeriesData): force data, pre-converted to wrench at
            robot handle
        data_force_abs (TimeSeriesData): 1D force data containing absolute
            value of output force only
        data_ultrasound (TimeSeriesData): ultrasound time series data,
            extracted using FUNCTION TODO
    """

    def __init__(self):
        """Standard initializer for TrialData objects.

        Returns:
            (empty) TrialData object
        """
        self.subj = None
        self.trial_no = None
        self.wp = None
        self.ang = None
        self.data_emg = None
        self.data_amg = None
        self.data_force = None
        self.data_force_abs = None
        self.data_ultrasound = None

    @classmethod
    def from_preprocessed_mat_file(cls, filename_mat, filename_us, subj, struct_no):
        """Initialize TrialData object from specialized MATLAB .mat file.

        This initializer is designed for use with publication-specific
        preformatted MATLAB .mat files used in prior data analysis.

        Args:
            filename_mat (str): path to MATLAB .mat data file
            filename_us (str): path to ultrasound .csv data file
            subj (str): desired subject identifier
            struct_no (int): cell to access within the saved MATLAB struct
                (NOTE: zero-indexed, e.g., 0-12)

        Returns:
            TrialData object containing data from file
        """
        td = cls()
        td.subj = subj

        # define trial numbers and angle values for each waypoint (measured
        # during data collection)
        wp_to_trial = {
            '1': '5',
            '2': '6',
            '3': '8',
            '4': '9',
            '5': '11',
            '6': '24',
            '7': '14',
            '8': '15',
            '9': '16',
            '10': '25',
            '11': '20',
            '12': '21',
            '13': '23'
        }
        wp_to_angle = {
            '1': '155',
            '2': '136',
            '3': '132',
            '4': '119',
            '5': '111',
            '6': '113',
            '7': '107',
            '8': '98',
            '9': '90',
            '10': '83',
            '11': '78',
            '12': '73',
            '13': '63'
        }

        # load data from .mat file
        data = loadmat(filename_mat)
        data_struct = data['data'][0, struct_no]

        # set waypoint and dependent values
        td.wp = int(data_struct['wp'])
        td.trial_no = wp_to_trial[str(td.wp)]
        td.ang = wp_to_angle[str(td.wp)]

        # set EMG data
        emg_data = data_struct['filtEmg'][0, 0]
        emg_labels = ['forearm', 'biceps', 'triceps', 'NONE']
        emg_freq = 1000
        emg_offset = 0
        td.data_emg = TimeSeriesData.from_array('sEMG', emg_data, emg_labels,
                                                emg_freq, emg_offset)

        # set AMG data
        amg_data = data_struct['rawAmg'][0, 0]
        amg_labels = [
            'forearm (front/wrist)', 'forearm (back)', 'biceps', 'triceps'
        ]
        amg_freq = 2000
        amg_offset = 0 #TODO: calculate these
        td.data_amg = TimeSeriesData.from_array('AMG', amg_data, amg_labels,
                                                amg_freq, amg_offset)

        # set force data
        force_data = data_struct['forceHandle'][0, 0]
        force_labels = ['F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']
        force_freq = 2400
        force_offset = 0
        td.data_force = TimeSeriesData.from_array('force', force_data,
                                                  force_labels, force_freq,
                                                  force_offset)

        # set absolute value force data
        force_abs_data = td._compute_abs_force()
        force_abs_labels = ['Net Force']
        td.data_force_abs = TimeSeriesData.from_array('force_abs',
                                                      force_abs_data,
                                                      force_abs_labels,
                                                      force_freq, force_offset)


        # set ultrasound data
        us_labels = ['CSA']
        us_freq = 8.3856 #TODO: note on empirical calculation
        us_offset = 0
        td.data_ultrasound = TimeSeriesData.from_file('US', filename_us,
                                                      us_labels, us_freq,
                                                      us_offset)

        td.build_synced_dataframe(us_freq)

        return td

    def build_synced_dataframe(self, new_freq):
        """Build pandas dataframe with all data processed and synced.

        This method builds a pandas dataframe in which all desired time series
        data is correctly aligned and sampled at the same specified frequency
        (generally, the lowest sampling frequency of the comprising data
        series).

        Args:
            new_freq (float): desired sampling frequency for data series, in Hz

        Returns:
            pandas dataframe with publication-relevant data streams
        """
        # build force data series
        force_len = self.data_force.data_from_offset.shape[0]
        force_freq_pd_str = self.as_pd_freq(self.data_force.freq)
        force_index = pd.timedelta_range(0, periods=force_len,
                                         freq=force_freq_pd_str)
        force_series = pd.Series(self.data_force_abs.data_from_offset[:, 0], force_index)
        print(force_series)

        # build ultrasound data series
        us_len = self.data_ultrasound.data_from_offset.shape[0]
        us_freq_pd_str = self.as_pd_freq(self.data_ultrasound.freq)
        us_index = pd.timedelta_range(0, periods=us_len, freq=us_freq_pd_str)
        us_series = pd.Series(self.data_ultrasound.data_from_offset[:, 0], us_index)
        print(us_series)

        # build EMG data series
        emg_len = self.data_emg.data.shape[0]
        emg_freq_pd_str = self.as_pd_freq(self.data_emg.freq)
        emg_index = pd.timedelta_range(0, periods=emg_len, freq=emg_freq_pd_str)
        emg_series = pd.Series(self.data_emg.data_from_offset[:, 1], emg_index)
        #TODO: choose forearm (0) or biceps (1)
        print(emg_series)

        sns.set()

        fig, axs = plt.subplots(4)
        fig.suptitle('test pandas plot')
        axs[0].plot(force_series)
        axs[1].plot(emg_series)
        axs[2].plot(us_series)
        axs[3].plot(us_series)

        plt.show()


    def as_pd_freq(self, freq):
        """Convert frequency expressed in Hz to pandas-compatible frequency.

        When pandas says they want frequency, what they're actually asking for
        is period, which is stupid and confusing. This method converts a
        frequency expressed in Hz to its equivalent pandas frequency, aka its
        period in microseconds. WARNING: Don't use this with data sample at
        >1MHz, since this will run into precision errors!

        Args:
            freq (float): data series frequency (in Hz), <1e6

        Returns:
            str pandas frequency corresponding to freq

        Raises:
            ValueError if freq is >1e6
        """
        if freq > 1e6:
            raise ValueError('Specified frequency is too high for this method',
                             'and will result in catastrophic precision loss.')

        freq_pd = int(1e6/freq)
        freq_pd_str = str(freq_pd) + 'U'
        return freq_pd_str


    def _compute_abs_force(self):
        force_data_comps = self.data_force.data
        force_abs = np.zeros((force_data_comps.shape[0], 1))
        for i in range(force_abs.shape[0]):
            x_i = force_data_comps[i, 0]
            y_i = force_data_comps[i, 1]
            z_i = force_data_comps[i, 2]

            force_abs[i] = math.sqrt(math.pow(x_i, 2)+math.pow(y_i, 2)+math.pow(z_i, 2))

        return force_abs













