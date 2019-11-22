#!/usr/bin/env python3
"""Class containing all muscle data for a single experimental trial.

This module contains the TrialData class, which contains all time series data
associated with a particular trial and associated metadata. Specifically, this
class is used to aggregate associated surface electromyography (sEMG), acoustic
myography (AMG), force, and quantitative features associated with time series
ultrasound (e.g., muscle cross-sectional area, or CSA), alongside subject
metadata.

"""

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
        self.data_ultrasound = None

    @classmethod
    def from_preprocessed_mat_file(cls, filename, subj, struct_no):
        """Initialize TrialData object from specialized MATLAB .mat file.

        This initializer is designed for use with publication-specific
        preformatted MATLAB .mat files used in prior data analysis.

        Args:
            filename(str): path to MATLAB .mat data file
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
        data = loadmat(filename)
        data_struct = data['data'][0, struct_no]

        # set waypoint and dependent values
        td.wp = int(data_struct['wp'])
        td.trial_no = wp_to_trial[str(td.wp)]
        td.ang = wp_to_angle[str(td.wp)]

        # set EMG data
        emg_data = data_struct['filtEmg'][0, 0]
        emg_labels = ['forearm', 'biceps', 'triceps', 'NONE']
        emg_freq = 1000
        td.data_emg = TimeSeriesData.from_array('sEMG', emg_data, emg_labels,
                                                emg_freq)

        # set AMG data
        amg_data = data_struct['rawAmg'][0, 0]
        amg_labels = [
            'forearm (front/wrist)', 'forearm (back)', 'biceps', 'triceps'
        ]
        amg_freq = 2000
        td.data_amg = TimeSeriesData.from_array('AMG', amg_data, amg_labels,
                                                amg_freq)

        # set force data
        force_data = data_struct['forceHandle'][0, 0]
        force_labels = ['F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']
        force_freq = 2400
        td.data_force = TimeSeriesData.from_array('force', force_data,
                                                  force_labels, force_freq)

        # TODO: set ultrasound data
        td.data_ultrasound = None

        return td
