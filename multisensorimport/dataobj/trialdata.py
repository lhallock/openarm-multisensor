#!/usr/bin/env python3
"""Class containing all muscle data for a single experimental trial.

This module contains the TrialData class, which contains all time series data
associated with a particular trial and associated metadata. Specifically, this
class is used to aggregate associated surface electromyography (sEMG), acoustic
myography (AMG), force, and quantitative features associated with time series
ultrasound (e.g., muscle cross-sectional area, or CSA), alongside subject
metadata.
"""
import re
import pickle5 as pickle
import math

import numpy as np
import pandas as pd
from scipy.io import loadmat

import multisensorimport.dataobj.data_utils as utils
from multisensorimport.dataobj.timeseriesdata import TimeSeriesData

# MAGIC NUMBERS (fungible)
EMG_EWM_SPAN = 500
FORCE_DETREND_CUTOFF = 5.0
POLYNOMIAL_ORDER = 3
PREPEAK_VAL = 3
INTERPOLATION_METHOD = 'linear'


class TrialData():
    """Class containing muscle time series data and associated metadata.

    Attributes:
        subj (str): subject identifier
        trial_no (str): trial number and type identifier
        mins (dict): dictionary of minimum signal values
        maxs (dict): dictionary of maximum signal values
        traj_start (int): index of trajectory start
        df (pandas.DataFrame): pandas dataframe containing all timesynced data
            streams, including force, ultrasound-measured muscle thickness,
            sEMG, and plotted goal trajectory, both raw and processed
    """

    def __init__(self):
        """Standard initializer for TrialData objects.

        Returns:
            (empty) TrialData object
        """
        self.subj = None
        self.trial_no = None
        self.mins = None
        self.maxs = None
        self.traj_start = None
        self.df = None

    @classmethod
    def from_pickle(cls, filename, subj):
        """Initialize TrialData object from pickle generated during collection.

        Args:
            filename (str): path to pickle .p data file
            subj (str): desired subject identifier

        Returns:
            TrialData object containing data from file
        """
        td = cls()
        td.subj = subj

        # parse trial identifier from filename
        td.trial_no = re.split('_|\.', filename)[-2]

        # extract data from pickle
        trialdict = np.load(filename, allow_pickle=True)

        # load mins
        td.mins = {}
        td.mins['force'] = trialdict['Processed-Mins'][2]
        td.mins['us'] = trialdict['Processed-Mins'][0]
        td.mins['emg'] = trialdict['Processed-Mins'][1]

        # load maxs
        td.maxs = {}
        td.maxs['force'] = trialdict['Processed-Maxs'][2]
        td.maxs['us'] = trialdict['Processed-Maxs'][0]
        td.maxs['emg'] = trialdict['Processed-Maxs'][1]

        # load trajectory start index
        td.traj_start = trialdict['Traj-Changes'][1]

        # load data streams into frame

        times = trialdict['Times']

        force_raw = pd.Series(data=trialdict['Raw'][2][0], index=times)
        us_raw = pd.Series(data=trialdict['Raw'][0][0], index=times)
        emg_raw = pd.Series(data=trialdict['Raw'][1][0], index=times)

        force_pro = pd.Series(data=trialdict['Processed'][2][0], index=times)
        us_pro = pd.Series(data=trialdict['Processed'][0][0], index=times)
        emg_pro = pd.Series(data=trialdict['Processed'][1][0], index=times)

        traj = pd.Series(data=trialdict['Trajs'], index=times)

        df_dict = {'force-raw': force_raw, 'us-raw': us_raw,
                   'emg-raw': emg_raw, 'force': force_pro,
                   'us': us_pro, 'emg': emg_pro, 'traj': traj}

        td.df = pd.DataFrame(df_dict)

        return td

