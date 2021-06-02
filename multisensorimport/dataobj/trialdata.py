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
        df_{sus, ramp, steps, sin} (pandas.DataFrame): pandas dataframes
            containing time series data for only {sustained, ramp, "stair"
            step, sine} portions of the trial
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
        self.df_sus = None
        self.df_ramp = None
        self.df_steps = None
        self.df_sin = None

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

        # load trajectory start index (compensate for 500-sample offset +
        # starting baseline, accounting for some minor start time glitches in
        # particular data sets)
        if td.trial_no == '1b':
            if td.subj == '6':
                td.traj_start = trialdict['Traj-Changes'][1]-1100
            elif td.subj == '10': # also 9 w/ old version
                td.traj_start = trialdict['Traj-Changes'][1]-650
            else:
                td.traj_start = trialdict['Traj-Changes'][1]-700
        elif td.trial_no == '2b':
            if td.subj == '1':
                td.traj_start = trialdict['Traj-Changes'][1]-1000
                print(td.traj_start)
            elif td.subj == '4':
                td.traj_start = trialdict['Traj-Changes'][1]-800
            elif td.subj == '8':
                td.traj_start = trialdict['Traj-Changes'][1]-600
            else:
                td.traj_start = trialdict['Traj-Changes'][1]-700
        elif td.trial_no == '3b':
            if td.subj == '1':
                td.traj_start = trialdict['Traj-Changes'][1]-800
            elif td.subj == '3':
                td.traj_start = trialdict['Traj-Changes'][1]-800
            else:
                td.traj_start = trialdict['Traj-Changes'][1]-700
        else:
            raise ValueError('No offsets have been defined for this trial series.')

        # load data streams into frame

        times_raw = trialdict['Times'][td.traj_start:]
        start_time = int(times_raw[0])
        times = [t - start_time for t in times_raw]

        force_raw = pd.Series(data=trialdict['Raw'][2][0][td.traj_start:], index=times)
        us_raw = pd.Series(data=trialdict['Raw'][0][0][td.traj_start:], index=times)
        emg_raw = pd.Series(data=trialdict['Raw'][1][0][td.traj_start:], index=times)

        force_pro = pd.Series(data=trialdict['Processed'][2][td.traj_start:], index=times)
        us_pro = pd.Series(data=trialdict['Processed'][0][td.traj_start:], index=times)
        emg_pro = pd.Series(data=trialdict['Processed'][1][td.traj_start:], index=times)

        traj = pd.Series(data=trialdict['Trajs'][td.traj_start:], index=times)

        df_dict = {'force-raw': force_raw, 'us-raw': us_raw,
                   'emg-raw': emg_raw, 'force': force_pro,
                   'us': us_pro, 'emg': emg_pro, 'traj': traj}

        df = pd.DataFrame(df_dict)

        # full trial
        td.df = df[df.index < 86.0]

        # sustained
        td.df_sus = df[df.index < 36.0]

        # ramp
        df_ramp = df[df.index < 53.0]
        td.df_ramp = df_ramp[df_ramp.index > 36.0]

        # "stair" steps
        df_steps = df[df.index < 72.0]
        td.df_steps = df_steps[df_steps.index > 53.0]

        # sine
        df_sin = df[df.index < 86.0]
        td.df_sin = df_sin[df_sin.index > 72.0]

        return td

    def get_corrs(self, col):
        """Get correlation values between specified data column and force.

        Args:
            col (str): data column with which to compute correlation (generally
                'us' or 'emg')

        Returns:
            dictionary of correlations for each trial segment
        """
        corr_dict = {}
        corr_dict['corr-all'] = self.df.corr()['force'][col]
        corr_dict['corr-sus'] = self.df_sus.corr()['force'][col]
        corr_dict['corr-ramp'] = self.df_ramp.corr()['force'][col]
        corr_dict['corr-steps'] = self.df_steps.corr()['force'][col]
        corr_dict['corr-sin'] = self.df_sin.corr()['force'][col]

        return corr_dict

    def get_tracking_errors(self, tracked_val):
        """Get time series errors from trajectory tracking.

        Args:
            tracked_val (str): sensor used in trajectory tracking game
                (generally 'us' or 'emg', or possibly 'force')

        Returns:
            pandas.Dataframe time series errors
        """
        return abs(self.df[tracked_val] - self.df['traj'])

