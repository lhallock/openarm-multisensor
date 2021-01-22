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
        trial_no (int): trial number
        force_only (bool): whether data series contains only force and
            ultrasound data (i.e., no sEMG or AMG)
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
        data_us_csa (TimeSeriesData): ultrasound-extracted time series data of
            muscle cross-sectional area, extracted to CSV using tracking module
        data_us_csa_tracked (TimeSeriesData): ultrasound-extracted time series
            data of muscle cross-sectional area, computed via tracking
        data_us_thickness (TimeSeriesData): ultrasound-extracted time series
            data of muscle thickness, extracted to CSV using tracking module
        data_us_thickness_tracked (TimeSeriesData): ultrasound-extracted time
            series data of muscle thickness, computed via tracking
        data_us_th_rat (TimeSeriesData): ultrasound-extracted time series data
            of maximum length/width ratio, extracted to CSV using tracking
            module
        data_us_th_rat_tracked (TimeSeriesData): ultrasound-extracted time
            series data of maximum length/width ratio, computed via tracking
        data_us_jd_error (TimeSeriesData): ultrasound-extracted time series
            tracking error, as 1-IoU (Jaccard Distance)
        df (pandas.DataFrame): pandas dataframe containing all timesynced data
            streams
        df_dt (pandas.DataFrame): truncated version of df containing only
            values for which force is below FORCE_DETREND_CUTOFF
    """

    def __init__(self):
        """Standard initializer for TrialData objects.

        Returns:
            (empty) TrialData object
        """
        self.subj = None
        self.trial_no = None
        self.force_only = None
        self.wp = None
        self.ang = None
        self.data_emg = None
        self.data_amg = None
        self.data_force = None
        self.data_force_abs = None
        self.data_us_csa = None
        self.data_us_csa_tracked = None
        self.data_us_thickness = None
        self.data_us_thickness_tracked = None
        self.data_us_th_rat = None
        self.data_us_th_rat_tracked = None
        self.data_us_jd_error = None
        self.df = None
        self.df_dt = None

    @classmethod
    def from_preprocessed_mat_file(cls,
                                   filename_mat,
                                   filedir_us,
                                   subj,
                                   struct_no,
                                   emg_peak=0,
                                   amg_peak=0,
                                   force_peak=0,
                                   us_peak=0,
                                   force_only=False,
                                   tracking_data_type=None):
        """Initialize TrialData object from specialized MATLAB .mat file.

        This initializer is designed for use with publication-specific
        preformatted MATLAB .mat files used in prior data analysis.

        Args:
            filename_mat (str): path to MATLAB .mat data file
            filedir_us (str): path to ultrasound .csv data directory
            subj (str): desired subject identifier
            struct_no (int): cell to access within the saved MATLAB struct
                (NOTE: zero-indexed, e.g., 0-12)
            {emg,amg,force,us}_peak (int): location of first peak in each data
                series, used for alignment across series (specified as index
                within array)
            force_only (bool): build frame using force and ultrasound data only
            tracking_data_type (str): what type of tracking data should be
                imported, if any (options are 'LK', 'FRLK', 'BFLK-G',
                'BFLK-T', 'SBLK-G', and 'SBLK-T')

        Returns:
            TrialData object containing data from file
        """
        td = cls()
        td.subj = subj
        td.force_only = force_only

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

        if not force_only:

            # set EMG data
            emg_data = data_struct['filtEmg'][0, 0]
            emg_labels = ['forearm', 'biceps', 'triceps', 'NONE']
            emg_freq = 1000
            emg_offset = utils.offset_from_peak(emg_peak, emg_freq, PREPEAK_VAL)
            td.data_emg = TimeSeriesData.from_array('sEMG', emg_data,
                                                    emg_labels, emg_freq,
                                                    emg_offset)

            # set AMG data
            amg_data = data_struct['rawAmg'][0, 0]
            amg_labels = [
                'forearm (front/wrist)', 'forearm (back)', 'biceps', 'triceps'
            ]
            amg_freq = 2000
            amg_offset = utils.offset_from_peak(amg_peak, amg_freq, PREPEAK_VAL)
            td.data_amg = TimeSeriesData.from_array('AMG', amg_data, amg_labels,
                                                    amg_freq, amg_offset)

        # set force data
        force_data = data_struct['forceHandle'][0, 0]
        force_labels = ['F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']
        force_freq = 2400
        force_offset = utils.offset_from_peak(force_peak, force_freq,
                                              PREPEAK_VAL)
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

        # set ultrasound CSA data
        us_csa_labels = ['CSA']
        us_freq = 8.3856  # empirical calculation
        us_offset = utils.offset_from_peak(us_peak, us_freq, PREPEAK_VAL)
        filename_us_csa = filedir_us + '/ground_truth_csa.csv'
        td.data_us_csa = TimeSeriesData.from_file('US-CSA', filename_us_csa,
                                                  us_csa_labels, us_freq,
                                                  us_offset)
        if tracking_data_type:
            filename_us_csa_tracked = filedir_us + '/' + tracking_data_type + '/tracking_csa.csv'
            td.data_us_csa_tracked = TimeSeriesData.from_file(
                'US-CSA-T', filename_us_csa_tracked, us_csa_labels, us_freq,
                us_offset)

        # set ultrasound thickness data
        us_t_labels = ['T']
        filename_us_t = filedir_us + '/ground_truth_thickness.csv'
        td.data_us_thickness = TimeSeriesData.from_file('US-T', filename_us_t,
                                                        us_t_labels, us_freq,
                                                        us_offset)
        if tracking_data_type:
            filename_us_t_tracked = filedir_us + '/' + tracking_data_type + '/tracking_thickness.csv'
            td.data_us_thickness_tracked = TimeSeriesData.from_file(
                'US-T-T', filename_us_t_tracked, us_t_labels, us_freq,
                us_offset)

        # set ultrasound thickness ratio data
        us_tr_labels = ['TR']
        filename_us_tr = filedir_us + '/ground_truth_thickness_ratio.csv'
        td.data_us_th_rat = TimeSeriesData.from_file('US-TR', filename_us_tr,
                                                     us_tr_labels, us_freq,
                                                     us_offset)
        if tracking_data_type:
            filename_us_tr_tracked = filedir_us + '/' + tracking_data_type + '/tracking_thickness_ratio.csv'
            td.data_us_th_rat_tracked = TimeSeriesData.from_file(
                'US-TR-T', filename_us_tr_tracked, us_tr_labels, us_freq,
                us_offset)

            # set ultrasound IoU error
            us_jd_labels = ['JD']
            filename_jd_error = filedir_us + '/' + tracking_data_type + '/iou_series.csv'
            td.data_us_jd_error = TimeSeriesData.from_file(
                'US-JD-E', filename_jd_error, us_jd_labels, us_freq, us_offset)

        td.build_synced_dataframe()

        return td

    def build_synced_dataframe(self):
        """Build pandas dataframe with all data processed and synced.

        This method builds a pandas dataframe in which all desired time series
        data is correctly aligned and sampled at the same specified frequency
        (generally, the lowest sampling frequency of the comprising data
        series).
        """
        # build ground truth ultrasound data series
        us_csa_series = utils.build_data_series(self.data_us_csa)
        us_t_series = utils.build_data_series(self.data_us_thickness)
        us_tr_series = utils.build_data_series(self.data_us_th_rat)

        # build tracked ultrasound data series
        us_csa_series_tracked = utils.build_data_series(
            self.data_us_csa_tracked)
        us_t_series_tracked = utils.build_data_series(
            self.data_us_thickness_tracked)
        us_tr_series_tracked = utils.build_data_series(
            self.data_us_th_rat_tracked)
        us_jd_error_series = utils.build_data_series(self.data_us_jd_error)

        # change ultrasound data series to appropriate units (pixel --> mm^2)
        us_res = 0.157676
        us_csa_series = us_csa_series.multiply(math.pow(us_res, 2))
        us_csa_series_tracked = us_csa_series_tracked.multiply(
            math.pow(us_res, 2))
        us_t_series = us_t_series.multiply(us_res)
        us_t_series_tracked = us_t_series_tracked.multiply(us_res)

        # compute Jaccard Distance from IoU series
        us_jd_error_series = 1 - us_jd_error_series

        # build ultrasound dataframe
        us_series_dict = {
            'us-csa': us_csa_series,
            'us-csa-t': us_csa_series_tracked,
            'us-t': us_t_series,
            'us-t-t': us_t_series_tracked,
            'us-tr': us_tr_series,
            'us-tr-t': us_tr_series_tracked,
            'us-jd-e': us_jd_error_series
        }
        df_us = pd.DataFrame(us_series_dict)

        # crop out zero values (i.e., where detection failed)
        df_us = df_us.loc[df_us['us-csa'] > 0]

        # build force data series
        force_series = utils.build_data_series(self.data_force_abs)

        if not self.force_only:
            # build EMG data series
            emg_series_brd = utils.build_data_series(self.data_emg, 0)
            emg_abs_series_brd = emg_series_brd.abs().ewm(
                span=EMG_EWM_SPAN).mean()
            emg_series_bic = utils.build_data_series(self.data_emg, 1)
            emg_abs_series_bic = emg_series_bic.abs().ewm(
                span=EMG_EWM_SPAN).mean()

        # combine all series into dataframe
        us_csa_series_nz = df_us['us-csa']
        us_csa_series_tracked_nz = df_us['us-csa-t']
        us_t_series_nz = df_us['us-t']
        us_t_series_tracked_nz = df_us['us-t-t']
        us_tr_series_nz = df_us['us-tr']
        us_tr_series_tracked_nz = df_us['us-tr-t']
        us_jd_error_series_nz = df_us['us-jd-e']

        if not self.force_only:
            series_dict = {
                'force': force_series,
                'emg-brd': emg_series_brd,
                'emg-abs-brd': emg_abs_series_brd,
                'emg-bic': emg_series_bic,
                'emg-abs-bic': emg_abs_series_bic,
                'us-csa': us_csa_series_nz,
                'us-csa-t': us_csa_series_tracked_nz,
                'us-t': us_t_series_nz,
                'us-t-t': us_t_series_tracked_nz,
                'us-tr': us_tr_series_nz,
                'us-tr-t': us_tr_series_tracked_nz,
                'us-jd-e': us_jd_error_series_nz
            }
        else:
            series_dict = {
                'force': force_series,
                'us-csa': us_csa_series_nz,
                'us-csa-t': us_csa_series_tracked_nz,
                'us-t': us_t_series_nz,
                'us-t-t': us_t_series_tracked_nz,
                'us-t': us_t_series_nz,
                'us-tr': us_tr_series_nz,
                'us-tr-t': us_tr_series_tracked_nz,
                'us-jd-e': us_jd_error_series_nz
            }

        df = pd.DataFrame(series_dict)

        # truncate data series to the same length
        if not self.force_only:
            min_time_completed = min(max(us_csa_series.index),
                                     max(force_series.index),
                                     max(emg_series_brd.index))
        else:
            min_time_completed = min(max(us_csa_series.index),
                                     max(force_series.index))

        df = df.truncate(after=min_time_completed)

        # interpolate values
        df = df.interpolate(method=INTERPOLATION_METHOD)

        # create df for detrending
        df_dt = df.loc[df['force'] <= FORCE_DETREND_CUTOFF]

        # generate polynomial fits
        us_csa_fitdata = utils.fit_data_poly(df_dt.index, df_dt['us-csa'],
                                             df.index, POLYNOMIAL_ORDER)
        us_t_fitdata = utils.fit_data_poly(df_dt.index, df_dt['us-t'], df.index,
                                           POLYNOMIAL_ORDER)
        us_tr_fitdata = utils.fit_data_poly(df_dt.index, df_dt['us-tr'],
                                            df.index, POLYNOMIAL_ORDER)

        # add polyfits to data frame
        df['us-csa-fit'] = us_csa_fitdata
        df['us-t-fit'] = us_t_fitdata
        df['us-tr-fit'] = us_tr_fitdata

        # add detrended data to data frame
        df['us-csa-dt'] = df['us-csa'] - df['us-csa-fit']
        df['us-t-dt'] = df['us-t'] - df['us-t-fit']
        df['us-tr-dt'] = df['us-tr'] - df['us-tr-fit']

        # add error columns to data frame
        df['us-csa-e'] = abs(df['us-csa-t'] - df['us-csa']) / df['us-csa']
        df['us-t-e'] = abs(df['us-t-t'] - df['us-t']) / df['us-t']
        df['us-tr-e'] = abs(df['us-tr-t'] - df['us-tr']) / df['us-tr']

        # set object values
        self.df = df
        self.df_dt = df_dt

    def _compute_abs_force(self):
        """Compute absolute value of force from 6-channel force data.

        Returns:
           numpy.ndarray absolute force time series
        """
        force_data_comps = self.data_force.data
        force_abs = np.zeros((force_data_comps.shape[0], 1))
        for i in range(force_abs.shape[0]):
            x_i = force_data_comps[i, 0]
            y_i = force_data_comps[i, 1]
            z_i = force_data_comps[i, 2]

            force_abs[i] = math.sqrt(
                math.pow(x_i, 2) + math.pow(y_i, 2) + math.pow(z_i, 2))

        return force_abs
