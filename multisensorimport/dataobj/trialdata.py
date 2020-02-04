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
#from numpy.polynomial import Polynomial as Poly
import numpy.polynomial.polynomial as poly
#from sklearn.preprocessing import PolynomialFeatures

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
        data_us_csa (TimeSeriesData): ultrasound-extracted time series data of
            muscle cross-sectional area, extracted using FUNCTION TODO
        data_us_thickness (TimeSeriesData): ultrasound-extracted time series
            data of muscle thickness, extracted using FUNCTION TODO
        data_us_th_rat (TimeSeriesData): ultrasound-extracted time series data
            of maximum length/width ration, extracted using FUNCTION TODO
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
        self.data_us_csa = None
        self.data_us_thickness = None
        self.data_us_th_rat = None

    @classmethod
    def from_preprocessed_mat_file(cls, filename_mat, filedir_us, subj,
                                   struct_no, emg_peak=0, amg_peak=0,
                                   force_peak=0, us_peak=0, force_only=False):
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

        if not force_only:

            # set EMG data
            emg_data = data_struct['filtEmg'][0, 0]
            emg_labels = ['forearm', 'biceps', 'triceps', 'NONE']
            emg_freq = 1000
            emg_offset = td.offset_from_peak(emg_peak, emg_freq)
            td.data_emg = TimeSeriesData.from_array('sEMG', emg_data,
                                                    emg_labels, emg_freq,
                                                    emg_offset)

            # set AMG data
            amg_data = data_struct['rawAmg'][0, 0]
            amg_labels = ['forearm (front/wrist)', 'forearm (back)', 'biceps',
                          'triceps']
            amg_freq = 2000
            amg_offset = td.offset_from_peak(amg_peak, amg_freq)
            td.data_amg = TimeSeriesData.from_array('AMG', amg_data, amg_labels,
                                                    amg_freq, amg_offset)

        # set force data
        force_data = data_struct['forceHandle'][0, 0]
        force_labels = ['F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']
        force_freq = 2400
        force_offset = td.offset_from_peak(force_peak, force_freq)
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
        us_freq = 8.3856 #TODO: note on empirical calculation
        us_offset = td.offset_from_peak(us_peak, us_freq)
        filename_us_csa = filedir_us + '/ground_truth_csa.csv'
        td.data_us_csa = TimeSeriesData.from_file('US-CSA', filename_us_csa,
                                                      us_csa_labels, us_freq,
                                                      us_offset)

        # set ultrasound thickness data
        us_t_labels = ['T']
        filename_us_t = filedir_us + '/ground_truth_thickness.csv'
        td.data_us_thickness = TimeSeriesData.from_file('US-T', filename_us_t,
                                                      us_t_labels, us_freq,
                                                      us_offset)

        # set ultrasound thickness ratio data
        us_tr_labels = ['TR']
        filename_us_tr = filedir_us + '/ground_truth_thickness_ratio.csv'
        td.data_us_th_rat = TimeSeriesData.from_file('US-TR', filename_us_tr,
                                                      us_tr_labels, us_freq,
                                                      us_offset)

        td.build_synced_dataframe(force_only)

        return td

    def build_synced_dataframe(self, force_only):
        """Build pandas dataframe with all data processed and synced.

        This method builds a pandas dataframe in which all desired time series
        data is correctly aligned and sampled at the same specified frequency
        (generally, the lowest sampling frequency of the comprising data
        series).

        Args:
            force_only (bool): whether only force and ultrasound data should be
                computed
        Returns:
            pandas dataframe with publication-relevant data streams
        """
        # build ultrasound data series
        us_csa_series = self.build_data_series(self.data_us_csa)
        us_t_series = self.build_data_series(self.data_us_thickness)
        us_tr_series = self.build_data_series(self.data_us_th_rat)

        # change ultrasound data series to appropriate units (pixel --> mm^2)
        us_res = 0.157676
        us_csa_series = us_csa_series.multiply(math.pow(us_res, 2))
        us_t_series = us_t_series.multiply(us_res)

        # build ultrasound dataframe
        us_series_dict = {'us-csa': us_csa_series, 'us-t': us_t_series,
                          'us-tr': us_tr_series}
        df_us = pd.DataFrame(us_series_dict)

        # crop out zero values (i.e., where detection failed)
        df_us = df_us.loc[df_us['us-csa'] > 0]

        # build force data series
        force_series = self.build_data_series(self.data_force_abs)

        if not force_only:
            # build EMG data series
            emg_series = self.build_data_series(self.data_emg, 1)
            #TODO: choose forearm (0) or biceps (1)
            emg_abs_series = emg_series.abs().ewm(span=500).mean()

        # combine all series into dataframe
        us_csa_series_nz = df_us['us-csa']
        us_t_series_nz = df_us['us-t']
        us_tr_series_nz = df_us['us-tr']

        if not force_only:
            series_dict = {'force': force_series, 'emg': emg_series, 'emg-abs':
                           emg_abs_series, 'us-csa': us_csa_series_nz, 'us-t':
                           us_t_series_nz, 'us-tr': us_tr_series_nz}
        else:
            series_dict = {'force': force_series, 'us-csa': us_csa_series_nz,
                           'us-t': us_t_series_nz, 'us-tr': us_tr_series_nz}

        df = pd.DataFrame(series_dict)

        # truncate data series to the same length
        if not force_only:
            min_time_completed = min(max(us_csa_series.index),
                                     max(force_series.index),
                                     max(emg_series.index))
        else:
            min_time_completed = min(max(us_csa_series.index),
                                     max(force_series.index))

        df = df.truncate(after=min_time_completed)
        print(df)

        # interpolate values
        df = df.interpolate(method='linear')
        #TODO: choose interpolation method/order
        print(df)

        # create df for detrending
        df_dt = df.loc[df['force'] <= 5.0]

        # create time index for fitting
        x_times_pd = df_dt.index.to_julian_date()-2457780
#        x_times_test = df_dt.index.astype(float)
        print(x_times_pd)
#        print(x_times_test)
        x_times = x_times_pd.to_numpy()

        # series to fit
        y_us_csa = df_dt['us-csa'].to_numpy()
        y_us_t = df_dt['us-t'].to_numpy()
        y_us_tr = df_dt['us-tr'].to_numpy()

        # calculate polynomial fit
        p_order = 30
        p_us_csa_coeffs = poly.polyfit(x_times, y_us_csa, p_order)
        p_us_t_coeffs = poly.polyfit(x_times, y_us_t, p_order)
        p_us_tr_coeffs = poly.polyfit(x_times, y_us_tr, p_order)
        print(p_us_csa_coeffs)

#        p_us_csa = np.poly1d(np.polyfit(x_times, y_us_csa, p_order))
#        p_us_t = np.poly1d(np.polyfit(x_times, y_us_t, p_order))
#        p_us_tr = np.poly1d(np.polyfit(x_times, y_us_tr, p_order))
#        p_us_csa = Poly.fit(x_times, y_us_csa, p_order)
#        print(p_us_csa)
#        p_us_t = Poly.fit(x_times, y_us_t, p_order)
#        p_us_tr = Poly.fit(x_times, y_us_tr, p_order)


        # generate polyfit line size of untruncated data frame
        x_times_full = df.index.to_julian_date().to_numpy()-2457780
        us_csa_fitdata = poly.polyval(x_times_full, p_us_csa_coeffs)
        us_t_fitdata = poly.polyval(x_times_full, p_us_t_coeffs)
        us_tr_fitdata = poly.polyval(x_times_full, p_us_tr_coeffs)
#        us_csa_fitdata = p_us_csa(x_times_full)
#        us_t_fitdata = p_us_t(x_times_full)
#        us_tr_fitdata = p_us_tr(x_times_full)
#        us_csa_fitdata = Poly.polyval(x_times_full, p_us_csa)
#        us_t_fitdata = Poly.polyval(x_times_full, p_us_t)
#        us_tr_fitdata = Poly.polyval(x_times_full, p_us_tr)

        # add polyfits to data frame
        df['us-csa-fit'] = us_csa_fitdata
        df['us-t-fit'] = us_t_fitdata
        df['us-tr-fit'] = us_tr_fitdata

        # add detrended data to data frame
        df['us-csa-dt'] = df['us-csa'] - df['us-csa-fit']
        df['us-t-dt'] = df['us-t'] - df['us-t-fit']
        df['us-tr-dt'] = df['us-tr'] - df['us-tr-fit']

        # try thickness squared too
        df['us-t-dt-sq'] = df['us-t-dt'].pow(2)

        df_corr = df.corr()
        print(df_corr)
        corr_out_path = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/' + self.subj + '/wp' + str(self.wp) + '.csv'
        df_corr.to_csv(corr_out_path)


        # TO REMOVE
#        df = df_dt.copy()


        sns.set()

        tstring = self.subj + ' test plot, wp' + str(self.wp)

        if not force_only:
            fig, axs = plt.subplots(7)
            fig.suptitle(tstring)
            axs[0].plot(df['force'])
            axs[0].set(ylabel='force')
            axs[1].plot(df['emg'])
            axs[1].set(ylabel='emg')
            axs[2].plot(df['emg-abs'])
            axs[2].set(ylabel='emg-abs')
            axs[3].plot(df['us-csa'])
            axs[3].set(ylabel='us-csa')
            axs[3].plot(df_dt['us-csa'], 'g-')
            axs[3].plot(df['us-csa-fit'], 'r-')
            axs[4].plot(df['us-t'])
            axs[4].set(ylabel='us-t')
            axs[4].plot(df_dt['us-t'], 'g-')
            axs[4].plot(df['us-t-fit'], 'r-')
            axs[5].plot(df['us-t-dt'])
            axs[5].set(ylabel='us-t-dt')
            axs[6].plot(df['us-tr'])
            axs[6].set(ylabel='us-tr')
            axs[6].plot(df_dt['us-tr'], 'g-')
            axs[6].plot(df['us-tr-fit'], 'r-')
        else:
            fig, axs = plt.subplots(5)
            fig.suptitle(tstring)
            axs[0].plot(df['force'])
            axs[0].set(ylabel='force')
            axs[1].plot(df['us-csa'])
            axs[1].set(ylabel='us-csa')
            axs[1].plot(df_dt['us-csa'], 'g-')
            axs[1].plot(df['us-csa-fit'], 'r-')
            axs[2].plot(df['us-t'])
            axs[2].set(ylabel='us-t')
            axs[2].plot(df_dt['us-t'], 'g-')
            axs[2].plot(df['us-t-fit'], 'r-')
            axs[3].plot(df['us-t-dt'])
            axs[3].set(ylabel='us-t-dt')
            axs[4].plot(df['us-tr'])
            axs[4].set(ylabel='us-tr')
            axs[4].plot(df_dt['us-tr'], 'g-')
            axs[4].plot(df['us-tr-fit'], 'r-')

        plt.show()


    def build_data_series(self, data, col=0):
        """
        """
        length = data.data_from_offset.shape[0]
        freq_pd_str = self.as_pd_freq(data.freq)
        index = pd.date_range('2017-01-27', periods=length, freq=freq_pd_str)
        return pd.Series(data.data_from_offset[:, col], index)

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


    def offset_from_peak(self, peak_ind, freq, prepeak=3):
        offset = int(peak_ind - prepeak*freq)
        #print(offset)
        return offset


    def _compute_abs_force(self):
        force_data_comps = self.data_force.data
        force_abs = np.zeros((force_data_comps.shape[0], 1))
        for i in range(force_abs.shape[0]):
            x_i = force_data_comps[i, 0]
            y_i = force_data_comps[i, 1]
            z_i = force_data_comps[i, 2]

            force_abs[i] = math.sqrt(math.pow(x_i, 2)+math.pow(y_i, 2)+math.pow(z_i, 2))

        return force_abs













