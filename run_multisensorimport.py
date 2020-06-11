#!/usr/bin/env python3
"""Example import of time series muscle data.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_multisensorimport.py

Todo:
    implement all the things!
"""
from multisensorimport.dataobj import trialdata as td
from multisensorimport.dataobj import data_utils
from multisensorimport.viz import plot_utils, stats_utils

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/'

READ_PATH_MAT = DATA_DIR + 'sub1/seg_data.mat'
READ_PATH_MAT_28 = DATA_DIR + 'sub2/seg_data.mat'
READ_PATH_MAT_33 = DATA_DIR + 'sub3/seg_data.mat'
READ_PATH_MAT_34 = DATA_DIR + 'sub4/seg_data.mat'
READ_PATH_MAT_37 = DATA_DIR + 'sub5/seg_data.mat'

READ_PATH_US_1 = DATA_DIR + 'sub1/wp1t5'
READ_PATH_US_2 = DATA_DIR + 'sub1/wp2t6'
READ_PATH_US_5 = DATA_DIR + 'sub1/wp5t11'
READ_PATH_US_8 = DATA_DIR + 'sub1/wp8t15'
READ_PATH_US_10 = DATA_DIR + 'sub1/wp10t25'
READ_PATH_US_28 = DATA_DIR + 'sub2/wp5t28'
READ_PATH_US_33 = DATA_DIR + 'sub3/wp5t33'
READ_PATH_US_34 = DATA_DIR + 'sub4/wp5t34'
READ_PATH_US_37 = DATA_DIR + 'sub5/wp5t37'

ANG_CORR_OUT_PATH = DATA_DIR + 'ang_corr.csv'
SUBJ_CORR_OUT_PATH = DATA_DIR + 'subj_corr.csv'

DEBUG = True

def main():
    """Execute all time series data analysis for [PUBLICATION FORTHCOMING]."""

    # import all time series data, detrend via polynomial fit
    # NOTE: AMG data not currently analyzed, peaks inaccurate
    print('Aggregating and fitting time series data (Sub1, 25deg)...')
    data1 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_1, 'sub1', 0,
                                                    emg_peak=5500,
                                                    amg_peak=13290,
                                                    force_peak=3721, us_peak=51)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 44deg)...')
    data2 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_2, 'sub1', 1,
                                                    emg_peak=5500,
                                                    amg_peak=13290,
                                                    force_peak=5800, us_peak=46)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 69deg)...')
    data5 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_5, 'sub1', 4,
                                                    emg_peak=5800,
                                                    amg_peak=13290,
                                                    force_peak=4476, us_peak=50)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 82deg)...')
    data8 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                    READ_PATH_US_8, 'sub1', 7,
                                                    emg_peak=5700,
                                                    amg_peak=13290,
                                                    force_peak=2469, us_peak=49)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 97deg)...')
    data10 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT,
                                                     READ_PATH_US_10, 'sub1', 9,
                                                     emg_peak=6000,
                                                     amg_peak=13290,
                                                     force_peak=4222, us_peak=48)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub2, 69deg)...')
    data28 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_28,
                                                     READ_PATH_US_28, 'sub2', 0,
                                                     emg_peak=None,
                                                     amg_peak=None,
                                                     force_peak=4033,
                                                     us_peak=53,
                                                     force_only=True)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub3, 69deg)...')
    data33 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_33,
                                                     READ_PATH_US_33, 'sub3', 0,
                                                     emg_peak=None,
                                                     amg_peak=None,
                                                     force_peak=7113,
                                                     us_peak=63,
                                                     force_only=True)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub4, 69deg)...')
    data34 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_34,
                                                     READ_PATH_US_34, 'sub4', 0,
                                                     emg_peak=None,
                                                     amg_peak=None,
                                                     force_peak=10810,
                                                     us_peak=72, force_only=True)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub5, 69deg)...')
    data37 = td.TrialData.from_preprocessed_mat_file(READ_PATH_MAT_37,
                                                     READ_PATH_US_37, 'sub5', 0,
                                                     emg_peak=None,
                                                     amg_peak=None,
                                                     force_peak=0, us_peak=36,
                                                     force_only=True)
    print('Done.')

    # show debugging plots for alignment and fit quality evaluation
    if DEBUG:
        plot_utils.gen_debug_time_plot(data1)
        plot_utils.gen_debug_time_plot(data2)
        plot_utils.gen_debug_time_plot(data5)
        plot_utils.gen_debug_time_plot(data8)
        plot_utils.gen_debug_time_plot(data10)
        plot_utils.gen_debug_time_plot(data28)
        plot_utils.gen_debug_time_plot(data33)
        plot_utils.gen_debug_time_plot(data34)
        plot_utils.gen_debug_time_plot(data37)

    # generate final formatted plots
        plot_utils.gen_time_plot(data1)
        plot_utils.gen_time_plot(data2)
        plot_utils.gen_time_plot(data5)
        plot_utils.gen_time_plot(data8)
        plot_utils.gen_time_plot(data10)
        plot_utils.gen_time_plot(data28)
        plot_utils.gen_time_plot(data33)
        plot_utils.gen_time_plot(data34)
        plot_utils.gen_time_plot(data37)

    # generate correlation data and print to console and CSV
    df_corr = data_utils.build_corr_table([data1, data2, data5, data8, data10, data28, data33,
                           data34, data37])

    df_ang, df_subj = stats_utils.gen_refined_corr_dfs(df_corr, ANG_CORR_OUT_PATH,
                                                       SUBJ_CORR_OUT_PATH)
    print(df_ang)
    print(df_subj)

if __name__ == "__main__":
    main()
