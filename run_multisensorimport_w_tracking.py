#!/usr/bin/env python3
"""Example import of time series muscle data.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_multisensorimport_w_tracking.py
"""
import os

import pandas as pd

from multisensorimport.dataobj import data_utils
from multisensorimport.dataobj import trialdata as td
from multisensorimport.viz import plot_utils, print_utils, stats_utils

# tracking frame to generate
TRACKER = 'SBLK-T'

# times for tracking evaluation (box plots)
TIME_IN = '00:00:00'
TIME_OUT = '00:02:00'

# directory containing all data (script path + relative string)
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/sandbox/data/FINAL/'

# paths to MATLAB structured data (force, sEMG, etc.)
READ_PATH_MAT_SUB1 = DATA_DIR + 'sub1/seg_data.mat'
READ_PATH_MAT_SUB2 = DATA_DIR + 'sub2/seg_data.mat'
READ_PATH_MAT_SUB3 = DATA_DIR + 'sub3/seg_data.mat'
READ_PATH_MAT_SUB4 = DATA_DIR + 'sub4/seg_data.mat'
READ_PATH_MAT_SUB5 = DATA_DIR + 'sub5/seg_data.mat'

# paths to ultrasound data frames
READ_PATH_US_SUB1_WP1 = DATA_DIR + 'sub1/wp1t5'
READ_PATH_US_SUB1_WP2 = DATA_DIR + 'sub1/wp2t6'
READ_PATH_US_SUB1_WP5 = DATA_DIR + 'sub1/wp5t11'
READ_PATH_US_SUB1_WP8 = DATA_DIR + 'sub1/wp8t15'
READ_PATH_US_SUB1_WP10 = DATA_DIR + 'sub1/wp10t25'
READ_PATH_US_SUB2_WP5 = DATA_DIR + 'sub2/wp5t28'
READ_PATH_US_SUB3_WP5 = DATA_DIR + 'sub3/wp5t33'
READ_PATH_US_SUB4_WP5 = DATA_DIR + 'sub4/wp5t34'
READ_PATH_US_SUB5_WP5 = DATA_DIR + 'sub5/wp5t37'

# desired out paths for correlation dataframes
ANG_CORR_OUT_PATH = DATA_DIR + 'ang_corr.csv'
SUBJ_CORR_OUT_PATH = DATA_DIR + 'subj_corr.csv'

# show polynomial fit debugging plots
DEBUG = False

# eliminate certain plot titles/labels for publication printing
PRINT_PUB_PLOTS = False


def main():
    """Execute all time series data analysis for TNSRE 2021 publication."""
    # import all time series data, detrend via polynomial fit
    # NOTE: AMG data not currently analyzed, peaks inaccurate
    print('Aggregating and fitting time series data (Sub1, 25deg)...')
    data_sub1_wp1 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP1,
        'sub1',
        0,
        emg_peak=5500,
        amg_peak=13290,
        force_peak=3721,
        us_peak=51,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 44deg)...')
    data_sub1_wp2 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP2,
        'sub1',
        1,
        emg_peak=5500,
        amg_peak=13290,
        force_peak=5800,
        us_peak=46,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 69deg)...')
    data_sub1_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP5,
        'sub1',
        4,
        emg_peak=5800,
        amg_peak=13290,
        force_peak=4476,
        us_peak=50,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 82deg)...')
    data_sub1_wp8 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP8,
        'sub1',
        7,
        emg_peak=5700,
        amg_peak=13290,
        force_peak=2469,
        us_peak=49,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub1, 97deg)...')
    data_sub1_wp10 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP10,
        'sub1',
        9,
        emg_peak=6000,
        amg_peak=13290,
        force_peak=4222,
        us_peak=48,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub2, 69deg)...')
    data_sub2_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB2,
        READ_PATH_US_SUB2_WP5,
        'sub2',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=4033,
        us_peak=53,
        force_only=True,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub3, 69deg)...')
    data_sub3_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB3,
        READ_PATH_US_SUB3_WP5,
        'sub3',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=7113,
        us_peak=63,
        force_only=True,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub4, 69deg)...')
    data_sub4_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB4,
        READ_PATH_US_SUB4_WP5,
        'sub4',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=10810,
        us_peak=72,
        force_only=True,
        tracking_data_type=TRACKER)
    print('Done.')

    print('\nAggregating and fitting time series data (Sub5, 69deg)...')
    data_sub5_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB5,
        READ_PATH_US_SUB5_WP5,
        'sub5',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=0,
        us_peak=36,
        force_only=True,
        tracking_data_type=TRACKER)
    print('Done.')

    print_utils.print_div()

    # construct time-sliced frames
    print('\nConstructing time-sliced frames from ' + TIME_IN + ' to ' + TIME_OUT + '...')
    df_sub1_wp1_crop = data_sub1_wp1.df.between_time(TIME_IN, TIME_OUT)
    df_sub1_wp2_crop = data_sub1_wp2.df.between_time(TIME_IN, TIME_OUT)
    df_sub1_wp5_crop = data_sub1_wp5.df.between_time(TIME_IN, TIME_OUT)
    df_sub1_wp8_crop = data_sub1_wp8.df.between_time(TIME_IN, TIME_OUT)
    df_sub1_wp10_crop = data_sub1_wp10.df.between_time(TIME_IN, TIME_OUT)
    df_sub2_wp5_crop = data_sub2_wp5.df.between_time(TIME_IN, TIME_OUT)
    df_sub3_wp5_crop = data_sub3_wp5.df.between_time(TIME_IN, TIME_OUT)
    df_sub4_wp5_crop = data_sub4_wp5.df.between_time(TIME_IN, TIME_OUT)
    df_sub5_wp5_crop = data_sub5_wp5.df.between_time(TIME_IN, TIME_OUT)
    print('Done.')

    # aggregate time-sliced frames
    print('\nAggregating time-sliced frames...')
    df_box_agg = pd.concat([df_sub1_wp1_crop, df_sub1_wp2_crop,
                            df_sub1_wp5_crop, df_sub1_wp8_crop,
                            df_sub1_wp10_crop, df_sub2_wp5_crop,
                            df_sub3_wp5_crop, df_sub4_wp5_crop, df_sub5_wp5_crop], ignore_index=True, sort=False)
    print('Done.')

    print_utils.print_div()

    # show debugging plots for alignment and fit quality evaluation
    if DEBUG:
        print('\nDisplaying debug plots...')
        plot_utils.gen_debug_time_plot(data_sub1_wp1)
        plot_utils.gen_debug_time_plot(data_sub1_wp2)
        plot_utils.gen_debug_time_plot(data_sub1_wp5)
        plot_utils.gen_debug_time_plot(data_sub1_wp8)
        plot_utils.gen_debug_time_plot(data_sub1_wp10)
        plot_utils.gen_debug_time_plot(data_sub2_wp5)
        plot_utils.gen_debug_time_plot(data_sub3_wp5)
        plot_utils.gen_debug_time_plot(data_sub4_wp5)
        plot_utils.gen_debug_time_plot(data_sub5_wp5)
        plot_utils.gen_error_box_plot(data_sub1_wp1.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub1_wp2.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub1_wp5.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub1_wp8.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub1_wp10.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub2_wp5.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub3_wp5.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub4_wp5.df.between_time(TIME_IN, TIME_OUT))
        plot_utils.gen_error_box_plot(data_sub5_wp5.df.between_time(TIME_IN, TIME_OUT))
        print('Done.')

        print_utils.print_div()

    # generate final formatted plots
    print('\nDisplaying final time series plots...')
    plot_utils.gen_time_plot(data_sub1_wp1, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub1_wp2, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub1_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub1_wp8, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub1_wp10, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub2_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub3_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub4_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot(data_sub5_wp5, PRINT_PUB_PLOTS)
    print('Done.')

    print_utils.print_div()

    # generate final formatted plots
    print('\nDisplaying final time series plots w/ tracking...')
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp1, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp2, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp8, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp10, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub2_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub3_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub4_wp5, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub5_wp5, PRINT_PUB_PLOTS)
    print('Done.')

    print_utils.print_div()


    # generate final formatted violin plot
    print('\nDisplaying final violin plot (may take a few moments)...')
    plot_utils.gen_error_box_plot(df_box_agg)
    print('Done.')

    print_utils.print_div()


    # generate correlation data and print to console and CSV
    print('\nComputing correlations...')
    df_corr = data_utils.build_corr_table([
        data_sub1_wp1, data_sub1_wp2, data_sub1_wp5, data_sub1_wp8,
        data_sub1_wp10, data_sub2_wp5, data_sub3_wp5, data_sub4_wp5,
        data_sub5_wp5
    ])

    df_ang, df_subj = stats_utils.gen_refined_corr_dfs(df_corr,
                                                       ANG_CORR_OUT_PATH,
                                                       SUBJ_CORR_OUT_PATH)
    print('Done.')

    print_utils.print_header('[SIGNAL]-FORCE CORRELATION ACROSS ANGLES (SUB1)')
    print(df_ang.T)
    print_utils.print_header(
        '[SIGNAL]-FORCE CORRELATION ACROSS SUBJECTS (69deg)')
    print(df_subj.T)


if __name__ == "__main__":
    main()
