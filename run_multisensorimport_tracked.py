#!/usr/bin/env python3
"""Example import of time series muscle data.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_multisensorimport.py
"""
import os

from multisensorimport.dataobj import data_utils
from multisensorimport.dataobj import trialdata as td
from multisensorimport.viz import plot_utils, print_utils, stats_utils

# tracking frame to generate
TRACKER = 'SBLK-T'

# directory containing all data (script path + relative string)
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/sandbox/data/FINAL/'

# paths to MATLAB structured data (force, sEMG, etc.)
READ_PATH_MAT_SUB1 = DATA_DIR + 'sub1/seg_data.mat'
READ_PATH_MAT_SUB2 = DATA_DIR + 'sub2/seg_data.mat'
READ_PATH_MAT_SUB3 = DATA_DIR + 'sub3/seg_data.mat'
READ_PATH_MAT_SUB4 = DATA_DIR + 'sub4/seg_data.mat'
READ_PATH_MAT_SUB5 = DATA_DIR + 'sub5/seg_data.mat'

# paths to ground truth ultrasound data frames
READ_PATH_US_SUB1_WP1 = DATA_DIR + 'sub1/wp1t5'
READ_PATH_US_SUB1_WP2 = DATA_DIR + 'sub1/wp2t6'
READ_PATH_US_SUB1_WP5 = DATA_DIR + 'sub1/wp5t11'
READ_PATH_US_SUB1_WP8 = DATA_DIR + 'sub1/wp8t15'
READ_PATH_US_SUB1_WP10 = DATA_DIR + 'sub1/wp10t25'
READ_PATH_US_SUB2_WP5 = DATA_DIR + 'sub2/wp5t28'
READ_PATH_US_SUB3_WP5 = DATA_DIR + 'sub3/wp5t33'
READ_PATH_US_SUB4_WP5 = DATA_DIR + 'sub4/wp5t34'
READ_PATH_US_SUB5_WP5 = DATA_DIR + 'sub5/wp5t37'

# paths to tracked ultrsound data frames
READ_PATH_US_SUB1_WP1_TRACKED = DATA_DIR + 'sub1/wp1t5/' + TRACKER
READ_PATH_US_SUB1_WP2_TRACKED = DATA_DIR + 'sub1/wp2t6/' + TRACKER
READ_PATH_US_SUB1_WP5_TRACKED = DATA_DIR + 'sub1/wp5t11/' + TRACKER
READ_PATH_US_SUB1_WP8_TRACKED = DATA_DIR + 'sub1/wp8t15/' + TRACKER
READ_PATH_US_SUB1_WP10_TRACKED = DATA_DIR + 'sub1/wp10t25/' + TRACKER
READ_PATH_US_SUB2_WP5_TRACKED = DATA_DIR + 'sub2/wp5t28/' + TRACKER
READ_PATH_US_SUB3_WP5_TRACKED = DATA_DIR + 'sub3/wp5t33/' + TRACKER
READ_PATH_US_SUB4_WP5_TRACKED = DATA_DIR + 'sub4/wp5t34/' + TRACKER
READ_PATH_US_SUB5_WP5_TRACKED = DATA_DIR + 'sub5/wp5t37/' + TRACKER

# desired out paths for correlation dataframes
ANG_CORR_OUT_PATH = DATA_DIR + 'ang_corr_tracked.csv'
SUBJ_CORR_OUT_PATH = DATA_DIR + 'subj_corr_tracked.csv'

# show polynomial fit debugging plots
DEBUG = False

# eliminate certain plot titles/labels for publication printing
PRINT_PUB_PLOTS = False


def main():
    """Execute all time series data analysis for BioRob 2020 publication."""
    # import all ground truth time series data, detrend via polynomial fit
    # NOTE: AMG data not currently analyzed, peaks inaccurate
    print('Aggregating and fitting ground truth time series data (Sub1, 25deg)...')
    data_sub1_wp1 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP1,
        'sub1',
        0,
        emg_peak=5500,
        amg_peak=13290,
        force_peak=3721,
        us_peak=51)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub1, 44deg)...')
    data_sub1_wp2 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP2,
        'sub1',
        1,
        emg_peak=5500,
        amg_peak=13290,
        force_peak=5800,
        us_peak=46)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub1, 69deg)...')
    data_sub1_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP5,
        'sub1',
        4,
        emg_peak=5800,
        amg_peak=13290,
        force_peak=4476,
        us_peak=50)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub1, 82deg)...')
    data_sub1_wp8 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP8,
        'sub1',
        7,
        emg_peak=5700,
        amg_peak=13290,
        force_peak=2469,
        us_peak=49)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub1, 97deg)...')
    data_sub1_wp10 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP10,
        'sub1',
        9,
        emg_peak=6000,
        amg_peak=13290,
        force_peak=4222,
        us_peak=48)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub2, 69deg)...')
    data_sub2_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB2,
        READ_PATH_US_SUB2_WP5,
        'sub2',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=4033,
        us_peak=53,
        force_only=True)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub3, 69deg)...')
    data_sub3_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB3,
        READ_PATH_US_SUB3_WP5,
        'sub3',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=7113,
        us_peak=63,
        force_only=True)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub4, 69deg)...')
    data_sub4_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB4,
        READ_PATH_US_SUB4_WP5,
        'sub4',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=10810,
        us_peak=72,
        force_only=True)
    print('Done.')

    print('\nAggregating and fitting ground truth time series data (Sub5, 69deg)...')
    data_sub5_wp5 = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB5,
        READ_PATH_US_SUB5_WP5,
        'sub5',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=0,
        us_peak=36,
        force_only=True)
    print('Done.')

    print_utils.print_div()

    # import all tracked time series data, detrend via polynomial fit
    # TODO: remove polyfit
    print('Aggregating and fitting tracked time series data (Sub1, 25deg)...')
    data_sub1_wp1_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP1_TRACKED,
        'sub1',
        0,
        emg_peak=5500,
        amg_peak=13290,
        force_peak=3721,
        us_peak=51,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub1, 44deg)...')
    data_sub1_wp2_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP2_TRACKED,
        'sub1',
        1,
        emg_peak=5500,
        amg_peak=13290,
        force_peak=5800,
        us_peak=46,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub1, 69deg)...')
    data_sub1_wp5_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP5_TRACKED,
        'sub1',
        4,
        emg_peak=5800,
        amg_peak=13290,
        force_peak=4476,
        us_peak=50,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub1, 82deg)...')
    data_sub1_wp8_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP8_TRACKED,
        'sub1',
        7,
        emg_peak=5700,
        amg_peak=13290,
        force_peak=2469,
        us_peak=49,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub1, 97deg)...')
    data_sub1_wp10_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB1,
        READ_PATH_US_SUB1_WP10_TRACKED,
        'sub1',
        9,
        emg_peak=6000,
        amg_peak=13290,
        force_peak=4222,
        us_peak=48,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub2, 69deg)...')
    data_sub2_wp5_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB2,
        READ_PATH_US_SUB2_WP5_TRACKED,
        'sub2',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=4033,
        us_peak=53,
        force_only=True,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub3, 69deg)...')
    data_sub3_wp5_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB3,
        READ_PATH_US_SUB3_WP5_TRACKED,
        'sub3',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=7113,
        us_peak=63,
        force_only=True,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub4, 69deg)...')
    data_sub4_wp5_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB4,
        READ_PATH_US_SUB4_WP5_TRACKED,
        'sub4',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=10810,
        us_peak=72,
        force_only=True,
        data_type='tracking')
    print('Done.')

    print('\nAggregating and fitting tracked time series data (Sub5, 69deg)...')
    data_sub5_wp5_tracked = td.TrialData.from_preprocessed_mat_file(
        READ_PATH_MAT_SUB5,
        READ_PATH_US_SUB5_WP5_TRACKED,
        'sub5',
        0,
        emg_peak=None,
        amg_peak=None,
        force_peak=0,
        us_peak=36,
        force_only=True,
        data_type='tracking')
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
        print('Done.')

        print_utils.print_div()

    # generate final formatted plots
    print('\nDisplaying final plots...')
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp1, data_sub1_wp1_tracked, PRINT_PUB_PLOTS)
    #plot_utils.gen_time_plot_w_tracking(data_sub1_wp2, data_sub1_wp2_tracked, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp5, data_sub1_wp5_tracked, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp8, data_sub1_wp8_tracked, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub1_wp10, data_sub1_wp10_tracked, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub2_wp5, data_sub2_wp5_tracked, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub3_wp5, data_sub3_wp5_tracked, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub4_wp5, data_sub4_wp5_tracked, PRINT_PUB_PLOTS)
    plot_utils.gen_time_plot_w_tracking(data_sub5_wp5, data_sub5_wp5_tracked, PRINT_PUB_PLOTS)
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
