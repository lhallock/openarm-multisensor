#!/usr/bin/env python3
"""Generate all plots for publication.

Example:
    Once filepaths are set appropriately, run this function via

        $ python gen_pub_figs.py
"""
import os

import pandas as pd

from multisensorimport.dataobj import trialdata as td
from multisensorimport.viz import plot_utils, stats_utils

# directory containing all data (script path + relative string)
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/sandbox/data/FINAL/'

# directory names for all subject (trial_*.p) data files
SUBJ_DIRS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# whether to print titles on time series plots (True for publication)
NO_TITLES = False


def main():
    """Generate all plots reported in TNSRE 2021 publication."""

    # generate correlation plots
    print('Plotting correlation by subject...')
    df_all_corrs = stats_utils.gen_corr_df(DATA_DIR, SUBJ_DIRS, 'trial_1b.p')
    plot_utils.gen_subj_corr_plot(df_all_corrs)
    print('done.\n')

    print('Plotting correlation by trajectory type...')
    plot_utils.gen_trajtype_corr_plot(df_all_corrs)
    print('done.\n')

    # generate trajectory tracking error plots
    print('Plotting trajectory tracking error by subject...')
    df_all_errors = stats_utils.gen_err_df(DATA_DIR, SUBJ_DIRS, 'trial_2b.p',
                                           'trial_3b.p')
    plot_utils.gen_subj_err_plot(df_all_errors)
    print('done.\n')

    print('Plotting trajectory tracking error by trajectory type...')
    plot_utils.gen_trajtype_err_plot(df_all_errors)
    print('done.\n')

    # generate survey plots
    print('Plotting user preferences (evaluating controllers separately)...')
    df_us = pd.read_csv(DATA_DIR + 'survey_us.csv')
    df_emg = pd.read_csv(DATA_DIR + 'survey_emg.csv')
    plot_utils.gen_survey_box_plot(df_us, df_emg)
    print('done.\n')

    print('Plotting user preferences (evaluating controllers together)...')
    df_comp = pd.read_csv(DATA_DIR + 'survey_comp.csv')
    plot_utils.gen_survey_comp_box_plot(df_comp)
    print('done.\n')

    print('Plotting time series data for all subjects...')
    for d in SUBJ_DIRS:
        readpath_corr = DATA_DIR + d + '/trial_1b.p'
        readpath_us = DATA_DIR + d + '/trial_2b.p'
        readpath_emg = DATA_DIR + d + '/trial_3b.p'

        data_corr = td.TrialData.from_pickle(readpath_corr, d)
        data_us = td.TrialData.from_pickle(readpath_us, d)
        data_emg = td.TrialData.from_pickle(readpath_emg, d)

        plot_utils.gen_time_plot(data_corr, no_titles=NO_TITLES)
        plot_utils.gen_tracking_time_plot(data_us,
                                          data_emg,
                                          no_titles=NO_TITLES)
    print('done.\n')


if __name__ == "__main__":
    main()
