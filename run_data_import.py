#!/usr/bin/env python3
"""Example import of time series muscle data.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_multisensorimport_w_tracking.py
"""
import os

import pickle5 as pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from multisensorimport.dataobj import data_utils
from multisensorimport.dataobj import trialdata as td
from multisensorimport.viz import plot_utils, print_utils, stats_utils


# directory containing all data (script path + relative string)
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/sandbox/data/FINAL/'

SUBJ_DIRS = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
             '10']

no_titles = False

def main():
    """Execute all time series data analysis for TNSRE 2021 publication."""
#    readpath = READ_PATH_TRIAL1

    for d in SUBJ_DIRS:
        readpath = DATA_DIR + d + '/trial_3b.p'

        data = td.TrialData.from_pickle(readpath, d)
        plot_utils.gen_time_plot(data)


    raise ValueError('break')
    # CORRELATION PLOTS

    df_all_corrs = stats_utils.gen_corr_df(DATA_DIR, SUBJ_DIRS, 'trial_1b.p')
#    print(df_all_corrs)

    plot_utils.gen_trajtype_corr_plot(df_all_corrs)
    plot_utils.gen_subj_corr_plot(df_all_corrs)



    # SURVEY PLOTS

    df_us = pd.read_csv(DATA_DIR + 'survey_us.csv')
    df_emg = pd.read_csv(DATA_DIR + 'survey_emg.csv')

    plot_utils.gen_survey_box_plot(df_us, df_emg)

    df_comp = pd.read_csv(DATA_DIR + 'survey_comp.csv')
    plot_utils.gen_survey_comp_box_plot(df_comp)



if __name__ == "__main__":
    main()
