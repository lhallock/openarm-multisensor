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

# paths to trial files
READ_PATH_TRIAL = []
READ_PATH_TRIAL0 = DATA_DIR + '/trial_0.p'
READ_PATH_TRIAL1 = DATA_DIR + '/trial_1b.p'
READ_PATH_TRIAL2 = DATA_DIR + '/trial_2b.p'
READ_PATH_TRIAL3 = DATA_DIR + '/trial_3b.p'

# sensor order in import files
US_IND = 0
EMG_IND = 1
FORCE_IND = 2

no_titles = False

def main():
    """Execute all time series data analysis for TNSRE 2021 publication."""
#    readpath = READ_PATH_TRIAL1

    corr_df_list = []
    for d in SUBJ_DIRS:
        readpath = DATA_DIR + d + '/trial_1b.p'

        data = td.TrialData.from_pickle(readpath, d)
        print(data.subj)
        print(data.trial_no)
        print(data.mins)
        print(data.maxs)
        print(data.traj_start)
        print(data.df)

        print(data.df.corr())
        print(data.df.corr()['force']['emg'])
        corrs_us = pd.Series(data.get_corrs('us'))
        corrs_emg = pd.Series(data.get_corrs('emg'))

        df_corrs = pd.DataFrame({'us': corrs_us, 'emg': corrs_emg})
        df_corrs_melt = pd.melt(df_corrs.reset_index(),
                      id_vars='index',value_vars=['us','emg'])
        print(df_corrs)
        df_corrs_melt['subj'] = data.subj
        print(df_corrs_melt)

        corr_df_list.append(df_corrs_melt)

#        raise ValueError('break')

        plot_utils.gen_time_plot(data)

    df_all_corrs = pd.concat(corr_df_list)
    print(df_all_corrs)

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
