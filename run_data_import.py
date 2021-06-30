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
    err_df_list = []
    for d in SUBJ_DIRS:
        readpath_corr = DATA_DIR + d + '/trial_1b.p'
        readpath_us = DATA_DIR + d + '/trial_2b.p'
        readpath_emg = DATA_DIR + d + '/trial_3b.p'

        data_corr = td.TrialData.from_pickle(readpath_corr, d)
        data_us = td.TrialData.from_pickle(readpath_us, d)
        data_emg = td.TrialData.from_pickle(readpath_emg, d)

        errors_us = pd.Series(data_us.get_tracking_errors('us'))
        errors_emg = pd.Series(data_emg.get_tracking_errors('emg'))

        df_errors = pd.DataFrame({'deformation': errors_us, 'activation': errors_emg})
        df_errors_melt = pd.melt(df_errors.reset_index(), id_vars='index',
                                 value_vars=['deformation', 'activation'])

        df_errors_melt['subj'] = d
        err_df_list.append(df_errors_melt)

        plot_utils.gen_time_plot(data_corr, no_titles=True)

        raise ValueError('break')

    df_all_errors = pd.concat(err_df_list)
    print(df_all_errors)


    err_ind_dict = {'ALL': 4, 'sustained': 0, 'ramp': 1, 'step': 2, 'sine': 3}
    df_all_errors['index_err'] = df_all_errors['index'].map(err_ind_dict)
    df_all_errors['subj'] = df_all_errors['subj'].apply(pd.to_numeric)

    print(df_all_errors)
#    df_error_melt = df_all_errors.melt(id_vars=['subj'], value_vars=['us', 'emg'])
#    ax = sns.barplot(x='index', y='value', hue='variable', data=df_all_errors)
#    plt.show()

    plot_utils.gen_trajtype_err_plot(df_all_errors)

#    df_err_agg = df_all_errors.loc[df_all_errors['index'] == 'ALL']
#    ax = sns.barplot(x='subj', y='value', hue='variable', data=df_err_agg)
#    plt.show()

    plot_utils.gen_subj_err_plot(df_all_errors)
#    raise ValueError('break')


    # CORRELATION PLOTS

    df_all_corrs = stats_utils.gen_corr_df(DATA_DIR, SUBJ_DIRS, 'trial_1b.p')
    print(df_all_corrs)
    print(df_all_corrs.dtypes)

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
