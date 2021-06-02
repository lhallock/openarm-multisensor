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

SUBJ_DIRS = ['2A/', '3B/', '4A/', '5B/', '6A/', '7B/', '8B/', '9B/', '10A/',
             '11A/']

# paths to trial files
READ_PATH_TRIAL = []
READ_PATH_TRIAL0 = DATA_DIR + 'trial_0.p'
READ_PATH_TRIAL1 = DATA_DIR + 'trial_1b.p'
READ_PATH_TRIAL2 = DATA_DIR + 'trial_2b.p'
READ_PATH_TRIAL3 = DATA_DIR + 'trial_3b.p'

# sensor order in import files
US_IND = 0
EMG_IND = 1
FORCE_IND = 2

no_titles = False

def main():
    """Execute all time series data analysis for TNSRE 2021 publication."""
#    readpath = READ_PATH_TRIAL1

    df_us = pd.read_csv(DATA_DIR + 'survey_us.csv')
    df_emg = pd.read_csv(DATA_DIR + 'survey_emg.csv')

    plot_utils.gen_survey_box_plot(df_us, df_emg)

    df_comp = pd.read_csv(DATA_DIR + 'survey_comp.csv')
    plot_utils.gen_survey_comp_box_plot(df_comp)

    raise ValueError('break')

    for d in SUBJ_DIRS:
        readpath = DATA_DIR + d + 'trial_1b.p'

        data = td.TrialData.from_pickle(readpath, d)
        print(data.subj)
        print(data.trial_no)
        print(data.mins)
        print(data.maxs)
        print(data.traj_start)
        print(data.df)

#    data.df.plot()
#    plt.show()

        print(data.df.corr())

        plot_utils.gen_time_plot(data)

#    trialdict = np.load(readpath, allow_pickle=True)
#    print(trialdict.keys())

#    print(trialdict['Traj-Changes'])
#    print(trialdict['Processed-Maxs'])
#    print(trialdict['Processed-Mins'])
    #for key in trialdict.keys():
    #    print(str(key) + ' ' + str(len(trialdict[key])))
    #    print(trialdict[key])


    # df = pd.DataFrame(trialdict)
    # print(df)

#    starttime = trialdict['Traj-Changes'][1]

    # ultrasound, emg, force
    sns.set()

    num_subplots = 3

    fig, axs = plt.subplots(num_subplots)

#    axs[0].plot(trialdict['Raw'][US_IND], 'k')
    axs[0].plot(trialdict['Processed'][US_IND][starttime:], 'b')
#    axs[1].plot(trialdict['Raw'][EMG_IND], 'k')
    axs[1].plot(trialdict['Processed'][EMG_IND][starttime:], 'b')
#    axs[2].plot(trialdict['Raw'][FORCE_IND], 'k')
    axs[2].plot(trialdict['Processed'][FORCE_IND][starttime:], 'b')

    axs[2].set_xlabel('time (s)')
#    axs[5].xaxis.set_label_coords(1.0, -0.15)

    if not no_titles:
#        fig.suptitle(READ_PATH_TRIAL0)
        axs[0].set(ylabel='US')
        axs[1].set(ylabel='EMG')
        axs[2].set(ylabel='f')

#    axs[0].xaxis.set_visible(False)

    plt.show()



if __name__ == "__main__":
    main()