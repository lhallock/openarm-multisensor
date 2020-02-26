#!/usr/bin/env python3
"""Generate all tracker plots.

Example:
    Once filepaths are set appropriately, run this function via

        $ python run_trackerplots.py

Todo:
    implement all the things!
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from multisensorimport.dataobj import trialdata as td
from multisensorimport.dataobj import data_utils as utils

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/'

DATA_DIR_SUB1 = DATA_DIR + 'sub1/wp5t11/'
DATA_DIR_SUB2 = DATA_DIR + 'sub2/wp5t28/'
DATA_DIR_SUB3 = DATA_DIR + 'sub3/wp5t33/'
DATA_DIR_SUB4 = DATA_DIR + 'sub4/wp5t34/'
DATA_DIR_SUB5 = DATA_DIR + 'sub5/wp5t37/'

TRACKER_STRINGS = ['LK','FRLK','BFLK-G','BFLK-T','SBLK-G','SBLK-T']

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

CORR_OUT_PATH = DATA_DIR + 'correlations.csv'

PLOT = True

PLOT_FONT = 'Open Sans'

def main():

#    mpl.rc('font',family='Times New Roman')

    df_ang = pd.read_csv(DATA_DIR + 'ang_corr.csv', header=0,
                         index_col='measure')

    df_ang = df_ang.T

    print(df_ang)

    ang_colors = ['#fdae6b','#f16d13','#41b6c4','#41b6c4','#225ea8','#225ea8','#081d58','#081d58']
    styles = ['-','-','-','--','-','--','-','--']
    sns.set()
    ax = df_ang.plot(kind='line', style=styles, color=ang_colors, rot=0)
#    bars = ax.patches
#    patterns =('-', '+', 'x','/','//','O','o','\\','\\\\')
#    patterns = ('','////','','////','','////')
#    hatches = [p for p in patterns for i in range(len(df_subj_def))]
#    for bar, hatch in zip(bars, hatches):
#            bar.set_hatch(hatch)

    L = ax.legend(loc='lower left', ncol=4)
    plt.setp(L.texts, family=PLOT_FONT)
    ax.set_xlabel('Flexion Angle ($\degree$ from full extension)', fontname=PLOT_FONT)
    ax.set_ylabel('CC(·,f)', fontname=PLOT_FONT)
    for tick in ax.get_xticklabels():
        tick.set_fontname(PLOT_FONT)
    for tick in ax.get_yticklabels():
        tick.set_fontname(PLOT_FONT)
    plt.show()


#    raise ValueError('breakpoint')

    df_subj_def = pd.read_csv(DATA_DIR + 'subj_corr.csv', header=0,
                               index_col='measure')
    df_subj_def = df_subj_def.T

    print(df_subj_def)

    mod_subj_colors = ['#41b6c4','#41b6c4','#225ea8','#225ea8','#081d58','#081d58']
    sns.set()
    ax = df_subj_def.plot(kind='bar', color=mod_subj_colors, rot=0)
    bars = ax.patches
#    patterns =('-', '+', 'x','/','//','O','o','\\','\\\\')
    patterns = ('','////','','////','','////')
    hatches = [p for p in patterns for i in range(len(df_subj_def))]
    for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    L = ax.legend(loc='lower left', ncol=3)
    plt.setp(L.texts, family=PLOT_FONT)
    ax.set_xlabel('Subject', fontname=PLOT_FONT)
    ax.set_ylabel('CC(·,f)', fontname=PLOT_FONT)
    for tick in ax.get_xticklabels():
        tick.set_fontname(PLOT_FONT)
    for tick in ax.get_yticklabels():
        tick.set_fontname(PLOT_FONT)
    plt.show()


#    raise ValueError('breakpoint')



    df_subj = pd.read_csv(DATA_DIR + 'subj_track_iou.csv', header=0,
                          index_col='alg')

    df_subj = df_subj.T
    df_subj['LK'] = 1-df_subj['LK']
    df_subj['FRLK'] = 1-df_subj['FRLK']
    df_subj['BFLK-G'] = 1-df_subj['BFLK-G']
    df_subj['BFLK-T'] = 1-df_subj['BFLK-T']
    df_subj['SBLK-G'] = 1-df_subj['SBLK-G']
    df_subj['SBLK-T'] = 1-df_subj['SBLK-T']


    print(df_subj)

    track_colors = ['#f781bf','#a65628','#377eb8','#377eb8','#984ea3','#984ea3']
    sns.set()
    ax = df_subj.plot(kind='bar', color=track_colors, rot=0)
    bars = ax.patches
#    patterns =('-', '+', 'x','/','//','O','o','\\','\\\\')
    patterns = ('','','','////','','////')
    hatches = [p for p in patterns for i in range(len(df_subj))]
    for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    L = ax.legend(loc='upper left')
    plt.setp(L.texts, family=PLOT_FONT)
    ax.set_xlabel('Subject', fontname=PLOT_FONT)
    ax.set_ylabel('Jaccard Distance (1-IoU)', fontname=PLOT_FONT)
    for tick in ax.get_xticklabels():
        tick.set_fontname(PLOT_FONT)
    for tick in ax.get_yticklabels():
        tick.set_fontname(PLOT_FONT)
    plt.show()

#    raise ValueError('breakpoint')

    df_csa = pd.read_csv(DATA_DIR_SUB3 + 'ground_truth_csa.csv',
                                index_col=False, header=0, names=['csa-GT'])
    df_t = pd.read_csv(DATA_DIR_SUB3 + 'ground_truth_thickness.csv',
                              index_col=False, header=0, names=['t-GT'])
    df_tr = pd.read_csv(DATA_DIR_SUB3 +
                               'ground_truth_thickness_ratio.csv',
                               index_col=False, header=0, names=['tr-GT'])
    df_iou = pd.read_csv(DATA_DIR_SUB3 + '/LK/iou_series.csv',
                                index_col=False, header=0, names=['iou-LK'])

    for tracker in TRACKER_STRINGS:
        datapath_csa = DATA_DIR_SUB3 + tracker + '/tracking_csa.csv'
        datapath_t = DATA_DIR_SUB3 + tracker + '/tracking_thickness.csv'
        datapath_tr = DATA_DIR_SUB3 + tracker + '/tracking_thickness_ratio.csv'
        datapath_iou = DATA_DIR_SUB3 + tracker + '/iou_series.csv'
        df_csa[tracker] = pd.read_csv(datapath_csa)
        df_t[tracker] = pd.read_csv(datapath_t)
        df_tr[tracker] = pd.read_csv(datapath_tr)
        df_iou[tracker] = pd.read_csv(datapath_iou)

    df_csa = df_csa.loc[df_csa['csa-GT'] > 0]
    df_t = df_t.loc[df_t['t-GT'] > 0]
    df_tr = df_tr.loc[df_tr['tr-GT'] > 0]
    df_iou = df_iou.loc[df_iou['iou-LK'] > 1e-3]



    for tracker in TRACKER_STRINGS:
        df_csa[tracker] = abs(df_csa[tracker]-df_csa['csa-GT'])/df_csa['csa-GT']
        df_t[tracker] = abs(df_t[tracker]-df_t['t-GT'])/df_t['t-GT']
        df_tr[tracker] = abs(df_tr[tracker]-df_tr['tr-GT'])/df_tr['tr-GT']
        df_iou[tracker + '-JD'] = 1-df_iou[tracker]



    print(df_csa.mean())
    print(df_t.mean())
    print(df_tr.mean())
    print(df_iou.mean())

#    sns.set()
#    fig, axs = plt.subplots(6)
#    axs[0].plot(df_iou['LK-JD'])
#    axs[1].plot(df_iou['FRLK-JD'])
#    axs[2].plot(df_iou['BFLK-G-JD'])
#    axs[3].plot(df_iou['BFLK-T-JD'])
#    axs[4].plot(df_iou['SBLK-G-JD'])
#    axs[5].plot(df_iou['SBLK-T-JD'])
#    axs[0].plot(df_t['LK'])
#    axs[1].plot(df_t['FRLK'])
#    axs[2].plot(df_t['BFLK-G'])
#    axs[3].plot(df_t['BFLK-T'])
#    axs[4].plot(df_t['SBLK-G'])
#    axs[5].plot(df_t['SBLK-T'])

#    plt.show()

#    sns.set()
#
#    fig, axs = plt.subplots(4)
#    fig.suptitle('test plot')
#    axs[0].plot(data1.data_emg.data)
#    axs[1].plot(data1.data_amg.data)
#    axs[2].plot(data1.data_force.data)
#    axs[3].plot(data1.data_ultrasound.data)
#
#    plt.show()


if __name__ == "__main__":
    main()
