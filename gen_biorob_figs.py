#!/usr/bin/env python3
"""Generate all plots and data published in BIOROB 2020.

Example:
    Once filepaths are set appropriately, run this function via

        $ python gen_biorob_figs.py
"""

import os
import pandas as pd

from multisensorimport.viz import plot_utils

TERM_DIM = os.popen('stty size', 'r').read().split()

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/'

DATA_DIR_SUB1 = DATA_DIR + 'sub1/wp5t11/'
DATA_DIR_SUB2 = DATA_DIR + 'sub2/wp5t28/'
DATA_DIR_SUB3 = DATA_DIR + 'sub3/wp5t33/'
DATA_DIR_SUB4 = DATA_DIR + 'sub4/wp5t34/'
DATA_DIR_SUB5 = DATA_DIR + 'sub5/wp5t37/'

TRACKER_STRINGS = ['LK','FRLK','BFLK-G','BFLK-T','SBLK-G','SBLK-T']

def main():

    # generate angle correlation plot
    df_ang = pd.read_csv(DATA_DIR + 'ang_corr2.csv', header=0, index_col=0).T

    print_header('[SIGNAL]-FORCE CORRELATION ACROSS ANGLES (SUB1)')
    print(df_ang)
    plot_utils.gen_ang_plot(df_ang)

    # generate subject correlation plot
    df_subj = pd.read_csv(DATA_DIR + 'subj_corr2.csv', header=0,
                          index_col=0).T

    print_header('[SIGNAL]-FORCE CORRELATION ACROSS SUBJECTS (69deg)')
    print(df_subj)
    plot_utils.gen_subj_plot(df_subj)

    # generate tracking accuracy plot
    df_means, df_stds, df_sems = gen_tracking_dfs()

    print_header('TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - MEAN')
    print(df_means)
    print_header('TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - STDDEV')
    print(df_stds)
    print_header('TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - STDERR')
    print(df_sems)
    plot_utils.gen_tracking_error_plot(df_means, df_stds)


    # generate example tracking data table
    df_csa = pd.read_csv(DATA_DIR_SUB3 + 'ground_truth_csa.csv',
                                index_col=False, header=0, names=['csa-GT'])
    df_t = pd.read_csv(DATA_DIR_SUB3 + 'ground_truth_thickness.csv',
                              index_col=False, header=0, names=['t-GT'])
    df_tr = pd.read_csv(DATA_DIR_SUB3 +
                               'ground_truth_thickness_ratio.csv',
                               index_col=False, header=0, names=['tr-GT'])
    df_iou = pd.read_csv(DATA_DIR_SUB3 + 'LK/iou_series.csv',
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
        df_iou[tracker] = 1-df_iou[tracker]

    df_iou_mean = df_iou.mean()[TRACKER_STRINGS].to_frame()
    df_csa_mean = df_csa.mean()[TRACKER_STRINGS].to_frame()
    df_t_mean = df_t.mean()[TRACKER_STRINGS].to_frame()
    df_tr_mean = df_tr.mean()[TRACKER_STRINGS].to_frame()

    df_means = df_iou_mean.copy()
    df_means.rename(columns={0:'Jaccard Distance'}, inplace=True)
    df_means['CSA'] = df_csa_mean[0]
    df_means['T'] = df_t_mean[0]
    df_means['AR'] = df_tr_mean[0]
    print_header('EXAMPLE TRACKING ERROR (SUB3) - MEAN')
    print(df_means)

    df_iou_std = df_iou.std()[TRACKER_STRINGS].to_frame()
    df_csa_std = df_csa.std()[TRACKER_STRINGS].to_frame()
    df_t_std = df_t.std()[TRACKER_STRINGS].to_frame()
    df_tr_std = df_tr.std()[TRACKER_STRINGS].to_frame()

    df_stds = df_iou_std.copy()
    df_stds.rename(columns={0:'Jaccard Distance'}, inplace=True)
    df_stds['CSA'] = df_csa_std[0]
    df_stds['T'] = df_t_std[0]
    df_stds['AR'] = df_tr_std[0]
    print_header('EXAMPLE TRACKING ERROR (SUB3) - STDDEV')
    print(df_stds)


def gen_tracking_dfs():
    """Generate tracking error (Jaccard distance) data frames from raw IoU time
    series CSVs.

    Args:
        sub_dirs (list): list of file paths to each IoU CSV, ordered Sub1-SubN

    Returns:
        pandas.DataFrame mean Jaccard distance errors
        pandas.DataFrame standard deviation Jaccard distance errors
        pandas.DataFrame standard error Jaccard distance errors
    """
    sub1_iou = gen_iou_vals(DATA_DIR_SUB1)
    sub2_iou = gen_iou_vals(DATA_DIR_SUB2)
    sub3_iou = gen_iou_vals(DATA_DIR_SUB3)
    sub4_iou = gen_iou_vals(DATA_DIR_SUB4)
    sub5_iou = gen_iou_vals(DATA_DIR_SUB5)

    df_means = pd.DataFrame(index=TRACKER_STRINGS,
                            columns=['Sub1','Sub2','Sub3','Sub4','Sub5'])
    df_means['Sub1'] = sub1_iou.mean()
    df_means['Sub2'] = sub2_iou.mean()
    df_means['Sub3'] = sub3_iou.mean()
    df_means['Sub4'] = sub4_iou.mean()
    df_means['Sub5'] = sub5_iou.mean()

    df_stds = pd.DataFrame(index=TRACKER_STRINGS,
                            columns=['Sub1','Sub2','Sub3','Sub4','Sub5'])
    df_stds['Sub1'] = sub1_iou.std()
    df_stds['Sub2'] = sub2_iou.std()
    df_stds['Sub3'] = sub3_iou.std()
    df_stds['Sub4'] = sub4_iou.std()
    df_stds['Sub5'] = sub5_iou.std()

    df_sems = pd.DataFrame(index=TRACKER_STRINGS,
                            columns=['Sub1','Sub2','Sub3','Sub4','Sub5'])
    df_sems['Sub1'] = sub1_iou.sem()
    df_sems['Sub2'] = sub2_iou.sem()
    df_sems['Sub3'] = sub3_iou.sem()
    df_sems['Sub4'] = sub4_iou.sem()
    df_sems['Sub5'] = sub5_iou.sem()

    df_means = df_means.T
    df_stds = df_stds.T
    df_sems = df_sems.T

    return df_means, df_stds, df_sems


def gen_iou_vals(subj_dir):
    df_iou = pd.read_csv(subj_dir + 'LK/iou_series.csv',
                                index_col=False, header=0, names=['LK'])

    for tracker in TRACKER_STRINGS:
        if tracker != 'LK':
            datapath_iou = subj_dir + tracker + '/iou_series.csv'
            df_iou[tracker] = pd.read_csv(datapath_iou)

    df_iou = df_iou.loc[df_iou['LK'] > 1e-3]


    for tracker in TRACKER_STRINGS:
        df_iou[tracker] = 1-df_iou[tracker]

    return df_iou

def print_header(header_string):
    print('\n')# + '-'*int(TERM_DIM[1]))
    print(header_string)
    print('-'*int(TERM_DIM[1]))


if __name__ == "__main__":
    main()
