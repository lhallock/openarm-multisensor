#!/usr/bin/env python3
"""Generate all plots and data published in BIOROB 2020.

Example:
    Once filepaths are set appropriately, run this function via

        $ python gen_biorob_figs.py
"""

import pandas as pd

from multisensorimport.viz import plot_utils, print_utils

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/'

DATA_DIR_SUB1 = DATA_DIR + 'sub1/wp5t11/'
DATA_DIR_SUB2 = DATA_DIR + 'sub2/wp5t28/'
DATA_DIR_SUB3 = DATA_DIR + 'sub3/wp5t33/'
DATA_DIR_SUB4 = DATA_DIR + 'sub4/wp5t34/'
DATA_DIR_SUB5 = DATA_DIR + 'sub5/wp5t37/'

TRACKER_STRINGS = ('LK', 'FRLK', 'BFLK-G', 'BFLK-T', 'SBLK-G', 'SBLK-T')

def main():
    """Generate all plots reported in [PUBLICATION FORTHCOMING].
    """

    # generate angle correlation plot
    df_ang = pd.read_csv(DATA_DIR + 'ang_corr2.csv', header=0, index_col=0).T

    print_utils.print_header('[SIGNAL]-FORCE CORRELATION ACROSS ANGLES (SUB1)')
    print(df_ang)
    plot_utils.gen_ang_plot(df_ang)

    print_utils.print_div()

    # generate subject correlation plot
    df_subj = pd.read_csv(DATA_DIR + 'subj_corr2.csv', header=0,
                          index_col=0).T

    print_utils.print_header('[SIGNAL]-FORCE CORRELATION ACROSS SUBJECTS (69deg)')
    print(df_subj)
    plot_utils.gen_subj_plot(df_subj)

    print_utils.print_div()

    # generate tracking accuracy plot
    subj_dirs = [DATA_DIR_SUB1, DATA_DIR_SUB2, DATA_DIR_SUB3, DATA_DIR_SUB4, DATA_DIR_SUB5]
    df_means, df_stds, df_sems = gen_tracking_dfs(subj_dirs)

    print_utils.print_header('TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - MEAN')
    print(df_means)
    print_utils.print_header('TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - STDDEV')
    print(df_stds)
    print_utils.print_header('TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - STDERR')
    print(df_sems)
    plot_utils.gen_tracking_error_plot(df_means, df_stds)

    print_utils.print_div()

    # generate example tracking data table
    df_ex_means, df_ex_stds = gen_ex_tracking_df(DATA_DIR_SUB3)
    print_utils.print_header('EXAMPLE TRACKING ERROR (SUB3) - MEAN')
    print(df_ex_means)
    print_utils.print_header('EXAMPLE TRACKING ERROR (SUB3) - STDDEV')
    print(df_ex_stds)

def gen_ex_tracking_df(subj_dir):
    """Generate tracking error (Jaccard distance, CSA, T, AR) data frames from raw time
    series CSVs for single subject.

    Args:
        subj_dir (str): path to subject data directory, including final '/'

    Returns:
        pandas.DataFrame mean errors (Jaccard distance, CSA, T, AR)
        pandas.DataFrame standard deviation errors (Jaccard distance, CSA, T, AR)
    """
    df_iou = gen_jd_vals(subj_dir)
    df_csa = gen_def_err_vals(subj_dir, 'CSA')
    df_t = gen_def_err_vals(subj_dir, 'T')
    df_tr = gen_def_err_vals(subj_dir, 'AR')

    df_iou_mean = df_iou.mean().to_frame()
    df_csa_mean = df_csa.mean().to_frame()
    df_t_mean = df_t.mean().to_frame()
    df_tr_mean = df_tr.mean().to_frame()

    df_means = df_iou_mean.copy()
    df_means.rename(columns={0:'Jaccard Distance'}, inplace=True)
    df_means['CSA'] = df_csa_mean[0]
    df_means['T'] = df_t_mean[0]
    df_means['AR'] = df_tr_mean[0]

    df_iou_std = df_iou.std().to_frame()
    df_csa_std = df_csa.std().to_frame()
    df_t_std = df_t.std().to_frame()
    df_tr_std = df_tr.std().to_frame()

    df_stds = df_iou_std.copy()
    df_stds.rename(columns={0:'Jaccard Distance'}, inplace=True)
    df_stds['CSA'] = df_csa_std[0]
    df_stds['T'] = df_t_std[0]
    df_stds['AR'] = df_tr_std[0]

    return df_means, df_stds

def gen_tracking_dfs(subj_dirs, tracker_strings=TRACKER_STRINGS):
    """Generate tracking error (Jaccard distance) data frames from raw IoU time
    series CSVs of multiple subjects.

    Args:
        subj_dirs (list): list of file paths to each IoU CSV, ordered Sub1-SubN
        tracker_strings (list): list of tracker string identifiers (i.e.,
            directory names)

    Returns:
        pandas.DataFrame mean Jaccard distance errors
        pandas.DataFrame standard deviation Jaccard distance errors
        pandas.DataFrame standard error Jaccard distance errors
    """
    # determine data columns
    cols = []
    for i, _ in enumerate(subj_dirs):
        cols.append('Sub' + str(i + 1))

    # initialize data frame
    df_means = pd.DataFrame(index=tracker_strings, columns=cols)
    df_stds = df_means.copy()
    df_sems = df_means.copy()

    # aggregate data from each subject
    for i, subj_dir in enumerate(subj_dirs):
        df_col = 'Sub' + str(i + 1)
        jds = gen_jd_vals(subj_dir)
        df_means[df_col] = jds.mean()
        df_stds[df_col] = jds.std()
        df_sems[df_col] = jds.sem()

    df_means = df_means.T
    df_stds = df_stds.T
    df_sems = df_sems.T

    return df_means, df_stds, df_sems

def gen_def_err_vals(subj_dir, metric, tracker_strings=TRACKER_STRINGS):
    """Aggregate table of per-frame deformation metric error for each tracker.

    Args:
        subj_dir (str): path to subject data directory, including final '/'
        metric (str): metric identifier ('CSA', 'T', or 'AR')
        tracker_strings (list): list of tracker string identifiers (i.e.,
            directory names)

    Returns:
        pandas.DataFrame of deformation metric errors, indexed by frame number
    """
    if metric == 'CSA':
        df_metric = pd.read_csv(subj_dir + 'ground_truth_csa.csv',
                                index_col=False, header=0, names=['GT'])
    elif metric == 'T':
        df_metric = pd.read_csv(subj_dir + 'ground_truth_thickness.csv',
                                index_col=False, header=0, names=['GT'])
    elif metric == 'AR':
        df_metric = pd.read_csv(subj_dir +
                                'ground_truth_thickness_ratio.csv',
                                index_col=False, header=0, names=['GT'])
    else:
        raise ValueError('unknown deformation metric')

    for tracker in tracker_strings:
        if metric == 'CSA':
            datapath = subj_dir + tracker + '/tracking_csa.csv'
        elif metric == 'T':
            datapath = subj_dir + tracker + '/tracking_thickness.csv'
        elif metric == 'AR':
            datapath = subj_dir + tracker + '/tracking_thickness_ratio.csv'
        else:
            raise ValueError('unknown deformation metric')

        df_metric[tracker] = pd.read_csv(datapath)

    df_metric = df_metric.loc[df_metric['GT'] > 0]

    for tracker in tracker_strings:
        df_metric[tracker] = abs(df_metric[tracker]-df_metric['GT'])/df_metric['GT']

    return df_metric

def gen_jd_vals(subj_dir, tracker_strings=TRACKER_STRINGS):
    """Aggregate table of per-frame Jaccard distance error for each tracker.

    Args:
        subj_dir (str): path to subject data directory, including final '/'
        tracker_strings (list): list of tracker string identifiers (i.e.,
            directory names)

    Returns:
        pandas.DataFrame of Jaccard distance errors, indexed by frame number
    """
    df_iou = pd.read_csv(subj_dir + 'LK/iou_series.csv',
                         index_col=False, header=0, names=['LK'])

    for tracker in tracker_strings:
        if tracker != 'LK':
            datapath_iou = subj_dir + tracker + '/iou_series.csv'
            df_iou[tracker] = pd.read_csv(datapath_iou)

    df_iou = df_iou.loc[df_iou['LK'] > 1e-3]


    for tracker in tracker_strings:
        df_iou[tracker] = 1-df_iou[tracker]

    return df_iou


if __name__ == "__main__":
    main()
