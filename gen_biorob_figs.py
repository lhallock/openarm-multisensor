#!/usr/bin/env python3
"""Generate all bar plots and correlation statistics for [PUBLICATION
FORTHCOMING]

Example:
    Once filepaths are set appropriately, run this function via

        $ python gen_biorob_figs.py
"""
import os

import pandas as pd

from multisensorimport.viz import plot_utils, print_utils, stats_utils

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + '/sandbox/data/FINAL/'

DATA_DIR_SUB1 = DATA_DIR + 'sub1/wp5t11/'
DATA_DIR_SUB2 = DATA_DIR + 'sub2/wp5t28/'
DATA_DIR_SUB3 = DATA_DIR + 'sub3/wp5t33/'
DATA_DIR_SUB4 = DATA_DIR + 'sub4/wp5t34/'
DATA_DIR_SUB5 = DATA_DIR + 'sub5/wp5t37/'


def main():
    """Generate all plots reported in [PUBLICATION FORTHCOMING].
    """
    # generate angle correlation plot
    df_ang = pd.read_csv(DATA_DIR + 'ang_corr.csv', header=0, index_col=0).T

    print_utils.print_header('[SIGNAL]-FORCE CORRELATION ACROSS ANGLES (SUB1)')
    print(df_ang)
    plot_utils.gen_ang_plot(df_ang)

    print_utils.print_div()

    # generate subject correlation plot
    df_subj = pd.read_csv(DATA_DIR + 'subj_corr.csv', header=0, index_col=0).T

    print_utils.print_header(
        '[SIGNAL]-FORCE CORRELATION ACROSS SUBJECTS (69deg)')
    print(df_subj)
    plot_utils.gen_subj_plot(df_subj)

    print_utils.print_div()

    # generate tracking accuracy plot
    subj_dirs = [
        DATA_DIR_SUB1, DATA_DIR_SUB2, DATA_DIR_SUB3, DATA_DIR_SUB4,
        DATA_DIR_SUB5
    ]
    df_means, df_stds, df_sems = stats_utils.gen_tracking_dfs(subj_dirs)

    print_utils.print_header(
        'TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - MEAN')
    print(df_means)
    print_utils.print_header(
        'TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - STDDEV')
    print(df_stds)
    print_utils.print_header(
        'TRACKING ERROR ACROSS SUBJECTS (JACCARD DISTANCE) - STDERR')
    print(df_sems)
    plot_utils.gen_tracking_error_plot(df_means, df_stds)

    print_utils.print_div()

    # generate example tracking data table
    df_ex_means, df_ex_stds = stats_utils.gen_ex_tracking_df(DATA_DIR_SUB3)
    print_utils.print_header('EXAMPLE TRACKING ERROR (SUB3) - MEAN')
    print(df_ex_means)
    print_utils.print_header('EXAMPLE TRACKING ERROR (SUB3) - STDDEV')
    print(df_ex_stds)


if __name__ == "__main__":
    main()
