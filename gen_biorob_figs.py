#!/usr/bin/env python3
"""Generate all plots and data published in BIOROB 2020.

Example:
    Once filepaths are set appropriately, run this function via

        $ python gen_biorob_figs.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ensure plots use Type1 fonts when exported to PDF or PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

PLOT_FONT = 'Open Sans'

DATA_DIR = '/home/lhallock/Dropbox/DYNAMIC/Research/MM/code/openarm-multisensor/sandbox/data/FINAL/'

DATA_DIR_SUB1 = DATA_DIR + 'sub1/wp5t11/'
DATA_DIR_SUB2 = DATA_DIR + 'sub2/wp5t28/'
DATA_DIR_SUB3 = DATA_DIR + 'sub3/wp5t33/'
DATA_DIR_SUB4 = DATA_DIR + 'sub4/wp5t34/'
DATA_DIR_SUB5 = DATA_DIR + 'sub5/wp5t37/'

TRACKER_STRINGS = ['LK','FRLK','BFLK-G','BFLK-T','SBLK-G','SBLK-T']

def main():

    # generate angle correlation plot
#    df_ang = pd.read_csv(DATA_DIR + 'ang_corr.csv', header=0,
#                         index_col='measure')

#    df_ang = df_ang.T

#    print(df_ang)

    df_ang = pd.read_csv(DATA_DIR + 'ang_corr2.csv', header=0, index_col=0)

    df_ang = df_ang.T

    print(df_ang)
#    raise ValueError('breakpoint')



    ang_colors = ['#fdae6b','#f16d13','#41b6c4','#41b6c4','#225ea8','#225ea8','#081d58','#081d58']
    styles = ['-','-','-','--','-','--','-','--']
    sns.set()
    ax = df_ang.plot(kind='line', style=styles, color=ang_colors, rot=0)
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

    df_subj_def = pd.read_csv(DATA_DIR + 'subj_corr2.csv', header=0,
                               index_col=0)
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
    df_iou = pd.read_csv(DATA_DIR_SUB3 + 'LK/iou_series.csv',
                                index_col=False, header=0, names=['iou-LK'])

    print(DATA_DIR_SUB3 + '/LK/iou_series.csv')
    print(df_iou)
#    raise ValueError('breakpoint')

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


    print('MEANS-----------------------')
    print(df_csa.mean())
    print(df_t.mean())
    print(df_tr.mean())
    print(df_iou.mean())
    print('STDEVS-----------------------')
    print(df_csa.std())
    print(df_t.std())
    print(df_tr.std())
    print(df_iou.std())


    print('NEW TRIAL')

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

    print(df_means)
    print(df_stds)
    print(df_sems)

    track_colors = ['#f781bf','#a65628','#377eb8','#377eb8','#984ea3','#984ea3']
    sns.set()
    ax = df_means.plot(kind='bar', color=track_colors, rot=0, yerr=df_stds,
                       error_kw=dict(lw=0.5, capsize=0, capthick=0))
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


if __name__ == "__main__":
    main()
