#!/usr/bin/env python3
"""Utility functions for plotting.

This module contains functions used to plot time series data for [PUBLICATION
FORTHCOMING].
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# deal with pandas index compatibility errors
from pandas.plotting import register_matplotlib_converters

# ensure plots use Type1 fonts when exported to PDF or PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# plot defaults
PLOT_FONT = 'Open Sans'


def gen_time_plot(trialdata, no_titles=False, plot_font=PLOT_FONT):
    """Generate time series plot of force, sEMG, and ultrasound data.

    Args:
        trialdata (pandas.DataFrame): dataobj.TrialData object containing data
            to to be plotted
        no_titles (bool): whether to omit axis/title labels that are redundant
            with eventual use case (e.g., copying to table for publication)
        plot_font (str): desired matplotlib font family
    """
    register_matplotlib_converters()
    sns.set()

    num_subplots = 6

    fig, axs = plt.subplots(num_subplots)

    plot_ind = trialdata.df.index.to_julian_date().to_numpy() - 2457780.5
    plot_ind = plot_ind * 24 * 60 * 60

    axs[0].plot(trialdata.df['force'], 'k')
    if not trialdata.force_only:
        axs[1].plot(trialdata.df['emg-bic'] * 1e3, color='#cccccc')
        axs[1].plot(trialdata.df['emg-abs-bic'] * 1e4, color='#fdae6b')
        axs[2].plot(trialdata.df['emg-brd'] * 1e3, color='#cccccc')
        axs[2].plot(trialdata.df['emg-abs-brd'] * 1e4, color='#f16913')
    axs[3].plot(trialdata.df['us-csa-dt'], color='#41b6c4')
    axs[4].plot(trialdata.df['us-t-dt'], color='#225ea8')
    axs[5].plot(plot_ind, trialdata.df['us-tr-dt'], color='#081d58')
    axs[5].set_xlabel('time (s)', fontname=plot_font)
    axs[5].xaxis.set_label_coords(1.0, -0.15)

    if not no_titles:
        tstring = trialdata.subj + ', ' + str(180 -
                                              int(trialdata.ang)) + '$\degree$'
        fig.suptitle(tstring, fontname=plot_font)
        axs[0].set(ylabel='f')
        axs[1].set(ylabel='sEMG-BIC')
        axs[2].set(ylabel='sEMG-BRD')
        axs[3].set(ylabel='CSA-DT')
        axs[4].set(ylabel='T-DT')
        axs[5].set(ylabel='AR-DT')

    axs[0].xaxis.set_visible(False)
    axs[1].xaxis.set_visible(False)
    axs[2].xaxis.set_visible(False)
    axs[3].xaxis.set_visible(False)
    axs[4].xaxis.set_visible(False)

    for i in range(num_subplots):
        for tick in axs[i].get_xticklabels():
            tick.set_fontname(plot_font)
        for tick in axs[i].get_yticklabels():
            tick.set_fontname(plot_font)

    plt.show()


def gen_debug_time_plot(trialdata):
    """Generate force/sEMG/ultrasound time series plot for fit debugging.

    Colors used in plotting:
        blue: unmodified signal
        orange: filtered signal (sEMG)
        red: polyfit trendline
        green: samples used in trendline generation
        black: detrended data

    Args:
        trialdata (pandas.DataFrame): dataobj.TrialData object containing data
            to to be plotted
        plot_font (str): desired matplotlib font family
    """
    register_matplotlib_converters()
    sns.set()

    tstring = 'FIT TEST: ' + trialdata.subj + ', ' + str(
        180 - int(trialdata.ang)) + '$\degree$'

    fig, axs = plt.subplots(7)
    fig.suptitle(tstring)
    axs[0].plot(trialdata.df['force'])
    axs[0].set(ylabel='force')
    if not trialdata.force_only:
        axs[1].plot(trialdata.df['emg-brd'])
        axs[1].plot(trialdata.df['emg-abs-brd'])
        axs[1].set(ylabel='emg-brd')
        axs[2].plot(trialdata.df['emg-bic'])
        axs[2].plot(trialdata.df['emg-abs-bic'])
    axs[2].set(ylabel='emg-bic')
    axs[3].plot(trialdata.df['us-csa'])
    axs[3].set(ylabel='us-csa')
    axs[3].plot(trialdata.df_dt['us-csa'], 'g-')
    axs[3].plot(trialdata.df['us-csa-fit'], 'r-')
    axs[4].plot(trialdata.df['us-t'])
    axs[4].set(ylabel='us-t')
    axs[4].plot(trialdata.df_dt['us-t'], 'g-')
    axs[4].plot(trialdata.df['us-t-fit'], 'r-')
    axs[5].plot(trialdata.df['us-t-dt'], 'k-')
    axs[5].set(ylabel='us-t-dt')
    axs[6].plot(trialdata.df['us-tr'])
    axs[6].set(ylabel='us-tr')
    axs[6].plot(trialdata.df_dt['us-tr'], 'g-')
    axs[6].plot(trialdata.df['us-tr-fit'], 'r-')

    plt.show()


def gen_ang_plot(df_ang, plot_font=PLOT_FONT):
    """Generate plot showing correlation across angles from provided data frame.

    Args:
        df_ang (pandas.DataFrame): frame containing data to be plotted
        plot_font (str): desired matplotlib font family
    """
    ang_colors = [
        '#fdae6b', '#f16d13', '#41b6c4', '#41b6c4', '#225ea8', '#225ea8',
        '#081d58', '#081d58'
    ]
    styles = ['-', '-', '-', '--', '-', '--', '-', '--']
    sns.set()
    ax = df_ang.plot(kind='line', style=styles, color=ang_colors, rot=0)
    L = ax.legend(loc='lower left', ncol=4)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Flexion Angle ($\degree$ from full extension)',
                  fontname=plot_font)
    ax.set_ylabel('CC(·,f)', fontname=plot_font)
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
    plt.show()


def gen_subj_plot(df_subj, plot_font=PLOT_FONT):
    """Generate plot showing correlation across subjects from provided data frame.

    Args:
        df_subj (pandas.DataFrame): frame containing data to be plotted
        plot_font (str): desired matplotlib font family
    """
    mod_subj_colors = [
        '#41b6c4', '#41b6c4', '#225ea8', '#225ea8', '#081d58', '#081d58'
    ]
    sns.set()
    ax = df_subj.plot(kind='bar', color=mod_subj_colors, rot=0)
    bars = ax.patches
    patterns = ('', '////', '', '////', '', '////')
    hatches = [p for p in patterns for i in range(len(df_subj))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    L = ax.legend(loc='lower left', ncol=3)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Subject', fontname=plot_font)
    ax.set_ylabel('CC(·,f)', fontname=plot_font)
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
    plt.show()


def gen_tracking_error_plot(df_means, df_stds, plot_font=PLOT_FONT):
    """Generate plot showing correlation across subjects from provided data frame.

    Args:
        df_means (pandas.DataFrame): frame containing mean error data to be plotted
        df_stds (pandas.DataFrame): frame contain standard deviations (or
            standard errors) to be plotted
        plot_font (str): desired matplotlib font family
    """
    track_colors = [
        '#f781bf', '#a65628', '#377eb8', '#377eb8', '#984ea3', '#984ea3'
    ]
    sns.set()
    ax = df_means.plot(kind='bar',
                       color=track_colors,
                       rot=0,
                       yerr=df_stds,
                       error_kw=dict(lw=0.5, capsize=0, capthick=0))
    bars = ax.patches
    patterns = ('', '', '', '////', '', '////')
    hatches = [p for p in patterns for i in range(len(df_means))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    L = ax.legend(loc='upper left')
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Subject', fontname=plot_font)
    ax.set_ylabel('Jaccard Distance (1-IoU)', fontname=plot_font)
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
    plt.show()
