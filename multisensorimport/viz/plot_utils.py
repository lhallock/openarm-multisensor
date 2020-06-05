#!/usr/bin/env python3
"""Utility functions for plotting.

This module contains functions used to plot time series data for [PUBLICATION
FORTHCOMING].
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ensure plots use Type1 fonts when exported to PDF or PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# plot defaults
PLOT_FONT = 'Open Sans'


def gen_ang_plot(df_ang, plot_font=PLOT_FONT):
    """Generate plot showing correlation across angles from provided data frame.

    Args:
        df_ang (pandas.DataFrame): frame containing data to be plotted
        plot_font (str): desired matplotlib font family
    """
    ang_colors = ['#fdae6b','#f16d13','#41b6c4','#41b6c4','#225ea8','#225ea8','#081d58','#081d58']
    styles = ['-','-','-','--','-','--','-','--']
    sns.set()
    ax = df_ang.plot(kind='line', style=styles, color=ang_colors, rot=0)
    L = ax.legend(loc='lower left', ncol=4)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Flexion Angle ($\degree$ from full extension)', fontname=plot_font)
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
    mod_subj_colors = ['#41b6c4','#41b6c4','#225ea8','#225ea8','#081d58','#081d58']
    sns.set()
    ax = df_subj.plot(kind='bar', color=mod_subj_colors, rot=0)
    bars = ax.patches
    patterns = ('','////','','////','','////')
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
    track_colors = ['#f781bf','#a65628','#377eb8','#377eb8','#984ea3','#984ea3']
    sns.set()
    ax = df_means.plot(kind='bar', color=track_colors, rot=0, yerr=df_stds,
                       error_kw=dict(lw=0.5, capsize=0, capthick=0))
    bars = ax.patches
    patterns = ('','','','////','','////')
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


