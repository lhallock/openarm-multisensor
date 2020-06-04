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


def gen_ang_plot(df_ang, plot_font='Open Sans'):
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


