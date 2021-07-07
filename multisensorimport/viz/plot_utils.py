#!/usr/bin/env python3
"""Utility functions for plotting."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
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

    num_subplots = 3

    fig, axs = plt.subplots(num_subplots)

    axs[0].plot(trialdata.df['force'], 'k')
    axs[1].plot(trialdata.df['us'], color='C0')
    axs[2].plot(trialdata.df['emg'], color='C1')
    axs[2].set_xlabel('time (s)', fontname=plot_font)
    axs[0].plot(trialdata.df['traj'], 'k', linestyle='dotted')

    if not no_titles:
        tstring = 'correlation time series (force tracking): Sub' + trialdata.subj
        fig.suptitle(tstring, fontname=plot_font)

    axs[0].set(ylabel='force')
    axs[1].set(ylabel='deformation')
    axs[2].set(ylabel='activation')

    fig.text(0.04, 0.5, 'normalized signal value', va='center',
             rotation='vertical', fontname=plot_font, fontsize='large')

    axs[0].xaxis.set_visible(False)
    axs[1].xaxis.set_visible(False)

    for i in range(num_subplots):
        for tick in axs[i].get_xticklabels():
            tick.set_fontname(plot_font)
        for tick in axs[i].get_yticklabels():
            tick.set_fontname(plot_font)

    plt.tight_layout()

    plt.show()

def gen_tracking_time_plot(trialdata_us, trialdata_emg, no_titles=False, plot_font=PLOT_FONT):
    """Generate time series plot tracked sEMG and ultrasound data.
    Args:
        trialdata_us (pandas.DataFrame): dataobj.TrialData object containing
            data from ultrasound tracking trial to to be plotted
        trialdata_emg (pandas.DataFrame): dataobj.TrialData object containing
            data from sEMG tracking trial to to be plotted
        no_titles (bool): whether to omit axis/title labels that are redundant
            with eventual use case (e.g., copying to table for publication)
        plot_font (str): desired matplotlib font family
    """
    register_matplotlib_converters()
    sns.set()

    num_subplots = 2

    fig, axs = plt.subplots(num_subplots)

    axs[0].plot(trialdata_us.df['us'], color='C0')
    axs[0].plot(trialdata_us.df['traj'], color='k', linestyle='dotted')
    axs[1].plot(trialdata_emg.df['emg'], color='C1')
    axs[1].plot(trialdata_emg.df['traj'], color='k', linestyle='dotted')
    axs[1].set_xlabel('time (s)', fontname=plot_font)

    if not no_titles:
        tstring = 'trajectory tracking time series (deformation/activation tracking): Sub' + trialdata_us.subj
        fig.suptitle(tstring, fontname=plot_font)

    axs[0].set(ylabel='deformation')
    axs[1].set(ylabel='activation')

    fig.text(0.04, 0.5, 'normalized signal value', va='center',
             rotation='vertical', fontname=plot_font, fontsize='large')

    axs[0].xaxis.set_visible(False)

    for i in range(num_subplots):
        for tick in axs[i].get_xticklabels():
            tick.set_fontname(plot_font)
        for tick in axs[i].get_yticklabels():
            tick.set_fontname(plot_font)

    plt.tight_layout()

    plt.show()


def gen_trajtype_corr_plot(df_corr, plot_font=PLOT_FONT):
    """Generate bar plot showing correlation across trajectory types.

    Args:
        df_corr (pandas.DataFrame): frame containing data to be plotted
        plot_font (str): desired matplotlib font family
    """
    df_corr = df_corr.sort_values('index_corr')
    sns.set()
    ax = sns.barplot(x='index', y='value', hue='variable', data=df_corr,
                     hue_order=['deformation', 'activation'])
    plt.axvline(x = 3.5, color='#cccccc')
    L = ax.legend(loc='lower left', ncol=1, framealpha=1)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Trajectory Type', fontname=plot_font)
    ax.set_ylabel('CC(·, force)', fontname=plot_font)
    i = 0
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
        i = i+1
        if i == 5:
            tick.set_fontweight('bold')
        else:
            tick.set_style('italic')
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
    plt.tight_layout()
    plt.show()

def gen_subj_corr_plot(df_corr, plot_font=PLOT_FONT):
    """Generate bar plot showing correlation across subjects.

    Args:
        df_corr (pandas.DataFrame): frame containing data to be plotted
        plot_font (str): desired matplotlib font family
    """
    df_corr = df_corr.sort_values('subj')
    sns.set()
    df_corr_agg = df_corr.loc[df_corr['index'] == 'ALL']
    ax = sns.barplot(x='subj', y='value', hue='variable', data=df_corr_agg)
    L = ax.legend(loc='lower left', ncol=1, framealpha=1)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Subject', fontname=plot_font)
    ax.set_ylabel('CC(·, force)', fontname=plot_font)
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
    plt.tight_layout()
    plt.show()

def gen_trajtype_err_plot(df_err, plot_font=PLOT_FONT):
    """Generate bar plot showing tracking error across trajectory types.

    Args:
        df_err (pandas.DataFrame): frame containing data to be plotted
        plot_font (str): desired matplotlib font family
    """
    df_err = df_err.sort_values('index_err')
    sns.set()
    ax = sns.barplot(x='index', y='value', hue='variable', data=df_err,
                     hue_order=['deformation', 'activation'])
    plt.axvline(x = 3.5, color='#cccccc')
    L = ax.legend(loc='upper left', ncol=1, framealpha=1)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Trajectory Type', fontname=plot_font)
    ax.set_ylabel('RMS Error (fractional)', fontname=plot_font)
    i = 0
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
        i = i+1
        if i == 5:
            tick.set_fontweight('bold')
        else:
            tick.set_style('italic')
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
    plt.tight_layout()
    plt.show()


def gen_subj_err_plot(df_err, plot_font=PLOT_FONT):
    """Generate bar plot showing tracking error across subjects.

    Args:
        df_err (pandas.DataFrame): frame containing data to be plotted
        plot_font (str): desired matplotlib font family
    """
    df_err = df_err.sort_values('subj')
    sns.set()
    df_err_agg = df_err.loc[df_err['index'] == 'ALL']
    ax = sns.barplot(x='subj', y='value', hue='variable', data=df_err_agg)
    L = ax.legend(loc='upper right', ncol=1, framealpha=1)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('Subject', fontname=plot_font)
    ax.set_ylabel('RMS Error (fractional)', fontname=plot_font)
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
    plt.tight_layout()
    plt.show()

def gen_survey_box_plot(df_us, df_emg, plot_font=PLOT_FONT):
    """Generate box plot of subjects' tracking mode preferences.

    Args:
        df_us (pandas.DataFrame): dataframe containing ultrasound tracking
            preference data
        df_emg (pandas.DataFrame): dataframe containing sEMG tracking
            preference data
        plot_font (str): desired matplotlib font family
    """
    register_matplotlib_converters()
    sns.set()
    cdf = pd.concat([df_us, df_emg])
    cdf = cdf.rename(columns={'difficulty': r'$\bf{difficulty}$' '\n (1 = hard, \n 7 = easy)',
                              'match': r'$\bf{force\ match}$' '\n (1 = no match, \n 7 = perfect match)',
                              'responsivity': r'$\bf{responsivity}$' '\n (1 = slow, \n 7 = fast)'})
    mdf = cdf.melt(id_vars=['subj', 'sensor'])
    ax = sns.boxplot(y='variable', x='value', hue='sensor', data=mdf)

    L = ax.legend(loc='lower left', ncol=1, framealpha=1)
    plt.setp(L.texts, family=plot_font)
    ax.set_xlabel('', fontname=plot_font)
    ax.set_ylabel('', fontname=plot_font)
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
        tick.set_style('italic')
    plt.tight_layout()
    plt.show()

def gen_survey_comp_box_plot(df_comp, plot_font=PLOT_FONT):
    """Generate box plot of subjects' tracking mode preferences.

    Args:
        df_comp (pandas.DataFrame): dataframe containing comparative tracker
            preference data
        plot_font (str): desired matplotlib font family
    """
    register_matplotlib_converters()
    sns.set()
    df_comp = df_comp.rename(columns={'overall preference': 'overall \n preference'})
    mdf_comp = df_comp.melt(id_vars=['subj'])
    ax = sns.boxplot(y='variable', x='value', data=mdf_comp, color='C4')
    plt.axhline(y=2.5, color='#cccccc')
    ax.set_xlabel('', fontname=plot_font)
    ax.set_ylabel('', fontname=plot_font)
    labels = [int(label) for label in ax.get_xticks().tolist()]
    labels[1] = '1 \n (deformation \n preferred)'
    labels[7] = '7 \n (activation \n preferred)'
    ax.set_xticklabels(labels)
    for tick in ax.get_xticklabels():
        tick.set_fontname(plot_font)
    for tick in ax.get_yticklabels():
        tick.set_fontname(plot_font)
        tick.set_fontweight('bold')
    plt.tight_layout()
    plt.show()
