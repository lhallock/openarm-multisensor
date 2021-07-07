#!/usr/bin/env python3
"""Utility functions for basic dataframe building/statistics."""
import pandas as pd
from multisensorimport.dataobj import trialdata as td

def gen_corr_df(data_dir, subj_dirs, trial_filename):
    """Aggregate correlation values from multiple subjects.

    Args:
        data_dir (str): path to high-level data directory
        subj_dirs (list): list of all subject identifiers/directories
        trial_filename (str): filename of trial on which to generate
            correlation data

    Returns:
        pandas.DataFrame melted correlation table
    """
    corr_df_list = []
    for d in subj_dirs:
        readpath = data_dir + d + '/' + trial_filename

        data = td.TrialData.from_pickle(readpath, d)
        corrs_us = pd.Series(data.get_corrs('us'))
        corrs_emg = pd.Series(data.get_corrs('emg'))

        df_corrs = pd.DataFrame({'deformation': corrs_us, 'activation': corrs_emg})
        df_corrs_melt = pd.melt(df_corrs.reset_index(),
                      id_vars='index',value_vars=['deformation','activation'])
        df_corrs_melt['subj'] = data.subj

        corr_df_list.append(df_corrs_melt)

    df_all_corrs = pd.concat(corr_df_list)

    corr_ind_dict = {'ALL': 4, 'sustained': 0, 'ramp': 1, 'step': 2, 'sine': 3}
    df_all_corrs['index_corr'] = df_all_corrs['index'].map(corr_ind_dict)
    df_all_corrs['subj'] = df_all_corrs['subj'].apply(pd.to_numeric)

    return df_all_corrs

def gen_err_df(data_dir, subj_dirs, trial_us_filename, trial_emg_filename):
    """Aggregate tracking error values from multiple subjects.

    Args:
        data_dir (str): path to high-level data directory
        subj_dirs (list): list of all subject identifiers/directories
        trial_us_filename (str): filename of trial on which to generate
            ultrasound tracking error data
        trial_emg_filename (str): filename of trial on which to generate
            sEMG tracking error data

    Returns:
        pandas.DataFrame melted tracking error table
    """
    err_df_list = []
    for d in subj_dirs:
        readpath_us = data_dir + d + '/' + trial_us_filename
        readpath_emg = data_dir + d + '/' + trial_emg_filename

        data_us = td.TrialData.from_pickle(readpath_us, d)
        data_emg = td.TrialData.from_pickle(readpath_emg, d)
        errors_us = pd.Series(data_us.get_tracking_errors('us'))
        errors_emg = pd.Series(data_emg.get_tracking_errors('emg'))

        df_errors = pd.DataFrame({'deformation': errors_us, 'activation': errors_emg})
        df_errors_melt = pd.melt(df_errors.reset_index(), id_vars='index',
                                 value_vars=['deformation', 'activation'])
        df_errors_melt['subj'] = d

        err_df_list.append(df_errors_melt)

    df_all_errors = pd.concat(err_df_list)

    err_ind_dict = {'ALL': 4, 'sustained': 0, 'ramp': 1, 'step': 2, 'sine': 3}
    df_all_errors['index_err'] = df_all_errors['index'].map(err_ind_dict)
    df_all_errors['subj'] = df_all_errors['subj'].apply(pd.to_numeric)

    return df_all_errors
