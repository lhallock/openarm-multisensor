#!/usr/bin/env python3
"""Utility functions for data collation and analysis.

This module contains functions used during processing of TrialData and
TimeSeriesData objects.
"""

import pandas as pd
import numpy.polynomial.polynomial as poly


def build_data_series(data, col=0):
    """Build pandas Series object from column of TimeSeriesData object.

    Args:
        data (TimeSeriesData): data object from which to extract series
        col (int): column of series to extract

    Returns:
        pandas.Series corresponding to data object with time values based on
            collection frequency; note that for published data, the data of
            collection is accurate but the start time is arbitrary
    """
    length = data.data_from_offset.shape[0]
    freq_pd_str = as_pd_freq(data.freq)
    index = pd.date_range('2017-01-27', periods=length, freq=freq_pd_str)
    return pd.Series(data.data_from_offset[:, col], index)


def fit_data_poly(times_in, data_in, times_out, order):
    """Fit polynomial to pandas series and return values for given out times.

    Args:
        times_in (pandas.DateTimeIndex): times corresponding to data_in values
        data_in (pandas.Series): data used for fitting
        times_out (pandas.DateTimeIndex): times at which to calculate value of
            output polynomial series
        order (int): desired order of polynomial to fit

    Returns:
        numpy.ndarray of data values of fitted polynomial at times times_out
    """
    # preprocess data, avoiding numerical instability
    times_arr = times_in.to_julian_date().to_numpy()-2457780
    data_arr = data_in.to_numpy()

    # fit polynomial
    p_coeffs = poly.polyfit(times_arr, data_arr, order)

    # generate and return polynomial computed at times_out
    times_out_arr = times_out.to_julian_date().to_numpy()-2457780
    return poly.polyval(times_out_arr, p_coeffs)



def as_pd_freq(freq):
    """Convert frequency expressed in Hz to pandas-compatible frequency.

    When pandas says they want frequency, what they're actually asking for
    is period, which is stupid and confusing. This method converts a
    frequency expressed in Hz to its equivalent pandas frequency, aka its
    period in microseconds. WARNING: Don't use this with data sample at
    >1MHz, since this will run into precision errors!

    Args:
        freq (float): data series frequency (in Hz), <1e6

    Returns:
        str pandas frequency corresponding to freq

    Raises:
        ValueError if freq is >1e6
    """
    if freq > 1e6:
        raise ValueError('Specified frequency is too high for this method',
                         'and will result in catastrophic precision loss.')

    freq_pd = int(1e6/freq)
    freq_pd_str = str(freq_pd) + 'U'
    return freq_pd_str


def offset_from_peak(peak_ind, freq, prepeak):
    """Compute data offset index from location of first peak.

    Args:
        peak_ind (int): location of first peak as index value
        freq (float): data frequency in Hz
        prepeak (int/float): desired start point of data stream in s prior
            to first peak value

    Returns:
        int index of desired offset in original data
    """
    offset = int(peak_ind - prepeak*freq)
    return offset


def build_corr_table(data_list, out_path=None, correlate='force'):
    """Build table containing correlation relationships from multiple trials.

    This function builds and returns a pandas DataFrame containing specified
    correlation relationships, and if a write path is specified, writes it to a
    CSV file.

    Args:
        data_list (list): list of TrialData objects whose correlations to plot
        out_path (str): out path for correlation table writing, if desired
        correlate (str): desired data series within each trial to correlate
            with

    Returns:
        pandas.DataFrame correlation table
    """
    first = True
    for data in data_list:
        # build correlation matrix and extract desired column
        label = str(data.subj) + 'wp' + str(data.wp)
        data_corr = data.df.corr()[correlate]

        # initialize dataframe if necessary
        if first:
            df_corr = pd.DataFrame({label: data_corr})
            first = False

        # otherwise, add new data to dataframe
        else:
            df_corr[label] = data_corr

    # write to file, if specified
    if out_path:
        df_corr.to_csv(out_path)

    return df_corr


from scipy.stats import pearsonr
import pandas as pd

def calculate_pvalues(df):
    """Calculate table of correlation p-values of given data frame.

    This method outputs a table just like that of pandas' corr() function, but
    with p-values instead of correlations. Stolen shamelessly from this post:
    https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance.

    Args:
        df (pandas.DataFrame): table on which to compute correlation

    Returns:
        pandas.DataFrame containing p-values, in the style of corr()
    """
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues



