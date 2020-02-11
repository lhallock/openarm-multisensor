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

