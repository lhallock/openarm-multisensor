#!/usr/bin/env python3
"""Utility functions for data collation and analysis.

This module contains functions used during processing of TrialData and
TimeSeriesData objects.
"""

import pandas as pd

def build_data_series(data, col=0):
    """Build pandas Series object from column of TimeSeriesData object.

    Args:
        data (TimeSeriesData): data object from which to extract series
        col (int): column of series to extract

    Returns:
        pandas Series corresponding to data object with time values based on
            collection frequency; note that for published data, the data of
            collection is accurate but the start time is arbitrary
    """
    length = data.data_from_offset.shape[0]
    freq_pd_str = as_pd_freq(data.freq)
    index = pd.date_range('2017-01-27', periods=length, freq=freq_pd_str)
    return pd.Series(data.data_from_offset[:, col], index)


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


def offset_from_peak(peak_ind, freq, prepeak=3):
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

