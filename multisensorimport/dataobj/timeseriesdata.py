#!/usr/bin/env python3
"""Class containing time series data and associated metadata.

This module contains the TimeSeriesData class, which contains time series data
and associated metadata (resolution, frequency, channel labels, etc.). The
class is generic and designed for use with electromyography (EMG), acoustic
myography (AMG), and force data, as well as quantitative features extracted
from ultrasound (e.g., muscle cross-sectional area, or CSA). In practice, this
class can be used to describe any time-series data, biometric or non-biometric.

"""

import numpy as np
from scipy.io import wavfile


class TimeSeriesData():
    """Class containing time series data and associated metadata.

    Attributes:
        data (numpy.ndarray): n x n_ch array of data values
        data_from_offset (numpy.ndarray): n' x n_ch array of data values, where
            n' is the number of data values after the specified offset
        n_ch (int): number of data channels
        n (int): number of data points
        freq (int): sampling frequency (in Hz)
        offset (int): 'zero-valued' time point (for alignment with other
            TimeSeriesData objects), in terms of number of samples
        label (str): human-readable identifier for data series (e.g., 'EMG')
        ch_labels (list): length-n_ch ordered list of data channel labels

    """

    def __init__(self, label):
        """Standard initializer for TimeSeriesData objects.

        Args:
            label (str): desired label for data series (e.g., 'EMG')

        Returns:
            (empty) TimeSeriesData object
        """
        self._label = label
        self._data = None
        self._data_from_offset = None
        self._n_ch = None
        self._n = None
        self._freq = None
        self._offset = None
        self._ch_labels = None

    @classmethod
    def from_array(cls, label, data, ch_labels, freq, offset=0):
        """Initialize TimeSeriesData object from array and metadata.

        Example:
            obj = TimeSeriesData.from_array(label, data, ch_labels, freq)

        Args:
            label (str): desired label for data series (e.g., 'EMG')
            data (numpy.ndarray): n x n_ch array of desired data values
            ch_labels (list): length-n_ch ordered list of desired data channel
                labels
            freq (int): sampling frequency (in Hz)
            offset (int): 'zero-valued' time point (for alignment with other
                TimeSeriesData objects), in terms of number of samples

        Returns:
            TimeSeriesData object containing specified values
        """
        tsd = cls(label)
        tsd._data = data
        tsd._n_ch = tsd._data.shape[1]
        tsd._n = tsd._data.shape[0]
        tsd._freq = freq
        tsd._offset = offset
        tsd._ch_labels = ch_labels
        tsd._data_from_offset = tsd._compute_offset_data()
        tsd._assert_consistent()

        return tsd

    @classmethod
    def from_file(cls,
                  label,
                  filename,
                  ch_labels,
                  freq=None,
                  offset=0,
                  filetype='csv',
                  header_lines=0,
                  cols=-1):
        """Initialize TimeSeriesData object from file.

        Example:
            obj = TimeSeriesData.from_file(label, filename, freq, ch_labels)

        Args:
            label (str): desired label for data series (e.g., 'EMG')
            filename (str): path to data series file
            filetype (str): file format, either 'csv' or 'wav'
            header_lines (int): number of initial rows of CSV to skip when
                importing
            cols (sequence): columns of CSV to include during import (if -1,
                all columns will be imported)

        Returns:
            TimeSeriesData object containing data from file
        """
        tsd = cls(label)
        tsd._offset = offset
        tsd._ch_labels = ch_labels

        if filetype == 'csv':
            tsd._init_from_csv(filename, header_lines, cols)
            tsd._freq = freq
        elif filetype == 'wav':
            tsd._init_from_wav(filename)
            # error out if specified frequency is inconsistent with frequency
            # extracted from WAV file
            if freq and (freq != tsd._freq):
                raise ValueError('Specified frequency is inconsistent with',
                                 'internal WAV file frequency.')
        else:
            raise ValueError('Unable to instantiate TimeSeriesData object:',
                             'unsupported file type.')

        tsd._assert_consistent()

        return tsd

    def _init_from_csv(self, filename, header_lines, cols):
        """Internal helper method for instantiation from CSV.

        Args:
            filename (str): path to data series file
            header_lines (int): number of initial rows of CSV to skip when
                importing
            cols (sequence): columns of CSV to include during import (if -1,
                all columns will be imported)
        """
        if cols == -1:
            self._data = np.genfromtxt(filename,
                                       delimiter=',',
                                       skip_header=header_lines)
        else:
            self._data = np.genfromtxt(filename,
                                       delimiter=',',
                                       skip_header=header_lines,
                                       usecols=cols)

        if len(self._data.shape) == 1:  # handle single-channel data
            self._n_ch = 1
            self._n = self._data.shape[0]
            self._data = np.reshape(self._data, (-1, 1))
        else:
            self._n_ch = self._data.shape[1]
            self._n = self._data.shape[0]

        self._data_from_offset = self._compute_offset_data()

    def _init_from_wav(self, filename):
        """Internal helper method for instantiation from WAV.

        Args:
            filename (str): path to desired data series file

        Todo:
            check if data type is consistent w/ CSV read
        """
        self._freq, self._data = wavfile.read(filename)
        self._n_ch = self._data.shape[1]
        self._n = self._data.shape[0]

        self._data_from_offset = self._compute_offset_data()

    def _assert_consistent(self):
        """Confirm that dimensions of data and channel labels are consistent.

        This function is called on object initialization, as well as whenever
        self.ch_labels is accessed or modified externally. Note that it is NOT
        called when self.data is accessed -- data can exist without labels, but
        labels can't exist without data!

        Raises:
            ValueError if self.ch_labels is inconsistent with self.data
        """
        if self.ch_labels and (len(self.ch_labels) != self.n_ch):
            raise ValueError('Channel labels are of inconsistent dimension.')

    def _compute_offset_data(self):
        """Compute data matrix from specified offset.

        Note that if the specified index is before data collection begins, the
        array will pre-populate with zeros. (TODO: make these NaNs probably)

        Returns:
            numpy.ndarray matrix starting from specified offset index
        """
        # if offset is positive, just chop off the start of the matrix
        if self.offset >= 0:
            return self.data[self.offset:, :]

        # if offset is negative, prepend zeros
        else:
            prefix = np.zeros((-1 * self.offset, self.data.shape[1]))
            return np.vstack((prefix, self.data))

    ###########################################################################
    ## GETTERS AND SETTERS
    ###########################################################################

    @property
    def data(self):
        """Get object data.

        Returns:
            numpy.ndarray data object
        """
        return self._data

    @data.setter
    def data(self, data):
        """Set object data.

        Args:
            numpy.ndarray n_ch x n data array
        """
        self._data = data
        self._n_ch = self._data.shape[0]
        self._n = self._data.shape[1]

    @property
    def data_from_offset(self):
        """Get object data starting from specified offset.

        Returns:
            numpy.ndarray data object. computed from offset
        """
        return self._data_from_offset

    @property
    def n_ch(self):
        """Get number of data channels.

        Returns:
            int number of data channels
        """
        return self._n_ch

    @property
    def n(self):
        """Get number of data points in time series.

        Returns:
            int number of data points
        """
        return self._n

    @property
    def freq(self):
        """Get frequency of data collection (in Hz).

        Returns:
            int frequency in Hz
        """
        return self._freq

    @freq.setter
    def freq(self, freq):
        """Set data collection rate (in Hz).

        Args:
            freq (int): data collection frequency
        """
        self._freq = freq

    @property
    def offset(self):
        """Get 'zero-valued' time point.

        This method returns the point of the data series considered 'zero' for
        the purposes of alignment with other TimeSeriesData objects).

        Returns:
            int offset time point (in number of data points)
        """
        return self._offset

    @offset.setter
    def offset(self, offset):
        """Set 'zero-valued' time point.

        This method sets the point of the data series considered 'zero' for the
        purposes of alignment with other TimeSeriesData objects).

        Args:
            offset (int): desired offset (in number of data points)
        """
        self._offset = offset

    @property
    def label(self):
        """Get human-readable data series identifier.

        Returns:
            str data series identifier
        """
        return self._label

    @label.setter
    def label(self, label):
        """Set human-readable data series identifier.

        Args:
            label (str): desired data series identifier
        """
        self._label = label

    @property
    def ch_labels(self):
        """Get ordered list of data channel labels.

        Returns:
            list of length-n_ch data channel labels
        """
        return self._ch_labels

    @ch_labels.setter
    def ch_labels(self, ch_labels):
        """Set data channel labels.

        Args:
            ch_labels (sequence): length-n_ch sequence of data channel labels

        Raises:
            ValueError if channel label dimension is inconsistent with
                self.data
        """
        self._ch_labels = ch_labels
        self._assert_consistent()
