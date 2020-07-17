#!/usr/bin/env python3
"""Class containing optical flow tracking algorithm parameter values.

This module contains the ParamValues class, which contains as instance
variables all parameters used in optical flow tracking of ultrasound images.
All parameters are defaulted to values used commonly across algorithms, but can
be changed either during construction or via the provided setter method for
easy algorithm tuning.
"""


class ParamValues():
    """Class containing all optical flow parameter values.

    Attributes:

        #######################################################################
        ## Lucas-Kanade (LK) tracking parameters
        #######################################################################

            LK_window (int): window size for Lucas-Kanade
            pyr_level (int): level of image pyramiding for Lucas-Kanade

        #######################################################################
        ## Feature-refined Lucas-Kanade (FRLK) tracking parameters
        #######################################################################

            block_size (int): block size used for Sobel derivative kernel in
                Shi-Tomasi corner scoring
            point_frac (float): fraction of top points (based on Shi-Tomasi
                corner score) to keep during FRLK tracking

        #######################################################################
        ## Bilaterally-Filtered Lucas-Kanade (BFLK) tracking parameters
        #######################################################################

            coarse_diam (int): bilateral filter diameter for coarse (i.e., less
                aggressive) filter
            coarse_sigma_color (int): bilateral filter color sigma parameter
            for coarse (i.e., less aggressive) filter
            coarse_sigma_space (int): bilateral filter spatial sigma parameter
                for coarse (i.e., less aggressive) filter
            fine_diam (int): bilateral filter diameter for fine (i.e., more
                aggressive) filter
            fine_sigma_color (int): bilateral filter color sigma parameter for
                fine (i.e., more aggressive) filter
            fine_sigma_space (int): bilateral filter spatial sigma parameter
                for fine (i.e., more aggressive) filter
            percent_fine (float): fraction of points (ordered by Shi-Tomasi
                corner score) to track using fine bilateral filter
            percent_coarse (float): fraction of points (ordered by Shi-Tomasi
                corner score) to track using coarse bilateral filter

        #######################################################################
        ## Supporter-Based Lucas-Kanade (SBLK) tracking parameters
        #######################################################################

            displacement_weight (float): offset (alpha) used in weighting
                function for supporter points
            quality_level (float): quality level of corners chosen via
                Shi-Tomasi corner detection
            min_distance (int): minimum pixel distance between corners chosen
                via Shi-Tomasi corner detection
            max_corners (int): maximum number of good corner points chosen by
                Shi-Tomasi corner detection
            fine_threshold (float): fraction of points to track without using
                supporter points (i.e., to track using pure Lucas-Kanade)
            update_rate (float): update rate for exponential moving average
            num_bottom (int): number of (spatially) bottom-most contour points
                to keep (used to ensure points along the entire contour are
                tracked)
            fix_top (bool): whether to maintain the top set of contour points
                across tracking (mitigates downward drift)
            reset_frequency (int): how often to reset contour points to ground
                truth (used to analyze when and how often tracking drift
                occurs)
    """

    def __init__(self,
                 LK_window=35,
                 pyr_level=3,
                 quality_level=0.4,
                 min_distance=0,
                 max_corners=300,
                 block_size=7,
                 point_frac=0.7,
                 coarse_diam=5,
                 coarse_sigma_color=100,
                 coarse_sigma_space=100,
                 fine_diam=20,
                 fine_sigma_color=80,
                 fine_sigma_space=80,
                 percent_fine=0.2,
                 percent_coarse=0.8,
                 displacement_weight=40,
                 fine_threshold=0.45,
                 update_rate=0.7,
                 num_bottom=0,
                 fix_top=False,
                 reset_frequency=10000):
        """Standard initializer for ParamValues objects.

        Args: see class attributes above

        Returns:
            ParamValues object with default or argument-specified values
        """
        self.LK_window = LK_window
        self.pyr_level = pyr_level
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.max_corners = max_corners
        self.block_size = block_size
        self.point_frac = point_frac
        self.coarse_diam = coarse_diam
        self.coarse_sigma_color = coarse_sigma_color
        self.coarse_sigma_space = coarse_sigma_space
        self.fine_diam = fine_diam
        self.fine_sigma_color = fine_sigma_color
        self.fine_sigma_space = fine_sigma_space
        self.percent_fine = percent_fine
        self.percent_coarse = percent_coarse
        self.displacement_weight = displacement_weight
        self.fine_threshold = fine_threshold
        self.update_rate = update_rate
        self.num_bottom = num_bottom
        self.fix_top = fix_top
        self.reset_frequency = reset_frequency

    ###########################################################################
    ## GETTERS/SETTERS
    ###########################################################################

    def change_values(self, disp_weight, qual_level, min_dist, coarse_d,
                      coarse_sigma_c, coarse_sigma_s, fine_d, fine_sigma_c,
                      fine_sigma_s, window, pyr, fine_thresh, num_bot,
                      perc_fine, perc_coarse, reset_freq):
        """Modify parameter instance variables to given arguments.

        This method is used for parameter tuning and only changes arguments
        that are not None.

        Args: see class attributes above
        """
        if disp_weight is not None:
            self.displacement_weight = disp_weight
        if qual_level is not None:
            self.quality_level = qual_level
        if min_dist is not None:
            self.min_distance = min_dist
        if coarse_d is not None:
            self.coarse_diam = coarse_d
        if coarse_sigma_c is not None:
            self.coarse_sigma_color = coarse_sigma_c
        if coarse_sigma_s is not None:
            self.coarse_sigma_space = coarse_sigma_s
        if fine_d is not None:
            self.fine_diam = fine_d
        if fine_sigma_c is not None:
            self.fine_sigma_color = fine_sigma_c
        if fine_sigma_s is not None:
            self.fine_sigma_space = fine_sigma_s
        if window is not None:
            self.LK_window = window
        if pyr is not None:
            self.pyr_level = pyr
        if fine_thresh is not None:
            self.fine_threshold = fine_thresh
        if num_bot is not None:
            self.num_bottom = num_bot
        if perc_fine is not None:
            self.percent_fine = perc_fine
        if perc_coarse is not None:
            self.percent_coarse = perc_coarse
        if reset_freq is not None:
            self.reset_frequency = reset_freq

    def get_displacement_weight(self):
        """Get SBLK object displacement weight parameter.

        Returns:
            float displacement weight
        """
        return self.displacement_weight
