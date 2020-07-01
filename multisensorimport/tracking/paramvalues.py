"""Class containing tracking algorithm parameter values."""


class ParamValues():
    """
    This class contains all parameters used in optical flow tracking of
    ultrasound images as instance variables, as well as a method to modify
    these variables. It is used to easily modify parameters for tuning.
    Parameter values are defaulted to values used commonly across algorithms,
    but can be changed when constructing, or via the setter method.
    """

    def __init__(self,
                 LK_window=35,
                 pyr_level=3,
                 quality_level=0.4,
                 min_distance=0,
                 max_corners=300,
                 block_size=7,
                 point_frac=0.7,
                 course_diam=5,
                 course_sigma_color=100,
                 course_sigma_space=100,
                 fine_diam=20,
                 fine_sigma_color=80,
                 fine_sigma_space=80,
                 percent_fine=0.2,
                 percent_course=0.8,
                 displacement_weight=40,
                 fine_threshold=0.45,
                 update_rate=0.7,
                 num_bottom=0,
                 fix_top=False,
                 reset_frequency=10000):
        """
        Args:
            ###########################################################################
            ## LK PARAMETERS
            ###########################################################################
            LK_window: window size for Lucas Kanade
            pyr_level: level of image pyramiding for Lucas Kanade

            ###########################################################################
            ## FRLK PARAMETERS
            ###########################################################################
            quality_level: Quality Level of Corners chosen via Shi-Tomasi corner detection
            min_distance: minimum pixel distance between corners chosen via Shi-Tomasi corner detection
            max_corners: maximum number of good corner points chosen by shi-Tomasi corner detection
            block_size: block size used for Sobel derivative kernel in Shi-Tomasi corner scoring
            point_frac: fraction of top points (based on corner score) to keep in FRLK

            ###########################################################################
            ## BFLK PARAMETERS
            ###########################################################################
            course_diam: bilateral filter diameter for the course/less aggressive filter
            course_sigma_color: bilateral filter color sigma parameter for the course/less aggressive filter
            course_sigma_space: bilateral filter spacial sigma parameter for the course/less aggressive filter
            fine_diam: bilateral filter diameter for the fine/more aggressive filter
            fine_sigma_color: bilateral filter color sigma parameter for the fine/more aggressive filter
            fine_sigma_space: bilateral filter spacial sigma parameter for the fine/more aggressive filter
            percent_fine: fraction of points (ordered by corner score) to track using fine bilateral filter
            percent_course: fraction of points (ordered by corner score) to track using course bilateral filter

            ###########################################################################
            ## SBLK PARAMETERS
            ###########################################################################
            displacement_weight: offset (alpha) used in weighting function for supporter points
            fine_threshold: fraction of points to tracking without using supporters (i.e. track using pure Lucas Kanade)
            update_rate: update rate for exponential moving average
            num_bottom: number of bottom-most contour points to keep (used to ensure points along the entire contour are track)
            fix_top: boolean indicating whether to maintain the top set of contour points across tracking (mitigates downward drift)
            reset_frequency: how often to reset contour points to ground truth (used to analyze when and how often tracking drift occurs)
        """

        self.LK_window = LK_window
        self.pyr_level = pyr_level
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.max_corners = max_corners
        self.block_size = block_size
        self.point_frac = point_frac
        self.course_diam = course_diam
        self.course_sigma_color = course_sigma_color
        self.course_sigma_space = course_sigma_space
        self.fine_diam = fine_diam
        self.fine_sigma_color = fine_sigma_color
        self.fine_sigma_space = fine_sigma_space
        self.percent_fine = percent_fine
        self.percent_course = percent_course
        self.displacement_weight = displacement_weight
        self.fine_threshold = fine_threshold
        self.update_rate = update_rate
        self.num_bottom = num_bottom
        self.fix_top = fix_top
        self.reset_frequency = reset_frequency

    ###########################################################################
    ## GETTERS/SETTERS
    ###########################################################################

    def change_values(self, disp_weight, qual_level, min_dist, course_d,
                      course_sigma_c, course_sigma_s, fine_d, fine_sigma_c,
                      fine_sigma_s, window, pyr, fine_thresh, num_bot,
                      perc_fine, perc_course, reset_freq):
        """Modify parameter instance variables to given arguments.

        This method is used for parameter tuning, and only changes arguments
        that are not None.
        """
        if disp_weight is not None:
            self.displacement_weight = disp_weight
        if qual_level is not None:
            self.quality_level = qual_level
        if min_dist is not None:
            self.min_distance = min_dist
        if course_d is not None:
            self.course_diam = course_d
        if course_sigma_c is not None:
            self.course_sigma_color = course_sigma_c
        if course_sigma_s is not None:
            self.course_sigma_space = course_sigma_s
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
        if perc_course is not None:
            self.percent_course = perc_course
        if reset_freq is not None:
            self.reset_frequency = reset_freq

    def get_displacement_weight(self):
        """Get SBLK object displacement weight parameter.

        Returns:
            float displacement weight
        """
        return self.displacement_weight
