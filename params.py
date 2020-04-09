import numpy as np
import run_ultrasoundviz as run



class ParamValues():
    """
    Class containing instance variables which are parameters used in various tracking algorithms and image filtering techniques, and a method to modify these variables. Used to easily change parameters for tuning.
    """
    # offset (alpha) used in the weighting function for supporters points
    displacement_weight = 40

    # Quality level of corners chosen via Shi-Tomasi corner detection
    quality_level = 0.4
    # Minimum distance between corners chosen via Shi-Tomasi corner detection
    min_distance = 0

    #
    course_diam = 5
    course_sigma_color = 100
    course_sigma_space = 100
    fine_diam = 20
    fine_sigma_color = 80
    fine_sigma_space = 80

    LK_window = 35
    pyr_level = 3

    fine_threshold = 0.45
    num_bottom = 0

    percent_fine = 0.2
    percent_course =0.8
    fix_top = False

    reset_frequency = 100000


    def change_values(self, disp_weight, qual_level, min_dist, course_d, course_sigma_c, course_sigma_s, fine_d,
                      fine_sigma_c, fine_sigma_s, window, pyr, fine_thresh, num_bot, perc_fine, perc_course, reset_freq):
        """
        Method to modify the parameter instance variables to the given arguments. Only changes the arguments which are not None.
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
        return self.displacement_weight


global parameter_values
parameter_values = ParamValues()

def write_run():
    """
    Execute a run of tracking
    Run_type mappings:
        1: LK
        2: FRLK
        3: BFLK
        4: SBLK
    """

    run_type = 2
    run.run(parameter_values, run_type)


if __name__ == "__main__":
    write_run()
