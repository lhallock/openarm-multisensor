"""
Methods implementing image processing algorithms such as image filters and edge detectors. 
"""

import cv2
import numpy as np
import scipy


def get_filter_from_num(filter_type):
    """
    Maps numbers to corresponding image filters. Mapping is: 1 -> median filter, 2->aggressive (fine) bilateral filter, 3 -> less agressive (course) bilateral filter, 4 -> anisotropic diffusion filter, anything else -> no filter

    Args:
        filter_type: integer determining which filter to use
    Returns:
        image filter function. This function takes two arguments (img, run_params)
    """

    filter = None
    if filter_type == 1:
        filter = median_filter
    elif filter_type == 2:
        filter = fine_bilateral_filter
    elif filter_type == 3:
        filter = course_bilateral_filter
    elif filter_type == 4:
        filter = anisotropic_diffuse
    else:
        filter = no_filter
    return filter


# image filtering
def no_filter(img, run_params):
    """
    Applies no filter to the image. Convert to grayscale if the image is color.

    Args:
        img: image to be potentially grayscaled
        run_params: instance of ParamValues class, contains values of parameters used in tracking

    Returns:
        Grayscaled, non-filtered image
    """

    # check if image is color or grayscale, return grayscale version
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return img


def median_filter(img, run_params):
    """
    Applies a median filter to the given image.

    Args:
        img: image to be filtered
        run_params: instance of ParamValues class, contains values of parameters used in tracking

    Returns: median filtered version of the img
    """

    kernelSize = 5
    return cv2.medianBlur(img, kernelSize)


def fine_bilateral_filter(img, run_params):
    """
    Applies an "aggressive" bilateral filter to the given image.

    Args:
        img: image to be filtered
        run_params: instance of ParamValues class, contains values of parameters used in tracking

    Returns: bilateral filtered version of the img
    """

    # convert to color (what bilateral filter expects)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    diam = run_params.fine_diam
    sigmaColor = run_params.fine_sigma_color
    sigmaSpace = run_params.fine_sigma_space
    bilateralColor = cv2.bilateralFilter(img, diam, sigmaColor, sigmaSpace)

    # convert back to grayscale and return
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)


def course_bilateral_filter(img, run_params):
    """
    Applies a "less aggressive" filter to the given image.

    Args:
        img: image to be filtered
        run_params: instance of ParamValues class, contains values of parameters used in tracking

    Returns: bilateral filtered version of the img
    """

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # hyperparameters
    diam = run_params.course_diam
    sigmaColor = run_params.course_sigma_color
    sigmaSpace = run_params.course_sigma_space
    bilateralColor = cv2.bilateralFilter(img, diam, sigmaColor, sigmaSpace)
    return cv2.cvtColor(bilateralColor, cv2.COLOR_RGB2GRAY)


def anisotropic_diffuse(img, run_params):
    """
    Applies a Perona-Malik anisotropic diffusion filter to the given image.

    Args:
        img: image to be filtered
        run_params: instance of ParamValues class, contains values of parameters used in tracking

    Returns: anisotropic diffused version of the img
    """

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # hyperparameters
    alphaVar = 0.1
    KVar = 5
    nitersVar = 5
    diffusedColor = cv2.ximgproc.anisotropicDiffusion(src=img,
                                                      alpha=alphaVar,
                                                      K=KVar,
                                                      niters=nitersVar)
    return cv2.cvtColor(diffusedColor, cv2.COLOR_RGB2GRAY)


def otsu_binarization(gray_image):
    """
    Applies otsu binarization to the given image.

    Args:
        gray_img: grayscale image to be binarized

    Returns: Binarized version of the img
    """

    ret2, th2 = cv2.threshold(gray_image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def canny(gray_image):
    """
    Applies Canny Edge Detection to the given image.

    Args:
        gray_img: grayscale image in which edges should be fine

    Returns: Edges present in given image
    """

    edges = cv2.Canny(gray_image, 180, 200)
    return edges
