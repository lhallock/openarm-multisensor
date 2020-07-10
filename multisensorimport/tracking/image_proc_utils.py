#!/usr/bin/env python3
"""Utility functions for image processing algorithms.

This module contains functions used within image processing algorithms, such as
filters and edge detectors.
"""
import cv2


def get_filter_from_num(filter_type):
    """Map numbers to corresponding image filters.

    This function maps integer value to corresponding image filter functions,
    which are then used in ultrasound frame processing.

    The mapping is as follows:
        1: median filter
        2: aggressive (fine) bilateral filter
        3: less aggressive (coarse) bilateral filter
        4: anisotropic diffusion filter
        other: no filter

    Args:
        filter_type (int): integer determining which filter to use

    Returns:
        image filter function that takes two arguments (img, run_params)
    """
    filter_out = None
    if filter_type == 1:
        filter_out = median_filter
    elif filter_type == 2:
        filter_out = fine_bilateral_filter
    elif filter_type == 3:
        filter_out = coarse_bilateral_filter
    elif filter_type == 4:
        filter_out = anisotropic_diffuse
    else:
        filter_out = no_filter
    return filter_out


def no_filter(img, run_params):
    """Applies no filter and converts image to grayscale (if color).

    Args:
        img (numpy.ndarray): image to be (potentially) grayscaled
        run_params (ParamValues): values of parameters used in tracking

    Returns:
        numpy.ndarray grayscale, non-filtered version of input image
    """
    # check if image is color or grayscale, return grayscale version
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return img


def median_filter(img, run_params):
    """Applies a median filter to the given image.

    Args:
        img (numpy.ndarray): image to be filtered
        run_params (ParamValues): values of parameters used in tracking

    Returns:
        numpy.ndarray median-filtered version of input image
    """
    kernel_size = 5
    return cv2.medianBlur(img, kernel_size)


def fine_bilateral_filter(img, run_params):
    """Applies an "aggressive" bilateral filter to the given image.

    Args:
        img (numpy.ndarray): image to be filtered
        run_params (ParamValues): values of parameters used in tracking

    Returns:
        numpy.ndarray bilaterally-filtered version of input image
    """
    # convert to color (what bilateral filter expects)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    diam = run_params.fine_diam
    sigma_color = run_params.fine_sigma_color
    sigma_space = run_params.fine_sigma_space
    bilateral_color = cv2.bilateralFilter(img, diam, sigma_color, sigma_space)

    # convert back to grayscale and return
    return cv2.cvtColor(bilateral_color, cv2.COLOR_RGB2GRAY)


def coarse_bilateral_filter(img, run_params):
    """Applies a "less aggressive" bilateral filter to the given image.

    Args:
        img (numpy.ndarray): image to be filtered
        run_params (ParamValues): values of parameters used in tracking

    Returns:
        numpy.ndarray bilaterally-filtered version of input image
    """
    # convert to color (what bilateral filter expects)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    diam = run_params.coarse_diam
    sigma_color = run_params.coarse_sigma_color
    sigma_space = run_params.coarse_sigma_space
    bilateral_color = cv2.bilateralFilter(img, diam, sigma_color, sigma_space)

    # convert back to grayscale and return
    return cv2.cvtColor(bilateral_color, cv2.COLOR_RGB2GRAY)


def anisotropic_diffuse(img, run_params):
    """Applies a Perona-Malik anisotropic diffusion filter to the given image.

    Args:
        img (numpy.ndarray): image to be filtered
        run_params (ParamValues): values of parameters used in tracking

    Returns:
        numpy.ndarray anisotropic-diffused version of input image
    """
    # convert to color
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    alpha_var = 0.1
    k_var = 5
    niters_var = 5
    diffused_color = cv2.ximgproc.anisotropicDiffusion(src=img,
                                                       alpha=alpha_var,
                                                       K=k_var,
                                                       niters=niters_var)

    # convert back to grayscale and return
    return cv2.cvtColor(diffused_color, cv2.COLOR_RGB2GRAY)


def otsu_binarization(gray_image):
    """Applies Otsu binarization to the given image.

    Args:
        gray_image (numpy.ndarray): grayscale image to be binarized

    Returns:
        numpy.ndarray binarized version of input image
    """
    _, th2 = cv2.threshold(gray_image, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def canny(gray_image):
    """Applies Canny edge detection to the given image.

    Args:
        gray_image (numpy.ndarray): grayscale image in which edges should be
            detected

    Returns:
        numpy.ndarray edges present in input image
    """
    edges = cv2.Canny(gray_image, 180, 200)
    return edges
