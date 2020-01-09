import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline


def update_warp_params(curr_image_interp, template_image_interp, delIx_interp, delIy_interp, curr_image_shape, weights, X, Y, warp_params, eps, max_iters, debug = False):

    # initial guess for warp params
    updating_warp_params = warp_params.copy()

    num_params = len(updating_warp_params)

    iter = 0

    # steepest descent using Gauss-Newton method
    while (iter < max_iters):
        if debug:
            print('WARP PARAMS: ', updating_warp_params)
        # unpack warp parameters
        p1 = updating_warp_params[0]
        p2 = updating_warp_params[1]
        p3 = updating_warp_params[2]
        p4 = updating_warp_params[3]
        p5 = updating_warp_params[4]
        p6 = updating_warp_params[5]

        X_w = (1 + p1) * X + p3 * Y + p5
        if debug:
            print('MIN X: ', np.min(X_w))
            print('MAX X: ', np.max(X_w))

        Y_w = p2 * X + (1 + p4) * Y + p6

        if debug:
            print('MIN Y: ', np.min(Y_w))
            print('MAX Y: ', np.max(Y_w))

        valid_pos = (X_w >= 0) & (X_w < curr_image_shape[1]) & (Y_w >= 0) & (Y_w < curr_image_shape[0])

        X_filtered = X[valid_pos]
        X_w = X_w[valid_pos]

        Y_filtered = Y[valid_pos]
        Y_w = Y_w[valid_pos]

        weights_filtered = weights[valid_pos]

        template_image_values = template_image_interp.ev(Y_filtered, X_filtered)
        warped_image_values = curr_image_interp.ev(Y_w, X_w)
        error_image = template_image_values - warped_image_values

        delIx_warped = curr_image_interp.ev(Y_w, X_w, dx = 0, dy = 1)
        delIy_warped = curr_image_interp.ev(Y_w, X_w, dx = 1, dy = 0)

        num_points = X_filtered.shape[0]
        hessian = np.zeros((num_params, num_params))
        steepest_descent_image = np.zeros(num_params)

        # print('iter: ', iter, ': ', num_points)

        for i in range(num_points):
            x = X_filtered[i]
            y = Y_filtered[i]
            weight = weights_filtered[i]
            image_diff = error_image[i]

            del_x = delIx_warped[i]
            del_y = delIy_warped[i]
            vec = np.array([x * del_x, x * del_y, y * del_x, y * del_y, del_x, del_y])
            if debug:
                print('VEC: ', vec)
                print('DELX: ', del_x, ' DELY: ', del_y)
            steepest_descent_image += weight * vec * image_diff
            hessian += weight * np.dot(vec.reshape(num_params, 1), vec.reshape(1, num_params))
        # print('Iter: ', iter, 'Steepest image norm: ', np.linalg.norm(steepest_descent_image))
        # print('Iter: ', iter, 'Hessian norm: ', np.linalg.norm(hessian))

        delta_p = np.dot(np.linalg.pinv(hessian), steepest_descent_image)
        if debug:
            print('Iter: ', iter)
            print('Num points: ', num_points)
            print('Delta p norm: ', np.linalg.norm(delta_p))
            print('Hessian norm: ', np.linalg.norm(hessian))
            print('Steepest descent norm: ', np.linalg.norm(steepest_descent_image))
            print('weights: ', weights)
        # take a descent step
        updating_warp_params += delta_p
        # print('Iter: ', iter, 'Delta p norm: ', np.linalg.norm(delta_p))
        if np.linalg.norm(delta_p) <= eps:
            break

        iter += 1

    return updating_warp_params


def robust_drift_corrected_tracking(curr_image, first_template, curr_template, curr_errors, full_warp_params, one_step_warp_params, point1, point2, eps, max_iters, debug = False):
    # spline interpolation of image, templates, errors for potential indexing into non-integer coordinates
    spline_inter_curr_image = RectBivariateSpline(np.arange(curr_image.shape[0]), np.arange(curr_image.shape[1]), curr_image)
    spline_inter_curr_template = RectBivariateSpline(np.arange(curr_template.shape[0]), np.arange(curr_template.shape[1]), curr_template)
    spline_inter_first_template = RectBivariateSpline(np.arange(first_template.shape[0]), np.arange(first_template.shape[1]), first_template)
    spline_inter_errors = RectBivariateSpline(np.arange(curr_errors.shape[0]), np.arange(curr_errors.shape[1]), curr_errors)

    # gradient computations
    delIx = imageDerivativeX(curr_image)
    delIy = imageDerivativeY(curr_image)

    # spline interpolation of gradients
    spline_inter_delIx = RectBivariateSpline(np.arange(delIx.shape[0]), np.arange(delIx.shape[1]), delIx)
    spline_inter_delIy = RectBivariateSpline(np.arange(delIy.shape[0]), np.arange(delIy.shape[1]), delIy)

    # print('One step: ', one_step_warp_params)
    # print('Full: ', full_warp_params)

    # unpack full warp parameters p*(0 -> n-1)
    p1 = full_warp_params[0]
    p2 = full_warp_params[1]
    p3 = full_warp_params[2]
    p4 = full_warp_params[3]
    p5 = full_warp_params[4]
    p6 = full_warp_params[5]

    # coordinates in first_template image (box)
    x_coords = np.arange(point1[0], point2[0] + 1, 1)
    y_coords = np.arange(point1[1], point2[1] + 1, 1)

    X, Y = np.meshgrid(x_coords, y_coords)

    # warp into curr template coordinates
    X_w = (1 + p1) * X + p3 * Y + p5
    Y_w = p2 * X + (1 + p4) * Y + p6


    # filter out the points which are out of bounds
    valid_pos = (X_w >= 0) & (X_w < curr_template.shape[1]) & (Y_w >= 0) & (Y_w < curr_template.shape[0])
    X_w = X_w[valid_pos].flatten()
    Y_w = Y_w[valid_pos].flatten()
    X_filtered = X[valid_pos].flatten()
    Y_filtered = Y[valid_pos].flatten()

    # get the errors at X_filtered, Y_filtered (errors are in the first_template coordinate frame)
    errors_filtered = spline_inter_errors.ev(Y_filtered, X_filtered)
    errors_filtered_median = np.median(errors_filtered.flatten())
    # get the weights from errors, using scheme described in Schreiber's Paper
    weights_filtered = (errors_filtered <= errors_filtered_median * 1.4826).astype(int)
    # print('Weights filtered ', weights_filtered)
    # update the one step parameters to obtain an estimate of warp p(n-1 -> n), using p*(n-2 -> n-1) as initial guess
    print('Obtaining one step estimate')
    one_step_warp_estimate = update_warp_params(spline_inter_curr_image, spline_inter_curr_template, spline_inter_delIx, spline_inter_delIy, curr_image.shape, weights_filtered, X_w, Y_w, one_step_warp_params, eps, max_iters, debug = debug)

    # combine p*(0 -> n-1) and p(n-1 -> n) to get initial guess for p(0 -> n)
    full_warp_guess = one_step_warp_estimate + full_warp_params
    print('FULL WARP GUESS: ', full_warp_guess)

    # get errors at x_coords, y_coords
    errors = spline_inter_errors.ev(Y, X)
    errors_median = np.median(errors.flatten())
    # get weights from errors
    weights = (errors <= errors_median * 1.4826).astype(int)

    # update full warp params by tracking first_template in curr_image with p(0 -> n) as guess, obtainin p*(0 -> n)
    print('Getting full optimal')
    full_warp_optimal = update_warp_params(spline_inter_curr_image, spline_inter_first_template, spline_inter_delIx, spline_inter_delIy, curr_image.shape, weights, X, Y, full_warp_guess, eps, max_iters, debug = debug)

    # obtain optimal one step warp, p*(n-1 -> n), by combining optimal full warp with previous optimal full warp
    one_step_warp_optimal = full_warp_optimal - full_warp_params

    print('Updating errors')
    # error updating
    # learning rate:
    alpha = 0.1

    # unpack full warp optimal parameters
    p1 = full_warp_optimal[0]
    p2 = full_warp_optimal[1]
    p3 = full_warp_optimal[2]
    p4 = full_warp_optimal[3]
    p5 = full_warp_optimal[4]
    p6 = full_warp_optimal[5]

    # warp coords
    X_w = (1 + p1) * X + p3 * X + p5
    Y_w = p2 * X + (1 + p4) * Y + p6

    # remove out of bounds coordinates
    valid_pos = (X_w >= 0) & (X_w < curr_image.shape[1]) & (Y_w >= 0) & (Y_w < curr_image.shape[0])
    X_w = X_w[valid_pos].flatten()
    Y_w = Y_w[valid_pos].flatten()

    X_filtered = X[valid_pos].flatten()
    Y_filtered = Y[valid_pos].flatten()


    for i in range(len(X_filtered)):
        x = X_filtered[i]
        x_w = X_w[i]
        for j in range(len(Y_filtered)):
            y = Y_filtered[j]
            y_w = Y_w[j]
            temp_err = np.abs(spline_inter_curr_image.ev(np.array([y_w]), np.array([x_w]))[0] - spline_inter_first_template.ev(np.array([y]), np.array([x]))[0])
            curr_errors[y][x] = (1 - alpha) * curr_errors[y][x] + alpha * temp_err


    # return full_warp_optimal, one_step_warp_optimal, errors
    return full_warp_optimal, one_step_warp_optimal, curr_errors


def imageDerivativeX(img):
    sobel_x64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    return sobel_x64f


def imageDerivativeY(img):
    sobel_y64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return sobel_y64f

# helper for testing
def affineWarp(point, warpParams):
    """ Applies an affine warp, parameterized by p, to coordinates in x (np array)

    Args:
        x: 2-element numpy array with x, y coordinates of image coordinate
        p: 6-element numpy array of parameters for the affine warp

    Returns:
        2-element numpy array of transformed coordinates
    """

    # unpack warp parameters
    p1 = warpParams[0]
    p2 = warpParams[1]
    p3 = warpParams[2]
    p4 = warpParams[3]
    p5 = warpParams[4]
    p6 = warpParams[5]

    x = point[0]
    y = point[1]

    elemOne = (1+p1)*x + p3 * y + p5
    elemTwo = p2 * x + (1+p4) * y + p6
    return np.array([elemOne, elemTwo])
