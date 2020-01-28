import numpy as np
import cv2
import scipy
from scipy import stats


def apply_supporters_model(predicted_target_point, prev_feature_points, feature_points, feature_params, use_tracking, alpha):
    """
    Do model learning or prediction based on learned model, based on conditions of image tracking
    original_feature_points: numpy array of tuples/arrays of x, y coords for features from original image
    curr_image: 2d np array of current grayscale image
    original_image: 2d np array of original grayscale image
    predicted_target_point: numpy array (2-d) of x, y coord of tracking prediction of current target point
    original_target_point: numpy array (2-d) of x, y coord of tracking target
    features: numpy array of 3-tuples of current feature x, y coord (2d np array), displacement average, covariance matrix average
    theta_correlation: threshold for how much current target neighborhood should be similar to original target neighborhood to trust tracking
    theta_prediction: threshold for how much current supporter probability should be to accept tracking prediction
    alpha: learning rate for exponential forgetting principle
    window_size: HALF of window size for obtaining neighborhood of target
    """
    feature_points = format_supporters(feature_points)

    predicted_target_point = np.round(predicted_target_point).astype(int)

    # initialize value to return
    target_point_final = None
    # initialize new target param tuple array
    new_feature_params = []


    # tracking is to be used (first x amount of frames, x determined a priori)
    if use_tracking:
        print("USING TRACKING RESULT, UPDATING SUPPORTERS")

        target_point_final = predicted_target_point

        # update supporter feature parameters
        for i in range(len(feature_points)):
            curr_feature_point = feature_points[i]
            curr_feature_point = np.round(curr_feature_point).astype(int)
            # displacement vector between the current feature and the target point
            curr_displacement = target_point_final - curr_feature_point
            # previous average for displacement
            # print("PARAMS: ", feature_params)
            prev_displacement_average = feature_params[i][0]
            # update displacement average using exponential forgetting principle
            new_displacement_average = alpha * prev_displacement_average + (1 - alpha) * curr_displacement

            # maybe also try outer product:
            displacement_mean_diff = curr_displacement - new_displacement_average
            curr_covariance_matrix = displacement_mean_diff.reshape(2, 1) @ displacement_mean_diff.reshape(1, 2)
            # update covariance matrix using exponential forgetting principle
            prev_covariance_matrix = feature_params[i][1]
            new_covariance_matrix = alpha * prev_covariance_matrix + (1 - alpha) * curr_covariance_matrix

            new_feature_params.append((new_displacement_average, new_covariance_matrix))

    # right now, take a weighted average of the mean displacements + supporter positions, weighted by probability of supporter and prediction
    # also consider taking argmax over multivariate gaussian
    else:
        numerator = 0
        denominator = 0
        displacements = []
        for i in range(len(feature_points)):
            feature_point = feature_points[i]
            prev_feature_point = prev_feature_points[i]
            displacement_norm = np.linalg.norm(feature_point - prev_feature_point)
            displacements.append(displacement_norm)
        displacements = np.array(displacements)
        mean = displacements.mean()
        variance = displacements.var()

        for i in range(len(feature_points)):
            feature_point = feature_points[i]
            prev_feature_point = prev_feature_points[i]
            displacement_norm = np.linalg.norm(feature_point - prev_feature_point)
            weight = weight_function(displacement_norm, mean, variance)
            covariance = feature_params[i][1]
            displacement = feature_params[i][0]
            numerator += (weight * (displacement + feature_point))/np.linalg.det(covariance)
            denominator += weight / np.linalg.det(covariance)

        target_point_final = numerator / denominator
        # x_coords = np.arange(original_target_point[0] - window_size, original_target_point[1] + window_size)
        # y_coords = np.arange(original_target_point[1] - window_size, original_target_point[1] + window_size)
        #
        # X, Y = np.meshgrid(x_coords, y_coords)
        # X = X.flatten()
        # Y = Y.flatten()
        #
        # argmax = -1
        # max_likelihood = -1
        # for i in range(len(X)):
        #     likelihood = point_likelihood(X[i], Y[i], feature_points, feature_params)
        #     if likelihood > max_likelihood:
        #         argmax = i
        #         max_likelihood = likelihood
        #
        # target_point_final = np.array([X[argmax], Y[argmax]])


    # TODO: check this
    if new_feature_params == []:
        return target_point_final, feature_params
    else:
        return target_point_final, new_feature_params



def weight_function(displacement_norm, mean, variance):
    alpha = 5
    # print("DISP NORM: ", displacement_norm)
    # return alpha * displacement_norm + (1 - alpha)
    rv = scipy.stats.multivariate_normal(mean = mean, cov = variance)
    # return rv.pdf(displacement_norm)
    # return displacement_norm
    #return displacement_norm
    # return alpha * displacement_norm + (1 - alpha)
    return 1 + alpha * displacement_norm

    # return 1

def point_likelihood(x, y, feature_points, feature_params):
    point = np.array([x, y])
    probability = 0
    for i in range(len(feature_points)):
        displacement = point - feature_points[i]
        displacement_mean = feature_params[i][0]
        covariance = feature_params[i][1]
        rv = scipy.stats.multivariate_normal(mean=displacement_mean, cov=covariance)
        probability += rv.pdf(displacement)
    return probability



def initialize_supporters(supporter_points, target_point, variance):
    supporters = []
    supporter_params = []
    for i in range(len(supporter_points)):
        supporter_point = supporter_points[i][0]
        supporters.append(supporter_point)
        supporter_params.append((target_point - supporter_point, variance * np.eye(2)))
    return supporters, supporter_params

def format_supporters(supporter_points):
    supporters = []
    for i in range(len(supporter_points)):
        supporters.append(supporter_points[i][0])
    return supporters

def angle(vec1, vec2):
    inter = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.arccos(inter)






