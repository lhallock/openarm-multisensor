import numpy as np
import scipy
from scipy import stats


def apply_supporters_model(run_params, predicted_target_point, prev_feature_points, feature_points, feature_params, use_tracking, alpha):
    """
    Do model learning or prediction based on learned model, based on conditions of image tracking

    run_params: instance of ParamValues class holding relevent parameters
    predicted_target_point: numpy array (2-element) of x, y coord of tracking prediction of current target point
    prev_feature_points: list of (x,y) coordinates of the feature (supporter) points in previous frame
    feature_points: list of [x,y] coordinate array of the feature (supporter) points in current frame
    feature_params: list of 2-tuples of (displacement vector average, covariance matrix aveage) and  for each feature point
    use_tracking: boolean determining whether to return the pure Lucas Kanade prediction or the supporters based prediction
    alpha: learning rate for exponential forgetting principle

    Returns: predicted location of target point, updated parameters corresponding for the supporter points
    """

    # reformat feature points for easier processing
    feature_points = format_supporters(feature_points)

    # round to integer so that the prediction lands on pixel coordinates
    predicted_target_point = np.round(predicted_target_point).astype(int)

    # initialize value to return
    target_point_final = None
    # initialize new target param tuple array
    new_feature_params = []


    # tracking is to be used (first x amount of frames, x determined a priori)
    if use_tracking:

        target_point_final = predicted_target_point

        # update supporter feature parameters
        for i in range(len(feature_points)):
            curr_feature_point = feature_points[i]
            curr_feature_point = np.round(curr_feature_point).astype(int)
            # displacement vector between the current feature and the target point
            curr_displacement = target_point_final - curr_feature_point
            # previous average for displacement
            prev_displacement_average = feature_params[i][0]
            # update displacement average using exponential forgetting principle
            new_displacement_average = alpha * prev_displacement_average + (1 - alpha) * curr_displacement

            displacement_mean_diff = curr_displacement - new_displacement_average
            # compute current covariance matrix
            curr_covariance_matrix = displacement_mean_diff.reshape(2, 1) @ displacement_mean_diff.reshape(1, 2)
            # update covariance matrix average using exponential forgetting principle
            prev_covariance_matrix = feature_params[i][1]
            new_covariance_matrix = alpha * prev_covariance_matrix + (1 - alpha) * curr_covariance_matrix

            new_feature_params.append((new_displacement_average, new_covariance_matrix))

    # Use supporter prediction: take a weighted average of the mean displacements + supporter positions, weighted by probability of supporter and prediction
    else:
        # quantities used in calculation
        numerator = 0
        denominator = 0
        displacements = []

        for i in range(len(feature_points)):
            feature_point = feature_points[i]
            prev_feature_point = prev_feature_points[i]
            displacement_norm = np.linalg.norm(feature_point - prev_feature_point)
            # determine the weight to assign to that point, as a function of displacement
            weight = weight_function(run_params, displacement_norm)
            covariance = feature_params[i][1]
            displacement = feature_params[i][0]

            numerator += (weight * (displacement + feature_point))/np.linalg.det(covariance)
            denominator += weight / np.linalg.det(covariance)

        # return weighted average
        target_point_final = numerator / denominator


    # if Supporters was used, return the old feature_params; else return the updated params
    if new_feature_params == []:
        return target_point_final, feature_params
    else:
        return target_point_final, new_feature_params



def weight_function(run_params, displacement_norm):
    """
    Determines the weight to apply to each supporter point, as a function of the norm of the displacement vector for that point.

    run_params: instance of ParamValues class holding relevent parameters
    displacement_norm: L2 norm of the displacement vector of the supporter point being considered

    Returns: weight to place for the supporter point being considered

    """
    alpha = run_params.displacement_weight

    return 1 + (alpha * displacement_norm)


def initialize_supporters(supporter_points, target_point, variance):
    """
    Reformats list of given supporter points, and initializes parameters (displacement, covariance) for each supporter point, for a given target point

    supporter_points: numpy array of 1 element numpy arrays, where the 1 element is a 2-element numpy array containing supporter point locations
    target_point: numpy array containing x,y coordinates for the target point being tracked
    variance: scalar value, indicates the initial variance for each element of the displacement

    Returns: list of 2-element numpy arrays containing supporter point locations
    """

    # initialize empty lists
    supporters = []
    supporter_params = []
    for i in range(len(supporter_points)):
        # extract numpy array of the supporter location
        supporter_point = supporter_points[i][0]
        supporters.append(supporter_point)
        # initialize displacement average with initial displacement and a diagonal covariance matrix
        supporter_params.append((target_point - supporter_point, variance * np.eye(2)))

    return supporters, supporter_params

def format_supporters(supporter_points):
    """
    Reformats list of given supporter points into a list of numpy arrays containing the supporter point locations

    supporter_points: numpy array of 1 element numpy arrays, where the 1 element is a 2-element numpy array containing supporter point locations

    Returns: list of 2-element numpy arrays containing supporter point locations
    """
    supporters = []
    for i in range(len(supporter_points)):
        supporters.append(supporter_points[i][0])
    return supporters
