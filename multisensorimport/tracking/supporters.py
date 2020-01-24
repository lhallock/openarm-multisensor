import numpy as np
import cv2
import scipy
from scipy import stats


def apply_supporters_model(original_feature_points, curr_image, original_image, predicted_target_point, original_target_point, feature_points, feature_params, theta_correlation, theta_prediction, alpha, window_size):
    """
    Do model learning or prediction, based on conditions of image tracking
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

    predicted_target_point = np.round(predicted_target_point).astype(int)
    original_target_point = np.round(original_target_point).astype(int)

    target_start_y_curr = max(0, predicted_target_point[1] - window_size)
    target_y_start_diff = abs(target_start_y_curr - predicted_target_point[1])
    target_end_y_curr = min(curr_image.shape[0], predicted_target_point[1] + window_size)
    target_y_end_diff = abs(target_end_y_curr - predicted_target_point[1])

    target_start_x_curr = max(0, predicted_target_point[0] - window_size)
    target_x_start_diff = abs(target_start_x_curr - predicted_target_point[0])
    target_end_x_curr = min(curr_image.shape[1], predicted_target_point[0] + window_size)
    target_x_end_diff = abs(target_end_x_curr - predicted_target_point[0])

    target_start_y_initial = original_target_point[1] - target_y_start_diff
    target_end_y_initial = original_target_point[1] + target_y_end_diff

    target_start_x_initial = original_target_point[0] - target_x_start_diff
    target_end_x_initial = original_target_point[0] + target_x_end_diff





    # get the neighborhood image around initial target, and current target according to tracking
    target_neighorhood_initial = original_image[target_start_y_initial : target_end_y_initial, target_start_x_initial:target_end_x_initial]
    target_neighorhood_current = curr_image[target_start_y_curr : target_end_y_curr, target_start_x_curr:target_end_x_curr]




    # obtain normalized similarity coefficient (between 0 and 1)
    try:
        target_correlation_coefficient = cv2.matchTemplate(target_neighorhood_current, target_neighorhood_initial, cv2.TM_CCOEFF_NORMED)[0][0]
    except Exception as e:
        print("EXCEPTION IN TARGET: ", repr(e))
        target_correlation_coefficient = 0
    # initialize value to return
    target_point_final = None
    # initialize new target param tuple array
    new_feature_params = []


    # if the similarity is above a defined threshold
    if target_correlation_coefficient > theta_correlation:
        print("USING TRACKING RESULT, UPDATING SUPPORTERS")

        # can trust tracking, so set the final tracking position to the predicted point
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

            # maybe also try outer product: curr_covariance_matrix
            displacement_mean_diff = curr_displacement - new_displacement_average
            curr_covariance_matrix = np.array([[displacement_mean_diff[0]**2, 0], [0, displacement_mean_diff[1]**2]])
            # update covariance matrix using exponential forgetting principle
            prev_covariance_matrix = feature_params[i][1]
            new_covariance_matrix = alpha * prev_covariance_matrix + (1 - alpha) * curr_covariance_matrix

            new_feature_params.append((new_displacement_average, new_covariance_matrix))

    # right now, take a weighted average of the mean displacements + supporter positions, weighted by probability of supporter and prediction
    # also consider taking argmax over multivariate gaussian
    else:
        # estimate probability of target point from supporters
        numerator = 0
        denominator = 0
        predicted_probability = 0
        for i in range(len(feature_points)):
            original_supporter_point = original_feature_points[i]
            original_supporter_point_rounded = np.round(original_supporter_point).astype(int)

            curr_supporter_point = feature_points[i]
            curr_supporter_point_rounded = np.round(curr_supporter_point).astype(int)

            # start_y_curr = max(0, curr_supporter_point_rounded[1] - window_size)
            # y_start_diff = abs(start_y_curr - curr_supporter_point_rounded[1])
            # end_y_curr = min(curr_image.shape[0], curr_supporter_point_rounded[1] + window_size)
            # y_end_diff = abs(end_y_curr - curr_supporter_point_rounded[1])
            #
            # start_x_curr = max(0, curr_supporter_point_rounded[0] - window_size)
            # x_start_diff = abs(start_x_curr - curr_supporter_point_rounded[0])
            # end_x_curr = min(curr_image.shape[1], curr_supporter_point_rounded[0] + window_size)
            # x_end_diff = abs(end_x_curr - curr_supporter_point_rounded[0])




            start_y_curr = original_supporter_point_rounded[1] - window_size
            end_y_curr = original_supporter_point_rounded[1] + window_size

            start_x_curr = original_supporter_point_rounded[0] - window_size
            end_x_curr = original_supporter_point_rounded[0] + window_size

            # print("X END DIFF: ", x_end_diff)



            # start_y_initial = original_supporter_point_rounded[1] - y_start_diff
            # end_y_initial = original_supporter_point_rounded[1] + y_end_diff
            #
            # start_x_initial = original_supporter_point_rounded[0] - x_start_diff
            # end_x_initial = original_supporter_point_rounded[0] + x_end_diff


            start_y_initial = original_supporter_point_rounded[1] - window_size
            end_y_initial = original_supporter_point_rounded[1] + window_size

            start_x_initial = original_supporter_point_rounded[0] - window_size
            end_x_initial = original_supporter_point_rounded[0] + window_size

            supporter_neighorhood_initial = original_image[start_y_initial : end_y_initial, start_x_initial : end_x_initial]



            supporter_neighborhood_current = curr_image[start_y_curr : end_y_curr, start_x_curr : end_x_curr]

            # print("INITIAL SHAPE: ", supporter_neighorhood_initial.shape, " CURRENT SHAPE: ", supporter_neighborhood_current.shape)
            try:
                supporter_correlation_coefficient = cv2.matchTemplate(supporter_neighborhood_current, supporter_neighorhood_initial, cv2.TM_CCOEFF_NORMED)[0][0]
            except Exception as e:
                print("EXCEPTION IN FEATURE: ", i)
                continue

            curr_supporter_displacement_mean = feature_params[i][0]
            curr_supporter_covariance = feature_params[i][1]


            # TODO: check if this is right
            rv = scipy.stats.multivariate_normal(mean=curr_supporter_displacement_mean, cov = curr_supporter_covariance)
            predicted_probability += rv.pdf(predicted_target_point - curr_supporter_point) * supporter_correlation_coefficient

            numerator += (curr_supporter_displacement_mean + curr_supporter_point) * supporter_correlation_coefficient / np.sqrt(np.linalg.det(curr_supporter_covariance))
            denominator += supporter_correlation_coefficient / np.sqrt(np.linalg.det(curr_supporter_covariance))

        if predicted_probability >= theta_prediction:
            print("SUPPORTERS CONFIDENT IN TRACKING, USING TRACKING RESULT")
            target_point_final = predicted_target_point
        else:
            print("USING SUPPORTER PREDICTION")
            target_point_final = numerator / denominator

    # TODO: check this
    if new_feature_params == []:
        return target_point_final, feature_params
    else:
        return target_point_final, new_feature_params


def initialize_supporters(supporter_points, target_point):
    supporters = []
    supporter_params = []
    for i in range(len(supporter_points)):
        supporter_point = supporter_points[i][0]
        supporters.append(supporter_point)
        supporter_params.append((target_point - supporter_point, np.eye(2)))
    return supporters, supporter_params

def format_supporters(supporter_points):
    supporters = []
    for i in range(len(supporter_points)):
        supporters.append(supporter_points[i][0])
    return supporters

def angle(vec1, vec2):
    inter = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.arccos(inter)






