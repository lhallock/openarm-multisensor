import time
import os

import cv2
import numpy as np
from multisensorimport.tracking import schreiberAlgorithm as schreiber
from multisensorimport.tracking import schreibersAlgorithm as schreibers
from multisensorimport.tracking import lucasKanadeWarp as LKWarp
from multisensorimport.tracking import supporters as supporters
from multisensorimport.tracking import supporters_utils as supporters_utils



def main():
    # image info
    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/Exercise .mp4')
    cap.read()
    cap.read()
    cap.read()
    cap.read()
    cap.read()
    ret, imgOne = cap.read()
    ret, imgTwo = cap.read()
    # imgOnePath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball1.png'
    # imgTwoPath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball2.png'
    #
    # imgOne = cv2.imread(imgOnePath, -1)
    imgOneGray = cv2.cvtColor(imgOne, cv2.COLOR_RGB2GRAY)
    #
    #
    #
    # imgTwo = cv2.imread(imgTwoPath, -1)
    imgTwoGray = cv2.cvtColor(imgTwo, cv2.COLOR_RGB2GRAY)
    cv2.imshow('image one', imgOneGray)
    cv2.imshow('image two', imgTwoGray)
    cv2.waitKey()
    #
    # tracking info
    pointX = 105
    pointY = 583
    point = np.array([pointX, pointY])

    windowSize = 100

    # init errors
    errors = np.zeros(imgOneGray.shape)

    print(type(imgOneGray[100][100]))

    # init warp params
    fullWarpParams = np.zeros(6)
    oneStepWarpParams = np.zeros(6)

    # Params
    maxIters = 100
    eps = 0.3
    alpha = 0.1
    errorMedian = 0

    # fullWarpParams, oneStepWarpParams, errorMedian = schreiber.computeWarpOpticalFlow(fullWarpParams, oneStepWarpParams, imgTwoGray, imgOneGray, imgOneGray, errors, errorMedian, startX, endX, startY, endY, maxIters, eps, alpha)
    warpParams = LKWarp.lucas_kanade_affine_warp(imgTwoGray, imgOneGray, fullWarpParams, )
    fullWarpParams = schreiber.computeLucasKanadeOpticalFlow(fullWarpParams, imgTwoGray, imgOneGray, point, windowSize, maxIters, eps)

    newPoint = schreiber.affineWarp(point, fullWarpParams)
    print('OLD ', point)
    print('NEW ', newPoint)

    cv2.circle(imgOneGray, (pointX, pointY), 5, (0, 255, 0), -1)
    cv2.imshow('Frame', imgOneGray)


    cv2.circle(imgTwoGray, (int(round(newPoint[0])), int(round(newPoint[1]))), 5, (0, 255, 0), -1)
    cv2.imshow('Frame', imgTwoGray)
    cv2.waitKey()

def simpleTesting():
    imgOne = np.zeros((8,8))
    imgOne[1][1] = 1

    imgTwo = np.zeros((8,8))
    imgTwo[2][3] = 1

    fullWarpParams = np.zeros(6)
    maxIters = 1
    eps = 0.001

    point = np.array([1,1])
    windowSize = 3
    fullWarpParams = schreiber.computeLucasKanadeOpticalFlow(fullWarpParams, imgTwo, imgOne, point, windowSize, maxIters, eps)
    newPoint = schreiber.affineWarp(point, fullWarpParams)
    print('OLD ', point)
    print('NEW ', newPoint)

def testingTwo():
    # image info
    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/Exercise .mp4')
    cap.read()
    cap.read()
    cap.read()
    cap.read()
    cap.read()
    ret, imgOne = cap.read()
    ret, imgTwo = cap.read()
    # imgOnePath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball1.png'
    # imgTwoPath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball2.png'
    #
    # imgOne = cv2.imread(imgOnePath, -1)
    imgOneGray = cv2.cvtColor(imgOne, cv2.COLOR_RGB2GRAY)
    #
    #
    #
    # imgTwo = cv2.imread(imgTwoPath, -1)
    imgTwoGray = cv2.cvtColor(imgTwo, cv2.COLOR_RGB2GRAY)
    #
    # tracking info
    pointX = 105
    pointY = 583


    point = np.array([pointX, pointY])
    print('Original: ', point)

    windowSize = 100
    warp_params = np.zeros(6)

    xLower = pointX - (windowSize // 2)
    xUpper = pointX + (windowSize // 2)
    yLower = pointY - (windowSize // 2)
    yUpper = pointY + (windowSize // 2)

    template = imgOneGray[yLower:(yUpper + 1), xLower:(xUpper+1)]

    newP = LKWarp.lucas_kanade_affine_warp(imgTwoGray, imgOneGray, warp_params, np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 1000)
    newPoint = schreiber.affineWarp(point, newP)
    print('Lucas Kanade Warp: ', newPoint)
    # cv2.imshow('img 2', imgTwoGray)
    # cv2.waitKey()


def testingThree():
    # image info
    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/Exercise .mp4')
    cap.read()
    cap.read()
    cap.read()
    cap.read()
    cap.read()
    ret, imgOne = cap.read()
    ret, imgTwo = cap.read()
    # imgOnePath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball1.png'
    # imgTwoPath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball2.png'
    #
    # imgOne = cv2.imread(imgOnePath, -1)
    imgOneGray = cv2.cvtColor(imgOne, cv2.COLOR_RGB2GRAY)
    #
    #
    #
    # imgTwo = cv2.imread(imgTwoPath, -1)
    imgTwoGray = cv2.cvtColor(imgTwo, cv2.COLOR_RGB2GRAY)
    #
    # tracking info
    pointX = 105
    pointY = 583


    point = np.array([pointX, pointY])
    print('Schreiber0: ', point)

    windowSize = 100
    full_warp_params = np.zeros(6)
    one_step_warp_params = np.zeros(6)

    errors = np.zeros(imgOneGray.shape)

    xLower = pointX - (windowSize // 2)
    xUpper = pointX + (windowSize // 2)
    yLower = pointY - (windowSize // 2)
    yUpper = pointY + (windowSize // 2)

    full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(imgTwoGray, imgOneGray, imgOneGray, errors, full_warp_params, one_step_warp_params, np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 1000)

    point1 = schreiber.affineWarp(point, full_warp_params)
    print('Schreiber1: ', point1)
    # print(errors)

    ret, imgThree = cap.read()
    imgThreeGray = cv2.cvtColor(imgThree, cv2.COLOR_RGB2GRAY)
    full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(imgThreeGray, imgOneGray, imgTwoGray, errors, full_warp_params, one_step_warp_params, np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 1000)
    point2 = schreiber.affineWarp(point, full_warp_params)
    print('Schreiber2: ', point2)

    ret, imgFour = cap.read()
    imgFourGray = cv2.cvtColor(imgFour, cv2.COLOR_RGB2GRAY)
    full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(imgFourGray, imgOneGray, imgThreeGray, errors, full_warp_params, one_step_warp_params, np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 1000)
    point3 = schreiber.affineWarp(point, full_warp_params)
    print('Schreiber3: ', point3)

    ret, imgFive = cap.read()
    imgFiveGray = cv2.cvtColor(imgFive, cv2.COLOR_RGB2GRAY)
    full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(imgFiveGray, imgOneGray, imgFourGray, errors, full_warp_params, one_step_warp_params,  np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 1000)
    point4 = schreiber.affineWarp(point, full_warp_params)
    print('Schreiber4: ', point4)

    ret, imgSix = cap.read()
    imgSixGray = cv2.cvtColor(imgSix, cv2.COLOR_RGB2GRAY)
    full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(imgSixGray, imgOneGray, imgFiveGray, errors, full_warp_params, one_step_warp_params,  np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 1000)
    point5 = schreiber.affineWarp(point, full_warp_params)
    print('Schreiber5: ', point5)

    ret, imgSeven = cap.read()
    imgSevenGray = cv2.cvtColor(imgSeven, cv2.COLOR_RGB2GRAY)
    full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(imgSevenGray, imgOneGray, imgSixGray, errors, full_warp_params, one_step_warp_params,  np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 1000, debug = True)
    point6 = schreiber.affineWarp(point, full_warp_params)
    print('Schreiber6: ', point6)

    cv2.imshow('6', imgSevenGray)
    cv2.imshow('5', imgSixGray)
    cv2.imshow('4', imgFiveGray)
    cv2.imshow('3', imgFourGray)
    cv2.imshow('2', imgThreeGray)
    cv2.imshow('1', imgTwoGray)
    cv2.imshow('0', imgOneGray)
    cv2.waitKey()

    # cv2.imshow('img 2', imgTwoGray)
    # cv2.waitKey()

def testingFour():
    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/Exercise .mp4')
    iter = 0
    first_template = None
    curr_template = None
    curr_image = None
    errors = None


    pointX = 105
    pointY = 583
    windowSize = 100

    point = np.array([pointX, pointY])
    xLower = pointX - (windowSize // 2)
    xUpper = pointX + (windowSize // 2)
    yLower = pointY - (windowSize // 2)
    yUpper = pointY + (windowSize // 2)

    full_warp_params = np.zeros(6)
    one_step_warp_params = np.zeros(6)

    while True:
        print(iter)
        ret, frame = cap.read()

        if not ret:
            break

        if iter == 0:
            first_template_color = frame
            first_template = cv2.cvtColor(first_template_color, cv2.COLOR_RGB2GRAY)
            curr_template_color = frame
            curr_template = cv2.cvtColor(curr_template_color, cv2.COLOR_RGB2GRAY)
            errors = np.zeros(first_template.shape)
            iter += 1
            continue

        curr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(curr_image, first_template, curr_template, errors, full_warp_params, one_step_warp_params, np.array([xLower, yLower]), np.array([xUpper, yUpper]),0.3, 100)
        curr_template = curr_image
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        iter += 1
        if key == 27:
            break
        time.sleep(0.01)


def testingFive():
    # image info
    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/testVid.mp4')

    ret, imgOne = cap.read()
    ret, imgTwo = cap.read()
    # imgOnePath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball1.png'
    # imgTwoPath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball2.png'
    #
    # imgOne = cv2.imread(imgOnePath, -1)
    imgOneGray = cv2.cvtColor(imgOne, cv2.COLOR_RGB2GRAY)
    #
    #
    #
    # imgTwo = cv2.imread(imgTwoPath, -1)
    imgTwoGray = cv2.cvtColor(imgTwo, cv2.COLOR_RGB2GRAY)
    #
    # tracking info
    pointX = 520
    pointY = 35


    point = np.array([pointX, pointY])

    windowSize = 100
    full_warp_params = np.zeros(6)
    print('LK0: ', point)


    xLower = 506
    xUpper = 533
    yLower = 20
    yUpper = 50

    x_coords = np.arange(xLower, xUpper + 1)
    y_coords = np.arange(yLower, yUpper + 1)

    X, Y = np.meshgrid(x_coords, y_coords)

    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgTwoGray, imgOneGray, X, Y, 0.3, 1000)

    X, Y = LKWarp.affine_warp_point_set(X, Y, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    cv2.circle(imgTwo, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)
    print('LK1: ', point)
    # print(errors)

    ret, imgThree = cap.read()
    imgThreeGray = cv2.cvtColor(imgThree, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgThreeGray, imgTwoGray, X, Y, 0.03, 1000)
    X, Y = LKWarp.affine_warp_point_set(X, Y, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    cv2.circle(imgThree, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)

    print('LK2: ', point)

    ret, imgFour = cap.read()
    imgFourGray = cv2.cvtColor(imgFour, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgFourGray, imgThreeGray, X, Y, 0.03, 1000)
    X, Y = LKWarp.affine_warp_point_set(X, Y, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    cv2.circle(imgFour, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)

    print('LK3: ', point)

    ret, imgFive = cap.read()
    imgFiveGray = cv2.cvtColor(imgFive, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgFiveGray, imgFourGray, X, Y, 0.03, 1000)
    X, Y = LKWarp.affine_warp_point_set(X, Y, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    cv2.circle(imgFive, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)

    print('LK4: ', point)

    ret, imgSix = cap.read()
    imgSixGray = cv2.cvtColor(imgSix, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgSixGray, imgFiveGray, X, Y, 0.03, 1000)
    X, Y = LKWarp.affine_warp_point_set(X, Y, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    cv2.circle(imgSix, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)

    print('LK5: ', point)

    ret, imgSeven = cap.read()
    imgSevenGray = cv2.cvtColor(imgSeven, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgSevenGray, imgSixGray, X, Y, 0.03, 1000)
    X, Y = LKWarp.affine_warp_point_set(X, Y, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    cv2.circle(imgSeven, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)

    print('LK6: ', point)

    cv2.imshow('6', imgSeven)
    cv2.imshow('5', imgSix)
    cv2.imshow('4', imgFive)
    cv2.imshow('3', imgFour)
    cv2.imshow('2', imgThree)
    cv2.imshow('1', imgTwo)
    cv2.imshow('0', imgOne)
    cv2.waitKey()

def testingSix():

    pointX = 570
    pointY = 145
    point = np.array([pointX, pointY])

    xLower = 555
    xUpper = 585
    yLower = 130
    yUpper = 160
    x_coords = np.arange(xLower, xUpper + 1)
    y_coords = np.arange(yLower, yUpper + 1)

    X, Y = np.meshgrid(x_coords, y_coords)

    point_mapping = [(point, (X, Y))]

    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/walking.mp4')
    ret, prev_img = cap.read()
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)

    if not ret:
        return

    frame_num = 0
    cv2.namedWindow('Frame')
    while(True):
        ret, curr_img = cap.read()
        if not ret:
            break
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
        point_mapping = LKWarp.multi_point_lucas_kanade(curr_img_gray, prev_img_gray, point_mapping, 0.03, 100, 0)

        for p_m in point_mapping:
            point = p_m[0]
            # print(point)
            cv2.circle(curr_img, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)

        cv2.imshow('Frame', curr_img)
        key = cv2.waitKey(1)
        if key == 27:  # stop on escape key
            break
        time.sleep(0.01)

        prev_img_gray = curr_img_gray
        frame_num +=1


def testingSeven():
    window_size = 5
    point_x_1 = 114
    point_y_1 = 46
    point_1 = np.array([point_x_1, point_y_1])

    x_coords = np.arange(point_x_1 - window_size//2, point_x_1 +window_size // 2 + 1)
    y_coords = np.arange(point_y_1 - window_size//2, point_y_1 +window_size // 2 + 1)

    X_1, Y_1 = np.meshgrid(x_coords, y_coords)
    X_1 = X_1.flatten()
    Y_1 = Y_1.flatten()

    point_mapping = [(point_1, (X_1, Y_1))]

    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/test.mp4')


    ret, prev_img = cap.read()


    if not ret:
        return

    prev_img_gray = bilateralFilter(prev_img)


    frame_num = 0
    cv2.namedWindow('Frame')
    while(True):
        ret, curr_img = cap.read()
        curr_img_gray = bilateralFilter(curr_img)

        if not ret:
            break
        point_mapping = LKWarp.multi_point_lucas_kanade(curr_img_gray, prev_img_gray, point_mapping, 0.03, 1000, 0)

        for p_m in point_mapping:
            point_1 = p_m[0]
            print(point_1)
            cv2.circle(curr_img, (int(np.rint(point_1[0])), int(np.rint(point_1[1]))), 5, (0, 255, 0), -1)

        cv2.imshow('Frame', curr_img)
        key = cv2.waitKey(1)
        if key == 27:  # stop on escape key
            break
        time.sleep(0.01)

        prev_img_gray = curr_img_gray
        frame_num +=1

def testing_eight():
    window_size = 19

    points = [np.array([139., 169.]), np.array([137., 171.]), np.array([137., 172.]), np.array([149., 150.]), np.array([143., 111.]), np.array([149., 129.]), np.array([ 58.,  47.]), np.array([123., 196.]), np.array([113., 44.])]
    point_mapping = generate_point_mappings(points, window_size)

    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/test.mp4')



    ret, prev_img = cap.read()

    if not ret:
        return

    prev_img_gray = bilateralFilter(prev_img)


    frame_num = 0
    cv2.namedWindow('Frame')
    while(True):
        ret, curr_img = cap.read()
        curr_img_gray = bilateralFilter(curr_img)


        if not ret:
            break
        point_mapping = LKWarp.multi_point_lucas_kanade(curr_img_gray, prev_img_gray, point_mapping, 0.03, 1000)

        for p_m in point_mapping:
            point_1 = p_m[0]
            # print(point_1)
            cv2.circle(curr_img, (int(np.rint(point_1[0])), int(np.rint(point_1[1]))), 5, (0, 255, 0), -1)

        cv2.imshow('Frame', curr_img)
        key = cv2.waitKey(1)
        if key == 27:  # stop on escape key
            break
        time.sleep(0.01)

        prev_img_gray = curr_img_gray
        frame_num +=1


def testing_nine():

    pointX = 520
    pointY = 35
    point = np.array([pointX, pointY])

    xLower = 506
    xUpper = 533
    yLower = 20
    yUpper = 50
    x_coords = np.arange(xLower, xUpper + 1)
    y_coords = np.arange(yLower, yUpper + 1)

    point1 = np.array([xLower, yLower])
    point2 = np.array([xUpper, yUpper])

    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/testVid.mp4')
    ret, prev_img = cap.read()
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    first_template_gray = prev_img_gray
    prev_template_gray = prev_img_gray
    full_warp_params = np.zeros(6)
    one_step_warp_params = np.zeros(6)
    errors = np.zeros(first_template_gray.shape)

    if not ret:
        return

    frame_num = 0
    cv2.namedWindow('Frame')
    while(True):
        ret, curr_img = cap.read()
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
        if not ret:
            break
        full_warp_params, one_step_warp_params, errors = schreibers.robust_drift_corrected_tracking(curr_img_gray, first_template_gray, prev_template_gray, errors, full_warp_params, one_step_warp_params, point1, point2, 0.03, 1000)

        point = schreibers.affineWarp(point, one_step_warp_params)
        cv2.circle(curr_img, (int(np.rint(point[0])), int(np.rint(point[1]))), 5, (0, 255, 0), -1)

        cv2.imshow('Frame', curr_img)
        key = cv2.waitKey(1)
        if key == 27:  # stop on escape key
            break
        time.sleep(0.01)

        prev_template_gray = curr_img_gray
        frame_num +=1


def testing_CSRT():
    tracker = cv2.TrackerCSRT_create()
    window_size = 5
    point_x_1 = 114
    point_y_1 = 46
    point_1 = np.array([point_x_1, point_y_1])

    x_coords = np.arange(point_x_1 - window_size//2, point_x_1 +window_size // 2 + 1)
    y_coords = np.arange(point_y_1 - window_size//2, point_y_1 +window_size // 2 + 1)

    X_1, Y_1 = np.meshgrid(x_coords, y_coords)
    X_1 = X_1.flatten()
    Y_1 = Y_1.flatten()

    point_mapping = [(point_1, (X_1, Y_1))]

    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/test.mp4')

    bbox = (point_x_1 - window_size//2, point_y_1 - window_size//2, 7, 7)
    print(bbox)


    ret, prev_img = cap.read()


    if not ret:
        return

    prev_img_gray = bilateralFilter(prev_img)

    tracker.init(prev_img_gray, bbox)


    frame_num = 0
    cv2.namedWindow('Frame')
    while(True):
        ret, curr_img = cap.read()
        curr_img_gray = bilateralFilter(curr_img)

        ok, bbox = tracker.update(curr_img_gray)

        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(curr_img, p1, p2, (255, 0, 0), 2, 1)

        if not ret:
            break
        point_mapping = LKWarp.multi_point_lucas_kanade(curr_img_gray, prev_img_gray, point_mapping, 0.03, 1000)

        for p_m in point_mapping:
            point_1 = p_m[0]
            print(point_1)
            # cv2.circle(curr_img, (int(np.rint(point_1[0])), int(np.rint(point_1[1]))), 5, (0, 255, 0), -1)

        cv2.imshow('Frame', curr_img)
        key = cv2.waitKey(1)
        if key == 27:  # stop on escape key
            break
        time.sleep(0.01)

        prev_img_gray = curr_img_gray
        frame_num +=1

def generate_point_mappings(points, window_size):
    ret = []
    for point in points:
        x = point[0]
        y = point[1]
        x_lower = x - window_size // 2
        x_upper = x + window_size // 2

        y_lower = y - window_size // 2
        y_upper = y + window_size // 2

        x_coords = np.arange(x_lower, x_upper + 1)
        y_coords = np.arange(y_lower, y_upper + 1)

        X, Y = np.meshgrid(x_coords, y_coords)
        X = X.flatten()
        Y = Y.flatten()

        ret.append((point, (X, Y)))
    return ret

def test():

    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7)

    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/test.mp4')
    ret, prev_img = cap.read()
    filtered = bilateralFilter(prev_img)
    pts_good = cv2.goodFeaturesToTrack(filtered, mask=None, **feature_params)
    print(type(pts_good))

    cv2.imshow("Frame", filtered)
    cv2.waitKey()


def filter_supporters():
    print("FILTERING SUPPORTERS")
    lk_params = dict(winSize=(25, 25), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.5,
                          minDistance = 7,
                          blockSize = 7)

    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/test.mp4')
    ret, prev_img = cap.read()
    # prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    # prev_img_gray = cv2.pyrDown(prev_img_gray)
    prev_img_gray = bilateralFilter(prev_img)

    good_trackers = cv2.goodFeaturesToTrack(prev_img_gray, mask=None, **feature_params)
    supporter_points = supporters.format_supporters(good_trackers)

    movement = []
    for i in range(len(supporter_points)):
        movement.append(0)

    num_frames = 0
    while(True):
        ret, curr_img = cap.read()
        if not ret:
            break
        num_frames += 1


        curr_img_gray = bilateralFilter(curr_img)

        good_trackers, status, error = cv2.calcOpticalFlowPyrLK(prev_img_gray, curr_img_gray, good_trackers, None,
                                                             **lk_params)
        new_supporter_points = supporters.format_supporters(good_trackers)
        for i in range(len(new_supporter_points)):
            new_supporter_point = new_supporter_points[i]
            prev_supporter_point = supporter_points[i]
            movement[i] += np.linalg.norm(new_supporter_point - prev_supporter_point)

    movement = np.array(movement)
    movement / num_frames
    moved_points = movement >= 150000
    print(moved_points)
    print("FINISHED FILTERING")
    return moved_points





def supportersTesting():
    lk_params = dict(winSize=(25, 25), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.5,
                          minDistance = 7,
                          blockSize = 7)

    x_init = 125
    y_init = 72
    point_init = np.array([x_init, y_init])

    # supporter_init_x = 263
    # supporter_init_y = 51
    # supporter_init = np.array([supporter_init_x, supporter_init_y])
    #
    # feature_params = [(point_init - supporter_init, np.eye(2))]
    #
    # original_supporters = [supporter_init]
    #
    # curr_points = [np.array([[x_init, y_init]], dtype=np.float32), np.array([[supporter_init_x, supporter_init_y]], dtype=np.float32)]
    # curr_points = np.array(curr_points)


    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/test.mp4')
    ret, prev_img = cap.read()
    # prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    # prev_img_gray = cv2.pyrDown(prev_img_gray)
    prev_img_gray = bilateralFilter(prev_img)
    original_img_gray = prev_img_gray.copy()

    good_trackers = cv2.goodFeaturesToTrack(original_img_gray, mask=None, **feature_params)
    supporter_points, supporter_features = supporters_utils.initialize_supporters(good_trackers, point_init, 10)
    original_supporters = supporter_points.copy()

    points_to_track = np.insert(good_trackers, 0, point_init, axis=0)



    frame = 1
    while (True):
        print("FRAME: ", frame)
        ret, curr_img = cap.read()
        if not ret:
            break
        # curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
        # curr_img_gray = cv2.pyrDown(curr_img_gray)
        # curr_img = cv2.pyrDown(curr_img)
        curr_img_gray = bilateralFilter(curr_img)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_img_gray, curr_img_gray, points_to_track, None, **lk_params)

        predicted_target = new_points[0][0]
        curr_supporters = supporters.format_supporters(new_points[1:])

        target_point, supporter_features = supporters.apply_supporters_model(original_supporters, curr_img_gray,
                                                                         original_img_gray, predicted_target,
                                                                         point_init, curr_supporters, supporter_features,
                                                                         0.55, 0.7, 0.7, 8)
                                                                        # theta corr, theta pred, alpha

        # try:
        #     target_point, feature_params = supporters.apply_supporters_model(original_supporters, curr_img_gray, original_img_gray, predicted_target, point_init, curr_supporters, feature_params, 0.9, 0.9, 1, 10)
        # except Exception as e:
        #     target_point = predicted_target
        #     print("EXCEPTION IN SUPPORTERS METHOD: ", repr(e))

        new_points[0][0] = target_point
        points_to_track = new_points
        prev_img_gray = curr_img_gray



        for i in range(len(new_points)):
            point = new_points[i][0]
            if i == 0:
                cv2.circle(curr_img, (int(np.rint(point[0])), int(np.rint(point[1]))), 7, (0, 255, 0), -1)
            else:
                cv2.circle(curr_img, (int(np.rint(point[0])), int(np.rint(point[1]))), 7, (255, 0, 0), -1)

        cv2.imshow("Frame", curr_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        time.sleep(0.01)
        frame += 1





def simple_supporters_testing():
    lk_params = dict(winSize=(25, 25), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.5,
                          minDistance = 7,
                          blockSize = 7)

    x_init = 125
    y_init = 72
    point_init = np.array([x_init, y_init])


    valid_supporters = filter_supporters()


    cap = cv2.VideoCapture('/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/test.mp4')
    ret, prev_img = cap.read()

    prev_img_gray = bilateralFilter(prev_img)
    original_img_gray = prev_img_gray.copy()

    supporters_tracking = cv2.goodFeaturesToTrack(original_img_gray, mask=None, **feature_params)
    supporters_tracking = supporters_tracking[valid_supporters]
    supporter_points, supporter_features = supporters.initialize_supporters(supporters_tracking, point_init)

    num_tracking_frames = 10

    target_tracking = np.empty([1, 1, 2], dtype=np.float32)
    target_tracking[0][0] = [x_init, y_init]

    frame = 1
    while (True):
        print("FRAME: ", frame)
        ret, curr_img = cap.read()
        if not ret:
            break

        curr_img_gray = bilateralFilter(curr_img)

        print(target_tracking)

        target_tracking, status, error = cv2.calcOpticalFlowPyrLK(prev_img_gray, curr_img_gray, target_tracking, None, **lk_params)

        supporters_tracking, status, error = cv2.calcOpticalFlowPyrLK(prev_img_gray, curr_img_gray, supporters_tracking, None, **lk_params)

        predicted_target = target_tracking[0][0]
        supporter_points = supporters_utils.format_supporters(supporters_tracking)

        use_tracking = frame <= num_tracking_frames

        target_point, supporter_features = supporters_utils.apply_supporters_model(predicted_target, supporters_tracking, supporter_features, use_tracking, 0.75, 50)



        target_tracking[0] = target_point

        prev_img_gray = curr_img_gray


        cv2.circle(curr_img, (int(np.rint(target_point[0])), int(np.rint(target_point[1]))), 7, (0, 255, 0), -1)

        for i in range(len(supporters_tracking)):
            supporter_point = supporters_tracking[i][0]
            cv2.circle(curr_img, (int(np.rint(supporter_point[0])), int(np.rint(supporter_point[1]))), 7, (255, 0, 0), -1)


        cv2.imshow("Frame", curr_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        time.sleep(0.01)
        frame += 1




def bilateralFilter(color_image):
    # color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)

    # hyperparameters
    diam = 25
    sigma_color = 100
    sigma_space = 100
    bilateral_color = cv2.bilateralFilter(color_image, diam, sigma_color, sigma_space)
    return cv2.cvtColor(bilateral_color, cv2.COLOR_RGB2GRAY)


#
#
# computeWarpOpticalFlow(fullWarpParams, oneStepWarpParams, currImage, currTemplateImage, firstTemplateImage, currCumulativeErrors, eta, xLower, xUpper, yLower, yUpper, maxIters, eps, alpha):

if __name__ == "__main__":
    simple_supporters_testing()
