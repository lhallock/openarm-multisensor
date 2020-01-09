import time
import os

import cv2
import numpy as np
from multisensorimport.tracking import schreiberAlgorithm as schreiber
from multisensorimport.tracking import schreibersAlgorithm as schreibers
from multisensorimport.tracking import lucasKanadeWarp as LKWarp


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
    warpParams = LKWarp.lucas_kanade_affine_warp(imageTwoGray, imageOneGray, fullWarpParams, )
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
        warped
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        iter += 1
        if key == 27:
            break
        time.sleep(0.01)


def testingFive():
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

    windowSize = 100
    full_warp_params = np.zeros(6)
    print('LK0: ', full_warp_params)


    xLower = pointX - (windowSize // 2)
    xUpper = pointX + (windowSize // 2)
    yLower = pointY - (windowSize // 2)
    yUpper = pointY + (windowSize // 2)

    x_coords = np.arange(xLower, xUpper + 1)
    y_coords = np.arange(yLower, yUpper + 1)

    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgTwoGray, imgOneGray, full_warp_params, x_coords, y_coords, 0.3, 1000)

    x_coords, y_coords = LKWarp.affine_warp_point_set(x_coords, y_coords, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    print('LK1: ', point)
    # print(errors)

    ret, imgThree = cap.read()
    imgThreeGray = cv2.cvtColor(imgThree, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgThreeGray, imgTwoGray, full_warp_params, x_coords, y_coords, 0.3, 1000)
    x_coords, y_coords = LKWarp.affine_warp_point_set(x_coords, y_coords, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    print('LK2: ', point)

    ret, imgFour = cap.read()
    imgFourGray = cv2.cvtColor(imgFour, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgFourGray, imgThreeGray, full_warp_params, x_coords, y_coords, 0.3, 1000)
    x_coords, y_coords = LKWarp.affine_warp_point_set(x_coords, y_coords, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    print('LK3: ', point)

    ret, imgFive = cap.read()
    imgFiveGray = cv2.cvtColor(imgFive, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgFiveGray, imgFourGray, full_warp_params, x_coords, y_coords, 0.3, 1000)
    x_coords, y_coords = LKWarp.affine_warp_point_set(x_coords, y_coords, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    print('LK4: ', point)

    ret, imgSix = cap.read()
    imgSixGray = cv2.cvtColor(imgSix, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgSixGray, imgFiveGray, full_warp_params, x_coords, y_coords, 0.3, 1000)
    x_coords, y_coords = LKWarp.affine_warp_point_set(x_coords, y_coords, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    print('LK5: ', point)

    ret, imgSeven = cap.read()
    imgSevenGray = cv2.cvtColor(imgSeven, cv2.COLOR_RGB2GRAY)
    full_warp_params = LKWarp.lucas_kanade_affine_warp(imgSevenGray, imgSixGray, full_warp_params, x_coords, y_coords, 0.3, 1000)
    x_coords, y_coords = LKWarp.affine_warp_point_set(x_coords, y_coords, full_warp_params)
    point = LKWarp.affine_warp_single_point(point, full_warp_params)
    print('LK6: ', point)

    cv2.imshow('6', imgSevenGray)
    cv2.imshow('5', imgSixGray)
    cv2.imshow('4', imgFiveGray)
    cv2.imshow('3', imgFourGray)
    cv2.imshow('2', imgThreeGray)
    cv2.imshow('1', imgTwoGray)
    cv2.imshow('0', imgOneGray)
    cv2.waitKey()




#
#
# computeWarpOpticalFlow(fullWarpParams, oneStepWarpParams, currImage, currTemplateImage, firstTemplateImage, currCumulativeErrors, eta, xLower, xUpper, yLower, yUpper, maxIters, eps, alpha):

if __name__ == "__main__":
    testingFive()
