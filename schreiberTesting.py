import time
import os

import cv2
import numpy as np
from multisensorimport.tracking import schreiberAlgorithm as schreiber

def main():
    # image info
    imgOnePath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball1.png'
    imgTwoPath = '/Users/akashvelu/Documents/Research_HART2/openarm-multisensor/testData/ball2.png'

    imgOne = cv2.imread(imgOnePath, -1)
    imgOneGray = cv2.cvtColor(imgOne, cv2.COLOR_RGB2GRAY)



    imgTwo = cv2.imread(imgTwoPath, -1)
    imgTwoGray = cv2.cvtColor(imgTwo, cv2.COLOR_RGB2GRAY)

    # tracking info
    pointX = 332
    pointY = 424
    point = np.array([pointX, pointY])

    windowSize = 224
    startX = pointX - (windowSize // 2)
    endX = pointX + (windowSize // 2)
    startY = pointY - (windowSize // 2)
    endY = pointY + (windowSize // 2)



    # init errors
    errors = np.ones(imgOneGray.shape)

    print(type(imgOneGray[100][100]))

    # init warp params
    fullWarpParams = np.zeros(6)
    oneStepWarpParams = np.zeros(6)

    # Params
    maxIters = 2000
    eps = 0.3
    alpha = 0.1
    eta = 1

    fullWarpParams, oneStepWarpParams = schreiber.computeWarpOpticalFlow(fullWarpParams, oneStepWarpParams, imgTwoGray, imgOneGray, imgOneGray, errors, eta, startX, endX, startY, endY, maxIters, eps, alpha)
    newPoint = schreiber.affineWarp(point, fullWarpParams)
    print('OLD ', point)
    print('NEW ', newPoint)

    cv2.circle(imgOneGray, (pointX, pointY), 5, (0, 255, 0), -1)
    cv2.imshow('Frame', imgOneGray)


    cv2.circle(imgTwoGray, (int(round(newPoint[0])), int(round(newPoint[1]))), 5, (0, 255, 0), -1)
    cv2.imshow('Frame', imgTwoGray)
    cv2.waitKey()


#
#
# computeWarpOpticalFlow(fullWarpParams, oneStepWarpParams, currImage, currTemplateImage, firstTemplateImage, currCumulativeErrors, eta, xLower, xUpper, yLower, yUpper, maxIters, eps, alpha):

if __name__ == "__main__":
    main()
