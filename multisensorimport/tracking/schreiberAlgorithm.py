import numpy as np
import cv2
import statistics
from statistics import median
from scipy import signal



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

    # # construct transform matrix
    # transformMat = np.array([np.array([1 + p1, p3, p5]), np.array([p2, 1 + p4, p6])])
    #
    # # append 1 for affine shift
    # xVec = np.append(xVec, 1)
    #
    # return np.dot(transformMat, xVec)


def inverseAffineWarp(x, p):

    """ Applies the inverse affine warp, parameterized by p, to coordinates in x (np array)

    Args:
        x: 2-element numpy array with x, y coordinates of image coordinate
        p: 6-element numpy array of parameters for the affine warp

    Returns:
        2-element numpy array of transformed coordinates
    """
    # unpack warp parameters
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]
    p5 = p[4]
    p6 = p[5]

    # construct inverse transform matrix
    invMat = np.linalg.inv(np.array([np.array([1 + p1, p3]), np.array([p2, 1 + p4])]))

    # numpy array of affine shift
    affineShift = np.array([p5, p6])

    return np.dot(invMat, (x - affineShift))

# returning 1 makes this a standard LK algorithm
def computeRobustWeight(error, medianErrors):
    return 1
    # if error <= medianErrors * 1.4826:
    #     return 1
    # else:
    #     return 0

def templateErrorFunction(diff):
    return abs(diff)


def computeSteepestDescentImage(delIx, delIy, point, warpedPoint):
    x = point[0]
    y = point[1]

    warpedX = warpedPoint[0]
    warpedY = warpedPoint[1]

    partialX = getImageValue(warpedX, warpedY, delIx)
    partialY = getImageValue(warpedX, warpedY, delIy)
    # print('partial X ', partialX)
    # print('partial Y', partialY)

    return np.array([x * partialX, x * partialY, y * partialX, y * partialY, partialX, partialY])

def computeDeltaP(warpParams, warpParamsToTemplateImg, img, templateImg, errors, errorMedian, delIx, delIy, xLower, xUpper, yLower, yUpper):
    """
    Does one step of iteration
    """
    numParams = len(warpParams)

    hessian = np.zeros((numParams, numParams))
    vec = np.zeros(numParams)

    for x in range(xLower, xUpper + 1):
        for y in range(yLower, yUpper + 1):
            # print('x: ', x)
            # print('y: ', y)
            point = np.array([x, y])
            warpedPoint = affineWarp(point, warpParams)
            warpedX = warpedPoint[0]
            warpedY = warpedPoint[1]
            # print('warped x: ', warpedX)
            # print('warped y: ', warpedY)


            if warpParamsToTemplateImg is not None:
                templatePoint = affineWarp(point, warpParamsToTemplateImg)
            else:
                templatePoint = point

            # TODO: check if you need to use point or template point (pretty sure template point)

            steepestDescentImage = computeSteepestDescentImage(delIx, delIy, templatePoint, warpedPoint)
            # print('Steepest descent ', steepestDescentImage)

            templateX = templatePoint[0]
            templateY = templatePoint[1]

            # TODO: might need to check for out of bounds
            diff = getImageValue(templateX, templateY, templateImg) - getImageValue(warpedX, warpedY, img)
            # print('Diff ', diff)
            weight = computeRobustWeight(getImageValue(x, y, errors), errorMedian)
            # print('weight ', weight)

            vec += weight * steepestDescentImage * diff
            hessian += weight * np.dot(steepestDescentImage.reshape(len(steepestDescentImage), 1), steepestDescentImage.reshape(1, len(steepestDescentImage)))

    # print('Hessian ', hessian)
    # print('Vec ', vec)
    return np.dot(np.linalg.inv(hessian), vec)


def updateP(warpParams, warpParamsToTemplateImg, img, firstTemplateImage, errors, errorMedian, delIx, delIy, xLower, xUpper, yLower, yUpper, maxIters, eps):
    """
    Computes the new warp Params (updates the warpParams argument)

    Args:
        warpParams: initial "guess" for the warp parameters from the template to img
        warpParamsToTemplateImg: warp parameters to get from firstTemplateImage to template
        img: current image we are warping to
        firstTemplateImage: first image (I_0), a window in which we are tracking
        errors: errors used to compute weights for update
        eta: scaling factor used to compute weights
        delIx: partial derivative of img with respect to x
        delIy: partial derivative of img with respect to y
        xLower: lower x (horizontal) point for template window in firstTemplateImage
        xUpper: upper x (horizontal) ending for template window in firstTemplateImage
        yLower: lower y (vertical) point for template window in firstTemplateImage
        yUpper: upper y (vertical) ending for template window in firstTemplateImage
        maxIters: maximum number of iterations updateP should run for
        eps: when to stop update deltaP

    Returns:
        Updated warp parameters mapping from template to img
    """
    i = 0
    terminateLoop = False
    newWarpParams = warpParams.copy()
    while((i < maxIters) and (not terminateLoop)):
        deltaP = computeDeltaP(newWarpParams, warpParamsToTemplateImg, img, firstTemplateImage, errors, errorMedian, delIx, delIy, xLower, xUpper, yLower, yUpper)
        newWarpParams += deltaP
        if (np.linalg.norm(deltaP) <= eps):
            terminateLoop = True
        i += 1
    return newWarpParams

def computeLucasKanadeOpticalFlow(fullWarpParams, currImage, templateImage, point, windowSize, maxIters, eps):
    pointX = point[0]
    pointY = point[1]

    xLower = pointX - (windowSize // 2)
    xUpper = pointX + (windowSize // 2)
    yLower = pointY - (windowSize // 2)
    yUpper = pointY + (windowSize // 2)

    errors = np.zeros(templateImage.shape)
    print('image derivative computation for image')
    delIx = imageDerivativeX(currImage)
    delIy = imageDerivativeY(currImage)
    # print(delTy)
    return updateP(fullWarpParams, None, currImage, templateImage, errors, 1, delIx, delIy, xLower, xUpper, yLower, yUpper, maxIters, eps)

def computeDriftCorrectedOpticalFlow(fullWarpParams, oneStepWarpParams, currImage, currTemplateImage, firstTemplateImage, currCumulativeErrors, errorMedian, point, windowSize, maxIters, eps, alpha):
    """
    Args:
        fullWarpParams: params p*(0 -> n-1)
        oneStepWarpParams: params p*(n-2 -> n-1)
        currImage: current image (I_n)
        currTemplateImage: should be image before currImage (I_n-1) (T_n)
        firstTemplateImage: should be first image (I_0)
        currCumulativeErrors: errors for each point in template at current tep (E_n)
        eta: scaling factor to compute weights from errors
        xLower: lower x (horizontal) point for template window in firstTemplateImage
        xUpper: upper x (horizontal) ending for template window in firstTemplateImage
        yLower: lower y (vertical) point for template window in firstTemplateImage
        yUpper: upper y (vertical) ending for template window in firstTemplateImage
        maxIters: maximum number of iterations updateP should run for
        eps: when to stop update deltaP
        alpha: temporal difference factor when updating errors
    """
    pointX = point[0]
    pointY = point[1]

    xLower = pointX - (windowSize // 2)
    xUpper = pointX + (windowSize // 2)
    yLower = pointY - (windowSize // 2)
    yUpper = pointY + (windowSize // 2)

    # image derivative computation for currTemplateImage
    print('image derivative computation for currTemplateImage')
    delTx_curr = imageDerivativeX(currTemplateImage)
    delTy_curr = imageDerivativeY(currTemplateImage)

    # image derivative computation for firstTemplateImage
    print('image derivative computation for firstTemplateImage')
    delTx_first = imageDerivativeX(firstTemplateImage)
    delTy_first = imageDerivativeY(firstTemplateImage)

    # p(n - 1 -> n) parameter first approximation
    print('p(n - 1 -> n) parameter first approximation')
    prevToCurrWarpParams = updateP(oneStepWarpParams, fullWarpParams, currImage, firstTemplateImage, currCumulativeErrors, errorMedian, delTx_curr, delTy_curr, xLower, xUpper, yLower, yUpper, maxIters, eps)

    # p(0 -> n) parameter first approximation: p(0 -> n) = p(0 -> n-1) + p(n-1 -> n)
    print('p(0 -> n) parameter first approximation: p(0 -> n) = p(0 -> n-1) + p(n-1 -> n)')
    startToCurrWarpParams = fullWarpParams + prevToCurrWarpParams

    # optimize p(0 -> n) using robust algorithm (obtaining p*(0 -> n)
    print('optimize p(0 -> n) using robust algorithm (obtaining p*(0 -> n)')
    startToCurrWarpParams = updateP(startToCurrWarpParams, None, currImage, firstTemplateImage, currCumulativeErrors, errorMedian, delTx_first, delTy_first, xLower, xUpper, yLower, yUpper, maxIters, eps)

    # combine p*(0 -> n-1) and p*(0 -> n) to obtain p*(n-1->n): p*(n-1->n) = p*(0 -> n) - p*(0 -> n-1)
    print('combine p*(0 -> n-1) and p*(0 -> n) to obtain p*(n-1->n): p*(n-1->n) = p*(0 -> n) - p*(0 -> n-1)')
    newOneStepParams = startToCurrWarpParams - fullWarpParams

    # update errors (update in place)
    print('update errors (update in place)')
    print(currCumulativeErrors.shape)
    errors = []
    for y in range(yLower, yUpper + 1):
        for x in range(xLower, xUpper + 1):
            point = np.array([x, y])

            # warp point to corresponding point in current image, using newly computed warp parameters
            warpedPoint = affineWarp(point, startToCurrWarpParams)
            warpedX = warpedPoint[0]
            warpedY = warpedPoint[1]

            # difference between current template point and original template point
            diff = getImageValue(warpedX, warpedY, currImage) - getImageValue(x, y, firstTemplateImage)

            # temporal difference update
            error = (1 - alpha) * getImageValue(x, y, currCumulativeErrors) + alpha * templateErrorFunction(diff)
            errors.append(error)
            currCumulativeErrors[y][x] = (1 - alpha) * getImageValue(x, y, currCumulativeErrors) + alpha * templateErrorFunction(diff)

    # return p*(0->n), p*(n-1 -> n) for next computation
    print('return p*(0->n), p*(n-1 -> n) for next computation')
    print('startToCurrWarpParams ', startToCurrWarpParams)
    print('newOneStepParams ', newOneStepParams)
    return startToCurrWarpParams, newOneStepParams, median(errors)

# helper methods
def getImageValue(x, y, img):
    x = int(round(x))
    y = int(round(y))
    # print(type(img[y][x]))
    return np.float64(img[y][x])

# gradient computations using sobel derivative
def imageDerivativeX(img):
    return np.gradient(img)[1]
    # sobel_x64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
    # return sobel_x64f


def imageDerivativeY(img):
    return np.gradient(img)[0]

    # sobel_y64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
    # return sobel_y64f
