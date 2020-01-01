import numpy as np
import cv2
from scipy import signal



def affineWarp(xVec, p):
    """ Applies an affine warp, parameterized by p, to coordinates in x (np array)

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

    # construct transform matrix
    transformMat = np.array([np.array([1 + p1, p3, p5]), np.array([p2, 1 + p4, p6])])

    # append 1 for affine shift
    xVec = np.append(xVec, 1)

    return np.dot(transformMat, xVec)


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

def computeRobustWeight(eta, error):
    return eta * error

def templateErrorFunction(diff):
    return abs(diff)


def computeSteepestDescentImage(delTx, delTy, point):
    x = point[0]
    y = point[1]

    partialX = getImageValue(x, y, delTx)
    partialY = getImageValue(x, y, delTy)

    return np.array([x * partialX, x * partialY, y * partialX, y * partialY, partialX, partialY])

def computeDeltaP(warpParams, warpParamsToTemplateImg, img, templateImg, errors, eta, delTx, delTy, xLower, xUpper, yLower, yUpper):
    """
    Does one step of iteration
    """
    numParams = len(warpParams)

    hessian = np.zeros((numParams, numParams))
    vec = np.zeros(numParams)

    for x in range(xLower, xUpper + 1):
        for y in range(yLower, yUpper + 1):
            point = np.array([x, y])
            warpedPoint = affineWarp(point, warpParams)
            warpedX = warpedPoint[0]
            warpedY = warpedPoint[1]

            steepestDescentImage = computeSteepestDescentImage(delTx, delTy, point)

            if warpParamsToTemplateImg is not None:
                templatePoint = affineWarp(point, warpParamsToTemplateImg)
            else:
                templatePoint = point

            templateX = templatePoint[0]
            templateY = templatePoint[1]

            # TODO: might need to check for out of bounds
            diff = getImageValue(warpedX, warpedY, img) - getImageValue(templateX, templateY, templateImg)
            weight = computeRobustWeight(eta, getImageValue(x, y, errors))

            vec += weight * steepestDescentImage * diff
            hessian += weight * np.dot(steepestDescentImage.reshape(len(steepestDescentImage), 1), steepestDescentImage.reshape(1, len(steepestDescentImage)))

    # print(hessian)
    return np.dot(np.linalg.inv(hessian), vec)


def updateP(warpParams, warpParamsToTemplateImg, img, firstTemplateImage, errors, eta, delTx, delTy, xLower, xUpper, yLower, yUpper, maxIters, eps):
    """
    Computes the new warp Params (updates the warpParams argument)

    Args:
        warpParams: initial "guess" for the warp parameters from the template to img
        warpParamsToTemplateImg: warp parameters to get from firstTemplateImage to template
        img: current image we are warping to
        firstTemplateImage: first image (I_0), a window in which we are tracking
        errors: errors used to compute weights for update
        eta: scaling factor used to compute weights
        delTx: partial derivative of img with respect to x
        delTy: partial derivative of img with respect to y
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
        deltaP = computeDeltaP(warpParams, warpParamsToTemplateImg, img, firstTemplateImage, errors, eta, delTx, delTy, xLower, xUpper, yLower, yUpper)
        newWarpParams += deltaP
        if (np.linalg.norm(deltaP) <= eps):
            terminateLoop = True
        i += 1
    return newWarpParams

def computeWarpOpticalFlow(fullWarpParams, oneStepWarpParams, currImage, currTemplateImage, firstTemplateImage, currCumulativeErrors, eta, xLower, xUpper, yLower, yUpper, maxIters, eps, alpha):
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
    prevToCurrWarpParams = updateP(oneStepWarpParams, fullWarpParams, currImage, firstTemplateImage, currCumulativeErrors, eta, delTx_curr, delTy_curr, xLower, xUpper, yLower, yUpper, maxIters, eps)

    # p(0 -> n) parameter first approximation: p(0 -> n) = p(0 -> n-1) + p(n-1 -> n)
    print('p(0 -> n) parameter first approximation: p(0 -> n) = p(0 -> n-1) + p(n-1 -> n)')
    startToCurrWarpParams = fullWarpParams + prevToCurrWarpParams

    # optimize p(0 -> n) using robust algorithm (obtaining p*(0 -> n)
    print('optimize p(0 -> n) using robust algorithm (obtaining p*(0 -> n)')
    startToCurrWarpParams = updateP(startToCurrWarpParams, None, currImage, firstTemplateImage, currCumulativeErrors, eta, delTx_first, delTy_first, xLower, xUpper, yLower, yUpper, maxIters, eps)

    # combine p*(0 -> n-1) and p*(0 -> n) to obtain p*(n-1->n): p*(n-1->n) = p*(0 -> n) - p*(0 -> n-1)
    print('combine p*(0 -> n-1) and p*(0 -> n) to obtain p*(n-1->n): p*(n-1->n) = p*(0 -> n) - p*(0 -> n-1)')
    newOneStepParams = startToCurrWarpParams - fullWarpParams

    # update errors (update in place)
    print('update errors (update in place)')
    print(currCumulativeErrors.shape)
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
            currCumulativeErrors[y][x] = (1 - alpha) * getImageValue(x, y, currCumulativeErrors) + alpha * templateErrorFunction(diff)

    # return p*(0->n), p*(n-1 -> n) for next computation
    print('return p*(0->n), p*(n-1 -> n) for next computation')
    print('startToCurrWarpParams ', startToCurrWarpParams)
    print('newOneStepParams ', newOneStepParams)
    return startToCurrWarpParams, newOneStepParams

# helper methods
def getImageValue(x, y, img):
    x = int(round(x))
    y = int(round(y))
    # print(type(img[y][x]))
    return np.float64(img[y][x])

# gradient computations using sobel derivative
def imageDerivativeX(img):
    sobel_x64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    return sobel_x64f


def imageDerivativeY(img):
    sobel_y64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return sobel_y64f
