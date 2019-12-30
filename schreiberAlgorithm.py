import numpy as np
import cv2
from scipy import signal



def affineWarp(x, p):
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

def templateErrorFunction(diffVec):
    return np.linalg.norm(diffVec)


def computeSteepestDescentImage(delTx, delTy, point):
    x = point[0]
    y = point[1]

    partialX = delTx[y][x]
    partialY = delTy[y][x]

    return np.array([x * partialX, x * partialY, y * partialX, y * partialY, partialX, partialY])

def computeDeltaP(warpParams, img, templateImg, weights, delTx, delTy, xLower, xUpper, yLower, yUpper):
    numParams = len(warpParams)

    hessian = np.zeros((numParams, numParams))
    vec = np.zeros(numParams)

    for x in range(xLower, xUpper + 1):
        for y in range(yLower, yUpper + 1):
            point = np.array([x, y])
            warpedPoint = affineWarp(point, warpParams)
            warpedX = warpedPoint[0]
            warpedY = warpedPoint[1]

            steepestDescentImage = computeSteepestDescentImage(delTx, delTy, point))
            # TODO: might need to check for out of bounds
            diff = img[warpedY][warpedX] - templateImg[x][y]
            weight = weights[x][y]

            vec += weight * steepestDescentImage * diff
            hessian += weight * np.dot(steepestDescentImage.reshape(len(steepestDescentImage), 1), steepestDescentImage.reshape(1, len(steepestDescentImage)))

    return np.dot(np.linalg.inv(hessian, vec))


def updateP(warpParams, img, templateImg, weights, delTx, delTy, xLower, xUpper, yLower, yUpper, maxIters, eps):
    i = 0
    terminateLoop = False
    while(i < maxIters && !terminateLoop):
        deltaP = computeDeltaP(warpParams, img, templateImg, weights, delTx, delTy, xLower, xUpper, yLower, yUpper)
        warpParams += deltaP
        if (np.linalg.norm(deltaP) <= eps):
            terminateLoop = True

# def computeWarpOpticalFlow(currImg, templateImg, currCumulativeErrors, fullWarpParams, oneStepWarpParams):
