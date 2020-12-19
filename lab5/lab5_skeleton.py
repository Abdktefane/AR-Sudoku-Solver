import numpy as np
import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('resources/sample1.jpg', 0)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    cv.imshow('skel', skel)
    cv.waitKey(0)
    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    cv.imshow('img', img)
    cv.waitKey(0)

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv.erode(img, element)
        dilated = cv.dilate(eroded, element)
        new_img = dilated - eroded
        skel = cv.bitwise_or(skel, new_img)
        cv.imshow('skel', skel)
        cv.waitKey(0)
        img = eroded.copy()
        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True

cv.imshow('skel', skel)
cv.waitKey(0)
