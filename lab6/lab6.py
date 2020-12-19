import numpy as np
import cv2 as cv


def fun1():
    img = cv.imread('chess.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.float32(img)
    dst = cv.cornerHarris(img, 2, 3, 0.05)
    dst = cv.dilate(dst, None)
    cv.imshow('dst', dst)
    cv.waitKey(0)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv.imshow('new', img)
    cv.waitKey(0)


def fun2():
    img1 = cv.imread('view.jpg')
    img2 = cv.imread('opencv.jpg')
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    cv.imshow("window1", img2gray)
    cv.waitKey(0)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    cv.imshow("window2", mask)
    cv.waitKey(0)
    mask_inv = cv.bitwise_not(mask)
    cv.imshow("window3", mask_inv)
    cv.waitKey(0)
    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    cv.imshow("window4", img1_bg)
    cv.waitKey(0)
    img2_fg = cv.bitwise_and(img2, img2,mask=mask)
    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv.imshow('res', img1)
    cv.waitKey(0)


if __name__ == '__main__':
    fun2()
