from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


def thresh():
    image = cv.imread('resources/numbers.jpg')
    ret, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    ret, thresh2 = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(image, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(image, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(image, 127, 255, cv.THRESH_TOZERO_INV)

    titles = ['base', 'BINARY', 'BINARY_INV', 'BINARY_TRUNC', 'BINARY_TOZERO', 'BINARY_INV']
    images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
    plt.show()


def adaptive_thresh():
    image = cv.imread('resources/sudoku.jpg', 0)
    image = cv.medianBlur(image, 5)
    ret, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    thresh2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thresh3 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    titles = ['base', 'GLOBAL', 'MEAN', 'GAUS']
    images = [image, thresh1, thresh2, thresh3]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
    plt.show()


def outso_thresh():
    image = cv.imread('resources/coins.jpg', 0)
    ret1, thresh1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    ret2, thresh2 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blur = cv.medianBlur(image, 5)
    ret3, thresh3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    titles = ['base', 'GLOBAL', 'MEAN', 'GAUS']
    images = [image, thresh1, thresh2, thresh3]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
    plt.show()


def erosion(kernal, iter):
    image = cv.imread('resources/j_letter.jpg', 0)
    erosion = cv.erode(image, kernal, iter)
    cv.imshow('erosion', erosion)
    cv.waitKey(0)


def delation(kernal, iter):
    image = cv.imread('resources/j_letter.jpg', 0)
    delation = cv.dilate(image, kernal, iter)
    cv.imshow('delation', delation)
    cv.waitKey(0)


def open(kernal):
    image = cv.imread('resources/j_letter_outside_noise.jpg', 0)
    open = cv.morphologyEx(image,cv.MORPH_OPEN ,kernal)
    cv.imshow('open', open)
    cv.waitKey(0)


if __name__ == '__main__':
    # thresh()
    # adaptive_thresh()
    # outso_thresh()
    kernal = np.ones((5, 5), np.uint8)
    # erosion(kernal, 3)
    # delation(kernal, 3)
    open(kernal)
