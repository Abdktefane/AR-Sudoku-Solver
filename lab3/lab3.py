from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


def plot_gray():
    image = cv.imread('kotlin.png', 0)
    cv.imshow('base', image)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()
    cv.waitKey(0)


def plot_color():
    image = cv.imread('kotlin.png')
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()


def plot_equalize_hist():
    image = cv.imread('kotlin.png', 0)
    hist = cv.equalizeHist(image)
    cv.imshow('base', hist)
    cv.waitKey(0)


def blur():
    image = cv.imread('kotlin.png')
    blur = cv.blur(image, (5, 5))
    cv.imshow('base', blur)
    cv.waitKey(0)


def gaussian_blur():
    image = cv.imread('kotlin.png')
    gaus = cv.GaussianBlur(image, (5, 5), 2)
    cv.imshow('base', gaus)
    cv.waitKey(0)


def median_blur():
    image = cv.imread('kotlin.png')
    blur = cv.medianBlur(image, 21)
    cv.imshow('base', blur)
    cv.waitKey(0)


def sobel_filter():
    image = cv.imread('kotlin.png', 0)
    subel_on_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    subel_on_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    subel_on_x_and_y = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=3)

    image_on_x = cv.convertScaleAbs(subel_on_x)
    image_on_y = cv.convertScaleAbs(subel_on_y)
    image_on_x_and_y = cv.convertScaleAbs(subel_on_x_and_y)
    cv.imshow('base_x', image_on_x)
    cv.imshow('base_y', image_on_y)
    cv.imshow('base_x_and_y', image_on_x_and_y)
    cv.waitKey(0)


# TODO add implements herel
# def cany_():


if __name__ == '__main__':
    # plot_gray()
    # plot_color()
    # plot_equalize_hist()
    # blur()
    # gaussian_blur()
    # median_blur()
    sobel_filter()
