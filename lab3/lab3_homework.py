from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


def sobel_edge_detector():
    image = cv.imread('kotlin.png', 0)
    cv.imshow('image_original', image)
    subel_on_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    subel_on_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    subel_on_x_and_y = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=3)

    image_on_x = cv.convertScaleAbs(subel_on_x)
    image_on_y = cv.convertScaleAbs(subel_on_y)
    image_on_x_and_y = cv.convertScaleAbs(subel_on_x_and_y)
    cv.imshow('sobel base_x', image_on_x)
    cv.imshow('sobel base_y', image_on_y)
    cv.imshow('sobel base_x_and_y', image_on_x_and_y)
    cv.waitKey(0)


def cany_edge_detector(showcase=None):
    threshold1 = 100
    threshold2 = 200
    named_window = 'cany_filter'
    if showcase:  # when the max range is low each pixel detected as edge
        threshold1 = 0
        threshold2 = 0
        named_window += '_max_range_small'

    elif not showcase and showcase is not None:  # when the max range is high no edge detection
        threshold1 = 2000
        threshold2 = 3000
        named_window += '_max_range_high'

    image = cv.imread('kotlin.png', 0)
    filtered_image = cv.Canny(image, threshold1=threshold1, threshold2=threshold2)
    cv.imshow(named_window, filtered_image)
    cv.waitKey(0)


def laplacian_edge_detector():
    image = cv.imread('kotlin.png', 0)
    laplacian_filtered = cv.Laplacian(image, cv.CV_64F, ksize=5)
    laplacian_image = cv.convertScaleAbs(laplacian_filtered)
    cv.imshow('laplacian_filter', laplacian_image)
    cv.waitKey(0)


if __name__ == '__main__':
    #  sobel_edge_detector()
    # cany_edge_detector()
    # cany_edge_detector(True)
    # cany_edge_detector(False)
    laplacian_edge_detector()
