import cv2 as cv
import numpy as np


if __name__ == '__main__':
    image = cv.imread('resources/opencv.jpg')
    image = cv.medianBlur(image,5)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    circels = cv.HoughCircles(gray,cv.HOUGH_GRADIENT)

    mask = cv.imread('mask',)