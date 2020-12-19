import cv2 as cv
import numpy as np


def resize(image):
    newImage = cv.resize(image, (200, 200), interpolation=cv.INTER_LANCZOS4)
    cv.imshow('resized-newImage', newImage)
    cv.waitKey(0)


def translate(image):
    rows, cols, depth = image.shape
    M = np.float32(
        [[1, 0, 50], [0, 1, 100]]
    )
    newImage = cv.warpAffine(image, M, (cols - 50, rows - 50))
    cv.imshow('newImage', newImage)
    cv.waitKey(0)


def rotate(image):
    rows, cols = image.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    newImage = cv.warpAffine(image, M, (rows, cols))
    cv.imshow('newImage', newImage)
    cv.waitKey(0)


def flip(image):
    newImage = cv.flip(image, 1)
    cv.imshow('newImage', newImage)
    cv.waitKey(0)


def affine(image):
    rows, cols, depth = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pts1, pts2)
    newImage = cv.warpAffine(image, M, (cols, rows))
    cv.imshow('real_affine_newImage', newImage)
    cv.waitKey(0)


def perspective(image):
    rows, cols, depth = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [389, 390]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250], [300, 300]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    newImage = cv.warpPerspective(image, M, (cols, rows))
    cv.imshow('real_affine_newImage', newImage)
    cv.waitKey(0)


if __name__ == '__main__':
    fresh_image = cv.imread('/home/abd/Pictures/Screenshot from 2020-10-07 08-14-27.png')
    cv.imshow('original-image', fresh_image)
    cv.waitKey(0)
    resize(fresh_image)
    translate(fresh_image)
    gray_fresh_image = cv.imread('/home/abd/Pictures/Screenshot from 2020-10-07 08-14-27.png', 0)
    rotate(gray_fresh_image)
    flip(fresh_image)
    affine(fresh_image)
    perspective(fresh_image)
