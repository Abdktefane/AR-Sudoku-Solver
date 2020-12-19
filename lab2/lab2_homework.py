import cv2 as cv
import numpy as np


def show_image_and_wait(image):
    cv.imshow('base_image', image)
    cv.waitKey(0)


def translate(image):
    rows, cols, depth = image.shape
    translate_matrix = np.float32(
        [[1, 0, 40],
         [0, 1, -80]]
    )
    translated_image = cv.warpAffine(image, translate_matrix, (cols, rows))
    show_image_and_wait(translated_image)


def rotate(image, angel, scale):
    rows, cols, depth = image.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angel, scale)
    rotated_image = cv.warpAffine(image, M, (rows, cols))
    show_image_and_wait(rotated_image)


def affine(image, pts1, pts2):
    rows, cols, depth = image.shape
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M = cv.getAffineTransform(pts1, pts2)
    affine_image = cv.warpAffine(image, M, (cols, rows))
    show_image_and_wait(affine_image)


def perspective(image, pts1, pts2):
    rows, cols, depth = image.shape
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M = cv.getPerspectiveTransform(pts1, pts2)
    perspective_image = cv.warpPerspective(image, M, (cols, rows))
    show_image_and_wait(perspective_image)


def on_click_events(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        affine_pixel_location.append([x, y])
        if len(affine_pixel_location) == 6:
            affine(input_image, affine_pixel_location[:3], affine_pixel_location[3:])
            affine_pixel_location.clear()
            show_image_and_wait(input_image)

    if event == cv.EVENT_RBUTTONDOWN:
        perspective_pixel_location.append([x, y])
        if len(perspective_pixel_location) == 8:
            perspective(input_image, perspective_pixel_location[:4], perspective_pixel_location[4:])
            perspective_pixel_location.clear()
            show_image_and_wait(input_image)


affine_pixel_location = []
perspective_pixel_location = []
input_image = cv.imread('kotlin.png')
if __name__ == '__main__':
    cv.namedWindow('base_image', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('base_image', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    show_image_and_wait(input_image)

    translate(input_image)

    rotate(input_image, 25, 1)

    rotate(input_image, -35, 0.5)

    cv.setMouseCallback('base_image', on_click_events)
    show_image_and_wait(input_image)
