import cv2 as cv
import numpy as np


def gaussian_blur(src, kernel_size=(7, 7), sigmaX=0):
    return cv.GaussianBlur(src, kernel_size, sigmaX)


def adaptive_thresh(src):
    return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                5, 2)


def cross_dilation(src):
    se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    return cv.dilate(src, se)


def draw_lines(src):
    edges = cv.Canny(src, 50, 150)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return src


def cross_closing(src):
    se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    return cv.morphologyEx(src, cv.MORPH_CLOSE, se)


def find_grid_with_floid(src):
    # Using flood filling to find the biggest blob in the picture
    outerbox = src
    maxi = -1
    maxpt = None
    value = 10
    height, width = np.shape(outerbox)
    for y in range(height):
        row = src[y]
        for x in range(width):
            if row[x] >= 128:
                area = cv.floodFill(outerbox, None, (x, y), 64)[0]
                if value > 0:
                    # cv2.imwrite("StagesImages/5.jpg", outerbox)
                    value -= 1
                if area > maxi:
                    maxpt = (x, y)
                    maxi = area

    # Floodfill the biggest blob with white (Our sudoku board's outer grid)
    cv.floodFill(outerbox, None, maxpt, (255, 255, 255))

    # Floodfill the other blobs with black
    for y in range(height):
        row = src[y]
        for x in range(width):
            if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
                cv.floodFill(outerbox, None, (x, y), 0)

    # Eroding it a bit to restore the image
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    outerbox = cv.erode(outerbox, kernel)
    return outerbox


def find_grid_with_contours(src):
    image = src.copy()
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        # the contours are drawn here
        # cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest area of the contour
        c = max(contours, key=cv.contourArea)

        x, y, w, h = cv.boundingRect(c)
        # draw the 'human' contour (in green)
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 40, 40), 2)
    return image


def pre_processing(src):
    gauss_blur = gaussian_blur(src)
    gauss_blur = gaussian_blur(gauss_blur, kernel_size=(3, 3), sigmaX=1)
    binary_image = adaptive_thresh(gauss_blur)
    inverse_binary_image = cv.bitwise_not(binary_image)
    closed_image = cross_closing(inverse_binary_image)
    dilated_image = cross_dilation(closed_image)
    # with_line_image = draw_lines(dilated_image)
    grid = find_grid_with_floid(dilated_image)

    cv.imshow('base', grid)
    cv.waitKey(0)


if __name__ == '__main__':
    original_sudoku = cv.imread('resources/sudoku.jpg', 0)
    pre_processing(original_sudoku)
