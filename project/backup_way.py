import cv2 as cv
import numpy as np
import time


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


def get_contours(src):
    image = src.copy()
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        return [x, y], [x + w, y + h]


def find_bigger_connected_object(src):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(src, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape, dtype=np.uint8)
    img2[output == max_label] = 255
    return img2


def find_bigger_connected_object_as_array(src):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(src, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape, dtype=np.uint8)
    img2[output == max_label] = 255
    return img2


def get_nearest_point(src, x, y):
    closest_point = src[0, 0]
    minDistance = 1000000
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i, j] == 0:
                continue
            dx = np.abs(x - j)
            dy = np.abs(y - i)
            distance = dx + dy
            if distance < minDistance:
                minDistance = distance
                closestPoint = [i, j]

    return closest_point


def get_corner_points(src, top_left, bottom_right):
    min_x = top_left[0]
    min_y = top_left[1]
    max_x = bottom_right[0]
    max_y = bottom_right[1]
    # return [topLeft,topRight,bottomLeft,bottomRight]
    return [
        get_nearest_point(src, min_x, min_y),
        get_nearest_point(src, max_x, min_y),
        get_nearest_point(src, min_x, max_y),
        get_nearest_point(src, max_x, max_y)
    ]


def pre_processing(src):
    start_time = time.time()
    gauss_blur = gaussian_blur(src)
    gauss_blur = gaussian_blur(gauss_blur, kernel_size=(3, 3), sigmaX=1)
    binary_image = adaptive_thresh(gauss_blur)
    inverse_binary_image = cv.bitwise_not(binary_image)
    closed_image = cross_closing(inverse_binary_image)
    dilated_image = cross_dilation(closed_image)
    bigger_connected_object = find_bigger_connected_object(dilated_image)
    # bigger_connected_object_wraped_with_contours = find_grid_with_contours(bigger_connected_object)
    top_left, bottom_right = get_contours(bigger_connected_object)

    corners = get_corner_points(bigger_connected_object, top_left, bottom_right)
    print(corners)

    print("FPS:{} MS".format((time.time() - start_time) * 1000))
    # cv.imshow('pre_processing_result', bigger_connected_object_wraped_with_contours)
    cv.imshow('pre_processing_result', bigger_connected_object)
    cv.waitKey(0)


if __name__ == '__main__':
    original_sudoku = cv.imread('resources/sudoku.jpg', 0)
    pre_processing(original_sudoku)
