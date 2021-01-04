import cv2 as cv
import numpy as np

digit_pic_size = 28


def gaussian_blur(src, kernel_size=(5, 5), sigmaX=0):
    return cv.GaussianBlur(src, kernel_size, sigmaX)


def cross_closing(src):
    se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    return cv.morphologyEx(src, cv.MORPH_CLOSE, se)


def adaptive_thresh(src):
    # return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    # cv.THRESH_BINARY_INV,
    # 5, 2)
    return cv.adaptiveThreshold(src, 255, 1, 1, 11, 2)  # for threshold and inverse at once


def find_contours(src):
    return cv.findContours(src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


def find_biggest_contour(src):
    contours, hierarchy = find_contours(src)
    if len(contours) != 0:
        return max(contours, key=cv.contourArea)
    pass


def coordinates_sum(corner):
    return corner[0] + corner[1]


def coordinates_division(corner):
    return corner[0] - corner[1]


def perspective_transformation(img, cnt):
    corners = np.zeros((4, 2), dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    for i in range(4):
        corners[i] = cnt[i][0]

    # Top left
    rect[0] = min(corners, key=coordinates_sum)
    # Bottom right
    rect[2] = max(corners, key=coordinates_sum)
    # Top right
    rect[1] = max(corners, key=coordinates_division)
    # Bottom left
    rect[3] = min(corners, key=coordinates_division)

    (tl, tr, br, bl) = rect
    # the actual width of our Sudoku board
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # the actual height of our Sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))
    pts1 = np.float32([rect[0], rect[1], rect[2], rect[3]])
    pts2 = np.float32(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]])  # TL -> TR -> BR -> BL
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (max_width, max_height))
    return dst, rect, M


def crop_image_to_cells(img):
    h = img.shape[0] // 9
    w = img.shape[1] // 9
    offset_w = np.math.floor(w / 10)  # Offset is used to get rid of the boundaries
    offset_h = np.math.floor(h / 10)
    blocks = np.zeros(
        (9, 9, h - (offset_h + offset_w), w - (offset_h + offset_w))
    )
    for i in range(9):
        for j in range(9):
            # n = i * 9 + j
            blocks[i][j] = img[h * i + offset_h:h * (i + 1) - offset_h, w * j + offset_w:w * (j + 1) - offset_w]
    return blocks


def is_number(number):
    match = True
    if number.sum() >= digit_pic_size ** 2 * 255 - digit_pic_size * 1 * 255:
        match = False
    else:
        # Criteria 2 for detecting white cell
        # Huge white area in the center
        center_width = number.shape[1] // 2
        center_height = number.shape[0] // 2
        x_start = center_height // 2
        x_end = center_height // 2 + center_height
        y_start = center_width // 2
        y_end = center_width // 2 + center_width
        center_region = number[x_start:x_end, y_start:y_end]
        if center_region.sum() >= center_width * center_height * 255 - 255:
            match = False
    return match


def largest_connected_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    if len(sizes) <= 1:
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2


def pretty_model_result(real_board, predicted_board):
    right = 0
    wrong = 0
    print('<------------------->')
    for i in range(9):
        for j in range(9):
            score = 'Wrong'
            if real_board[i][j] == 0:
                continue
            if real_board[i][j] == predicted_board[i][j]:
                right += 1
                score = 'Right'
            else:
                wrong += 1
            print("{} predict for {} position({},{})".format(score, real_board[i][j], i + 1, j + 1))
    print('<------------------->')
    print("{} Right predict and {} Wrong predict".format(right, wrong))
    print('<------------------->\n')
