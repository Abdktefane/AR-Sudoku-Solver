import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

number_featuers = []
sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

winSize = (28, 28)
blockSize = (4, 4)
blockStride = (4, 4)
cellSize = (4, 4)
nbins = 9
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=441))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights('resources/weights.h5')


def check_match(target):
    pred = model.predict(target)
    # print(pred)
    temp = pred[0]
    temp = np.delete(temp, 0)
    # print(temp)
    answer = np.argmax(temp, axis=0) + 1
    return answer


def gaussian_blur(src, kernel_size=(5, 5), sigmaX=0):
    return cv.GaussianBlur(src, kernel_size, sigmaX)


def adaptive_thresh(src):
    # return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    # cv.THRESH_BINARY_INV,
    # 5, 2)
    return cv.adaptiveThreshold(src, 255, 1, 1, 11, 2)  # for threshold and inverse at once


def cross_dilation(src, size=3):
    se = cv.getStructuringElement(cv.MORPH_CROSS, (size, size))
    return cv.dilate(src, se)


def cross_closing(src):
    se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    return cv.morphologyEx(src, cv.MORPH_CLOSE, se)


def find_grid_with_contours(src, edited, start_time):
    image = edited.copy()
    print(time.time() * 1000)
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv.contourArea)
        cnt = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
        print(time.time() * 1000)
        rect = 0

        if len(cnt) == 4:
            image, rect, src, M = perspective_transformation(src, cnt, start_time)
            crop_image(image, src, M)
    return image, src


def coordinates_sum(corner):
    return corner[0] + corner[1]


def coordinates_division(corner):
    return corner[0] - corner[1]


def perspective_transformation(img, cnt, start_time):
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
    print("FPS:{} MS".format((time.time() - start_time) * 1000))

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

    return dst, rect, img, M


def remove_side_lines(img, ratio):
    """
        Remove black lines from image sides
    """
    while np.sum(img[0]) <= (1 - ratio) * img.shape[1] * 255:
        img = img[1:]
    # Bottom
    while np.sum(img[:, -1]) <= (1 - ratio) * img.shape[1] * 255:
        img = np.delete(img, -1, 1)
    # Left
    while np.sum(img[:, 0]) <= (1 - ratio) * img.shape[0] * 255:
        img = np.delete(img, 0, 1)
    # Right
    while np.sum(img[-1]) <= (1 - ratio) * img.shape[0] * 255:
        img = img[:-1]
    return img


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


def get_best_shift(img):
    # Calculate how to centralize the image using its center of mass
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv.warpAffine(img, M, (cols, rows))
    return shifted


def write_solution_on_image(image, grid, user_grid):
    # Write grid on image
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if (user_grid[i][j] != 0):  # If user fill this cell
                continue  # Move on
            text = str(user_grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = np.floor(width / 7)
            marginY = np.floor(height / 7)

            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = int(width * j + np.floor((width - text_width) / 2) + off_set_x)
            bottom_left_corner_y = int(height * (i + 1) - np.floor((height - text_height) / 2) + off_set_y)
            image = cv.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y),
                               font, font_scale, (0, 255, 0), thickness=3, lineType=cv.LINE_AA)
    return image


def crop_image(img, src, M):
    original = img.copy()

    img = cv.resize(img, (500, 500))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = adaptive_thresh(img)  # needed to find largest connected component

    cv.imshow("img", img)

    blocks = []
    userGrid = np.zeros((9, 9), np.uint8)
    h = img.shape[0] // 9
    w = img.shape[1] // 9
    offset_w = np.math.floor(w / 10)  # Offset is used to get rid of the boundaries
    offset_h = np.math.floor(h / 10)
    for i in range(9):
        for j in range(9):
            userGrid[i][j] = 1
            match = True
            n = i * 9 + j
            blocks.append(img[h * i + offset_h:h * (i + 1) - offset_h, w * j + offset_w:w * (j + 1) - offset_w])

            if i == 0 and j == 1:
                cv.imshow("number1", blocks[n])
            #blocks[n] = cv.bitwise_not(blocks[n])
            blocks[n] = largest_connected_component(blocks[n])
            if i == 0 and j == 1:
                cv.imshow("number2", blocks[n])
            # Resize
            digit_pic_size = 28

            blocks[n] = cv.resize(blocks[n], (digit_pic_size, digit_pic_size))
            _, blocks[n] = cv.threshold(blocks[n], 200, 255, cv.THRESH_BINARY)

            # Criteria 1 for detecting white cell:
            # Has too little black pixels
            if blocks[n].sum() >= digit_pic_size ** 2 * 255 - digit_pic_size * 1 * 255:
                blocks[n] = np.zeros((digit_pic_size, digit_pic_size))
                userGrid[i][j] = 0
                match = False
            # Criteria 2 for detecting white cell
            # Huge white area in the center
            center_width = blocks[n].shape[1] // 2
            center_height = blocks[n].shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = blocks[n][x_start:x_end, y_start:y_end]
            if center_region.sum() >= center_width * center_height * 255 - 255:
                blocks[n] = np.zeros((digit_pic_size, digit_pic_size))
                userGrid[i][j] = 0
                match = False
            # Centralize the image according to center of mass / BUT Not working properly
            # blocks[n] = cv.bitwise_not(blocks[n])
            # shift_x, shift_y = get_best_shift(blocks[n])
            # blocks[n] = shift(blocks[n], shift_x, shift_y)
            # blocks[n] = cv.bitwise_not(blocks[n])
            # blocks[n] = cv.resize(blocks[n], (50, 50))

            if match:
                temp = np.uint8(blocks[n])
                # histogram = hog.compute(temp, None, None)
                # histogram = np.asarray(histogram)
                # histogram = histogram.reshape((-1, 441))
                # id = check_match(histogram)
                #print(i, j, id)
    test = np.uint8(blocks[1 * 9 + 4])
    cv.imshow("2, 3", test)
    test = np.uint8(blocks[1 * 9 + 5])
    cv.imshow("2, 2", test)

    # for now zeros are written in the empty spots
    image_with_solution = write_solution_on_image(original, userGrid, userGrid)
    result_sudoku = cv.warpPerspective(image_with_solution, M, (src.shape[1], src.shape[0])
                                       , flags=cv.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, src)
    cv.imshow("res", result)


def pre_processing(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    start_time = time.time()
    gauss_blur = gaussian_blur(gray)
    inverse_binary_image = adaptive_thresh(gauss_blur)
    grid, src = find_grid_with_contours(src, inverse_binary_image, start_time)
    print("FPS:{} MS".format((time.time() - start_time) * 1000))
    cv.imshow('base', grid)
    return src


if __name__ == '__main__':
    original_sudoku = cv.imread('resources/sod6.jpg')
    pre_processing(original_sudoku)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
    # cap = cv.VideoCapture(0)
    cap = cv.VideoCapture("resources/sudoku2.mp4")
    cap.set(3, 1280)  # HD Camera
    cap.set(4, 720)
    old_sudoku = None

    while (True):
        ret, frame = cap.read()  # Read the frame
        if ret == True:
            # frame = pre_processing(frame)
            # cv.imshow("lol", frame)
            # sudoku_frame = RealTimeSudokuSolver.recognize_and_solve_sudoku(frame, model, old_sudoku)
            # showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)  # Print the 'solved' image
            if cv.waitKey(1) & 0xFF == ord('q'):  # Hit q if you want to stop the camera
                break
        else:
            break

    cap.release()
    # out.release()
    cv.destroyAllWindows()
