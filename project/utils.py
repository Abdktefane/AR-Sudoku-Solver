from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import pytesseract

hog_output_shape = 441
number_featuers = []
# sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
winSize = (28, 28)
blockSize = (4, 4)
blockStride = (4, 4)
cellSize = (4, 4)
nbins = 9
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)


# hog_output_shape = 15876

# def create_model():
#     model = Sequential()
#     model.add(Dense(64, activation='relu', input_dim=hog_output_shape))
#     model.add(Dense(64, activation='relu', input_dim=hog_output_shape))
#     model.add(Dense(9, activation='softmax'))
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#
#     model.load_weights('resources/weights2.h5')
#     return model

def create_model():
    model = Sequential()
    model.add(Dense(364, activation='relu', input_dim=hog_output_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(52, activation='relu', input_dim=hog_output_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.load_weights('resources/weights.h5')
    return model


sud_numbers = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

wrong = 0
right = 0


def pretty_model_result(result, answer, i, j, print_details=True):
    listOfGlobals = globals()
    print('<------------------->')
    result = np.round(result[0], 2)
    score = 'Wrong'
    if sud_numbers[i][j] == answer:
        listOfGlobals['right'] += 1
        score = 'Right'
    else:
        listOfGlobals['wrong'] += 1

    print('%s predict for %d, position(%d,%d):' % (score, sud_numbers[i][j], i + 1, j + 1))
    if print_details:
        print('details result:')
        for i in range(len(result)):
            print("%.2f  for number %2d" % (result[i] * 100, i + 1))
    print('<------------------->\n')


# pred = pytesseract.image_to_string(target,lang='eng')
# arr = pred.split('\n')[0:-1]
# result = '\n'.join(arr)
# print(result)
def check_match(target, model, i, j):
    pred = model.predict(target, i, j)
    answer = np.argmax(pred[0], axis=0) + 1
    pretty_model_result(pred, answer, i, j)
    return answer


def adaptive_thresh(src):
    # return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    # cv.THRESH_BINARY_INV,
    # 5, 2)
    return cv.adaptiveThreshold(src, 255, 1, 1, 11, 2)  # for threshold and inverse at once


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


digit_pic_size = 28


def preprocess_for_number(src):
    src = np.uint8(src)
    src = cv.resize(src, (digit_pic_size, digit_pic_size))
    _, src = cv.threshold(src, 150, 255, cv.THRESH_BINARY)
    # src = cv.GaussianBlur(src, (7, 7), 0)
    # src = adaptive_thresh(src)
    # src = largest_connected_component(src)
    # kernal = np.ones((1, 1), np.uint8)
    # src = cv.dilate(src, kernal, 1)
    # src = np.uint8(src)
    return src


row_index_test = 8

col_index_test = 4


def preprocess_for_test_image(image, model):
    # temp = np.uint8(image)
    # temp = cv.resize(temp, (28, 28))
    # _, temp = cv.threshold(temp, 150, 255, cv.THRESH_BINARY)
    temp = preprocess_for_number(image)
    h = np.asarray(hog.compute(temp, None, None)).reshape((-1, hog_output_shape))
    pred = model.predict(h)
    temp = pred[0]
    # print(temp)
    answer = np.argmax(temp, axis=0) + 1
    print(answer)
    return h


def prepare_numbers_features(count, model):
    number_featuers.append(cv.imread("resources/Numbers1/1.jpg", 0))
    [preprocess_for_test_image(cv.imread("resources/Numbers1/" + str(i + 1) + ".jpg", 0), model) for i in range(count)]
