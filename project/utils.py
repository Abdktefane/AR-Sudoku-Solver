from keras.layers import BatchNormalization
from keras.layers import Dropout
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sudoku import Sudoku

hog_output_shape = 441
number_featuers = []
sift = cv.SIFT_create()
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


def check_match(target, model):
    pred = model.predict(target)
    print(pred)
    temp = pred[0]
    # print(temp)
    answer = np.argmax(temp, axis=0) + 1
    return answer


def preprocess_for_test_image(image, model):
    temp = np.uint8(image)
    temp = cv.resize(temp, (28, 28))
    _, temp = cv.threshold(temp, 150, 255, cv.THRESH_BINARY)
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
