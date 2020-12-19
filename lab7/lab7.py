import cv2 as cv
import numpy as np
import argparse


def k_means():
    image = cv.imread('7-session/Home2.jpg')
    z = image.reshape((-1, 3))
    z = np.float32(z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 6

    ret, label, center = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)

    cv.imshow('image', image)
    cv.imshow('res2', res2)
    cv.waitKey(0)


def subtract():
    # back_sub = cv.createBackgroundSubtractorMOG2()
    # back_sub = cv.bgsegm.createBackgroundSubtractorMOG()
    back_sub = cv.bgsegm.createBackgroundSubtractorGMG()

    capture = cv.VideoCapture('7-session/768x576.avi')
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        fgMask = back_sub.apply(frame)
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def motion():
    cap = cv.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:
        ret, frame = cap.read()
        d = cv.absdiff(frame1, frame2)
        grey = cv.cvtColor(d, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(grey, (5, 5), 0)
        ret, th = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(th, np.ones((7, 7), np.uint8), iterations=9)
        c, h = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame1, c, -1, 255, 3)
        cv.imshow("inter", frame1)
        if cv.waitKey(40) == 27:
            break
        frame1 = frame2
        ret, frame2 = cap.read()
    cv.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    # k_means()
    # subtract()
    motion()
