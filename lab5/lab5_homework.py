import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    img = cv2.imread('sample3.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minlinelength = 45
    maxlinegap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minlinelength, maxlinegap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Hough', img)
    cv2.waitKey(0)

    img = cv2.imread('sample3.jpg', 0)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.ifftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Forier Magnitude')
    plt.show()
