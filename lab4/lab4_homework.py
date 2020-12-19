from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


def calc_hist(image):
    return cv.calcHist([image], [0], None, [256], [0, 256])


def gaussian_filter(image):
    gauss = cv.GaussianBlur(image, (5, 5), 2)
    return gauss


def otsu_thresh(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


def homework():
    image = cv.imread('resources/coins.jpg', 0)
    base_histogram = calc_hist(image)
    otsu_ret, otsu = otsu_thresh(image)
    gauss = gaussian_filter(image)
    gaussian_histogram = calc_hist(gauss)
    gauss_otsu_ret, gauss_otsu = otsu_thresh(gauss)

    titles = ['BASE', 'BASE_HISTOGRAM', 'BASE_OTSU', 'GAUSS', 'GAUSS_HISTOGRAM', 'GAUSS_OTSU']
    images = [image, base_histogram, otsu, gauss, gaussian_histogram, gauss_otsu]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        if i == 1 or i == 4:
            plt.plot(images[i], 'gray')  # plot used for histogram
        else:
            plt.imshow(images[i], 'gray')  # imshow used for images

    plt.show()


if __name__ == '__main__':
    homework()
