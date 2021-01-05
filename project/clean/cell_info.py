import numpy as np
import cv2 as cv

from project.clean import utils


class CellInfo:
    def __init__(self, source_image, contour_pos, board_size):
        self.source_image = source_image
        self.cell_coord = contour_pos
        self.position = self.calc_position(board_size)
        self.image = cv.bitwise_not(cv.resize(self.source_image, (utils.digit_pic_size, utils.digit_pic_size)))
        self.value = None

    def calc_position(self, size):
        width = size[1] / 9
        height = size[0] / 9
        for i in range(9):
            if height * i < self.cell_coord[1] < height * (i+1):
                for j in range(9):
                    if width * j < self.cell_coord[0] < width * (j+1):
                        return [i, j]
        return None, None
