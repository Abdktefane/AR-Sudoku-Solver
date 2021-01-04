from project.clean import utils as utils
from project.clean.performance import Performance
import cv2 as cv
import numpy as np

from project.clean.predictors.neural_network_predictor import NeuralNetworkPredictor


class Sudoku:
    def __init__(self, predictor):
        self.predictor = predictor
        self.logger = Performance()
        self.solver = None  # TODO(adel): add solver here

        self.source_image = None
        self.preprocessed_image = None
        self.perspective_array = None
        self.sudoku_board = None
        self.sudoku_cells_images = None
        self.sudoku_cells = None

    def pre_process_source_image(self):
        self.logger.tick()
        src = self.source_image.copy()
        gauss_blur = utils.gaussian_blur(src)
        self.preprocessed_image = utils.adaptive_thresh(gauss_blur)
        self.logger.end('pre_process source image')

    def get_sudoku_board(self):
        self.logger.tick()
        src = self.preprocessed_image.copy()
        max_contour = utils.find_biggest_contour(src)
        if max_contour is not None:
            cnt = cv.approxPolyDP(max_contour, 0.01 * cv.arcLength(max_contour, True), True)
            if len(cnt) == 4:
                self.sudoku_board, rect, self.perspective_array = utils.perspective_transformation(src, cnt)

            else:
                self.sudoku_board = None
        else:
            self.sudoku_board = None

        self.logger.end('get sudoku board')

    def get_sudoku_cells_images(self):
        self.logger.tick()
        self.sudoku_cells_images = utils.crop_image_to_cells(self.sudoku_board.copy())
        self.sudoku_cells = np.zeros((9, 9), dtype=np.uint8)
        self.logger.end('get sudoku cells images')

    def pre_process_cell_image(self, src):
        self.logger.tick()
        src = np.uint8(src)
        src = cv.resize(src, (utils.digit_pic_size, utils.digit_pic_size))
        _, src = cv.threshold(src, 125, 255, cv.THRESH_BINARY)
        # PS: each pre_processing before largest_connected_component must consider that front object is 1 (white)
        # and the background is 0 (black)
        contour = utils.find_biggest_contour(src)
        src = utils.largest_connected_component(src)
        if not utils.is_number(src):
            return np.ones((utils.digit_pic_size, utils.digit_pic_size), dtype=np.uint8)
        src = np.uint8(src)
        padding = 2
        x, y, w, h = cv.boundingRect(contour)
        cropped = src[y - padding:y + padding + h, x - padding:x + w + padding]
        cropped = cv.resize(cropped, (28, 28))

        # rect = cv.rectangle(src.copy(), (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 0, 0), 1)
        # cv.imshow('image with bounding box', rect)
        # cv.waitKey(0)

        # cv.imshow('cropped image', cropped)
        # cv.waitKey(0)

        # src = np.uint8(src)
        # src = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        self.logger.end('pre process cell image')
        return np.uint8(cropped)

    def predict(self, src):
        self.logger.tick()
        value = self.predictor.predict(src)
        answer = np.argmax(value[0], axis=0) + 1
        self.logger.end('model predict')
        return answer

    def pre_process_cell_image_and_predict(self, i, j):
        # self.logger.tick()
        # Criteria 1 for detecting white cell:
        # Has too little black pixels
        # cell = self.sudoku_cells_images[i]
        # clean_number = self.pre_process_cell_image(self.sudoku_cells_images[i][j].copy())
        # match = True
        # if clean_number.sum() >= utils.digit_pic_size ** 2 * 255 - utils.digit_pic_size * 1 * 255:
        #     match = False
        # else:
        #     # Criteria 2 for detecting white cell
        #     # Huge white area in the center
        #     center_width = clean_number.shape[1] // 2
        #     center_height = clean_number.shape[0] // 2
        #     x_start = center_height // 2
        #     x_end = center_height // 2 + center_height
        #     y_start = center_width // 2
        #     y_end = center_width // 2 + center_width
        #     center_region = clean_number[x_start:x_end, y_start:y_end]
        #     if center_region.sum() >= center_width * center_height * 255 - 255:
        #         match = False
        # if match:
        clean_number = self.pre_process_cell_image(self.sudoku_cells_images[i][j].copy())
        if utils.is_number(clean_number):
            self.sudoku_cells[i][j] = self.predict(clean_number)
        # self.logger.end('pre process cell image and predict')

    # TODO create 81 thread and each one edit self.sudoku_cells_images in (x,y)
    # TODO consider not create thread to 0 or empty cell for optimise
    def pre_process_cells_images(self):
        self.logger.tick()
        for i in range(9):
            for j in range(9):
                self.pre_process_cell_image_and_predict(i, j)

        self.logger.end('pre process cells images')

    def show_result_image(self):
        # cv.imshow('source image', self.source_image)
        # cv.imshow('preprocessed image', self.preprocessed_image)
        cv.imshow('sudoku board', self.sudoku_board)
        # cv.imshow('sudoku cells', self.sudoku_cells_images[0][0])
        cv.waitKey(0)

    def feed(self, image, show_result_image=False, show_sudoku_predict_result=True, real_board=None):
        self.source_image = image
        self.pre_process_source_image()
        self.get_sudoku_board()
        self.get_sudoku_cells_images()
        self.pre_process_cells_images()
        if show_sudoku_predict_result:
            utils.pretty_model_result(real_board, self.sudoku_cells)
        if show_result_image:
            self.show_result_image()


if __name__ == '__main__':
    board = [
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
    model = NeuralNetworkPredictor()
    sudoku = Sudoku(model)
    sudoku.feed(cv.imread('resources/sod6.jpg', 0), real_board=board)
