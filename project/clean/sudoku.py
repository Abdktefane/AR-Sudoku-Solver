from project.clean import utils as utils
from project.clean.performance import Performance
import cv2 as cv
import numpy as np
import sudoku as s

from project.clean.predictors.neural_network_predictor import NeuralNetworkPredictor


class Sudoku:
    def __init__(self, predictor, debug=True, show_image_before_model_feed=False):
        self.solver = None  # s.Sudoku(3, 3, board=board)
        self.debug = debug
        self.logger = Performance(debug=self.debug)
        self.show_image_before_model_feed = show_image_before_model_feed
        self.predictor = predictor

        self.source_image = None
        self.preprocessed_image = None
        self.perspective_array = None
        self.sudoku_board = None
        self.sudoku_cells_images = None
        self.sudoku_cells = None

    def pre_process_source_image(self):
        self.logger.tick('pre_process source image')
        src = self.source_image.copy()
        gauss_blur = utils.gaussian_blur(src)
        self.preprocessed_image = utils.adaptive_thresh(gauss_blur)
        self.logger.end('pre_process source image')

    def get_sudoku_board(self):
        self.logger.tick('get sudoku board')
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
        self.logger.tick('get sudoku cells images')
        self.sudoku_cells_images = utils.crop_image_to_cells(self.sudoku_board.copy())
        self.sudoku_cells = np.zeros((9, 9), dtype=np.uint8)
        self.logger.end('get sudoku cells images')

    def pre_process_cell_image(self, src):
        self.logger.tick('pre process cell image')
        src = cv.resize(src, (utils.digit_pic_size, utils.digit_pic_size))
        src = np.uint8(src)
        # _, src = cv.threshold(src, 100, 255, cv.THRESH_BINARY)
        # src = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 2)
        # for now otsu is the best trade off between speed and accuracy
        _, src = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # PS: each pre_processing before largest_connected_component must consider that front object is 1 (white)
        # and the background is 0 (black)
        contour = utils.find_biggest_contour(src)
        src = utils.largest_connected_component(src)
        is_number = utils.is_number(src)
        if not is_number:
            return np.ones((utils.digit_pic_size, utils.digit_pic_size), dtype=np.uint8), is_number
        src = np.uint8(src)
        padding = 2
        x, y, w, h = cv.boundingRect(contour)
        cropped = src[y - padding:y + padding + h, x - padding:x + w + padding]
        cropped = cv.resize(cropped, (28, 28))
        self.logger.end('pre process cell image')
        return cropped, is_number,

    def predict(self, src):
        self.logger.tick('model predict')
        value = self.predictor.predict(src, show_image_before_model_feed=self.show_image_before_model_feed)
        answer = np.argmax(value[0], axis=0) + 1
        self.logger.end('model predict')
        return answer

    def pre_process_cell_image_and_predict(self, i, j):
        self.logger.tick('pre process cell image and predict')
        clean_number, is_number = self.pre_process_cell_image(self.sudoku_cells_images[i][j].copy())
        if is_number:
            self.sudoku_cells[i][j] = self.predict(clean_number)
        self.logger.end('pre process cell image and predict')

    # TODO create 81 thread and each one edit self.sudoku_cells_images in (x,y)
    # TODO consider not create thread to 0 or empty cell for optimise
    def pre_process_cells_images(self):
        self.logger.tick('pre process cells images')
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

    def feed(self, image, show_result_image=False, real_board=None):
        self.logger.tick('frame time', force_print=True)
        self.source_image = image
        self.pre_process_source_image()
        self.get_sudoku_board()
        self.get_sudoku_cells_images()
        self.pre_process_cells_images()
        self.logger.end('frame time', force_print=True)
        if self.debug:
            utils.pretty_model_result(real_board, self.sudoku_cells)
        if show_result_image:
            self.show_result_image()

    def solve(self):
        self.solver = s.Sudoku(3, 3, board=self.sudoku_cells)
        # image_with_solution = write_solution_on_image(original, solution, board)
        # result_sudoku = cv.warpPerspective(image_with_solution, M, (src.shape[1], src.shape[0])
        #                                    , flags=cv.WARP_INVERSE_MAP)
        # result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, src)


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
    sudoku = Sudoku(model, debug=True, show_image_before_model_feed=False)
    sudoku.feed(cv.imread('resources/sod6.jpg', 0), real_board=board)

# if __name__ == '__main__':
#     board = [
#         [4, 8, 3, 7, 2, 6, 1, 5, 9],
#         [7, 2, 6, 1, 5, 9, 4, 8, 3],
#         [1, 5, 9, 4, 8, 3, 7, 2, 6],
#         [8, 3, 7, 2, 6, 1, 5, 9, 4],
#         [2, 6, 1, 5, 9, 4, 8, 3, 7],
#         [5, 9, 4, 8, 3, 7, 2, 6, 1],
#         [3, 7, 2, 6, 1, 5, 9, 4, 8],
#         [6, 1, 5, 9, 4, 8, 3, 7, 2],
#         [9, 4, 8, 3, 7, 2, 6, 1, 5]
#     ]
#     model = NeuralNetworkPredictor()
#     sudoku = Sudoku(model, debug=True, show_image_before_model_feed=False)
#     sudoku.feed(cv.imread('resources/sod4.jpg', 0), real_board=board)
