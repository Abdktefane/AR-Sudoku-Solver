import utils as utils
from performance import Performance
import cv2 as cv
import numpy as np
from sudoku import Sudoku as s

from predictors.neural_network_predictor import NeuralNetworkPredictor


class Sudoku:
    def __init__(self, predictor, debug=True, show_image_before_model_feed=False):
        self.solver = None  # s.Sudoku(3, 3, board=board)
        self.debug = debug
        self.logger = Performance(debug=self.debug)
        self.show_image_before_model_feed = show_image_before_model_feed
        self.predictor = predictor

        self.color_source_image = None
        self.source_image = None
        self.preprocessed_image = None
        self.perspective_array = None
        self.color_sudoku_board = None
        self.sudoku_board = None
        self.sudoku_cells_images = None
        self.sudoku_cells = None
        self.video_images = []

    def pre_process_source_image(self, camera):
        self.logger.tick('pre_process source image')
        src = self.source_image.copy()
        gauss_blur = utils.gaussian_blur(src, (13, 13))
        self.preprocessed_image = utils.adaptive_thresh(gauss_blur)
        # if camera:
        #     self.preprocessed_image = cv.dilate(
        #         self.preprocessed_image.copy(), (19, 19), iterations=4)
        self.logger.end('pre_process source image')

    def get_sudoku_board(self):
        self.logger.tick('get sudoku board')
        src = self.preprocessed_image.copy()
        max_contour = utils.find_biggest_contour(src)
        if max_contour is not None:
            temp = cv.drawContours(self.color_source_image, [
                                   max_contour], 0, (0, 255, 0), 3)
            cnt = cv.approxPolyDP(max_contour, 0.01 *
                                  cv.arcLength(max_contour, True), True)
            if len(cnt) == 4:
                self.sudoku_board, self.color_sudoku_board, rect, self.perspective_array = \
                    utils.perspective_transformation(
                        src, self.color_source_image, cnt)

            else:
                self.sudoku_board = None
        else:
            self.sudoku_board = None

        self.logger.end('get sudoku board')

    def get_sudoku_cells_images(self):
        if self.sudoku_board is None:
            return False
        self.sudoku_cells_images = utils.get_numbers_contours(
            self.sudoku_board.copy(), self.color_sudoku_board.copy())
        self.sudoku_cells = np.zeros((9, 9), dtype=np.uint8)
        self.logger.end('get sudoku cells images')
        return True

    def pre_process_cell_image(self, src):
        self.logger.tick('pre process cell image')
        src = cv.resize(src, (utils.digit_pic_size, utils.digit_pic_size))
        src = np.uint8(src)
        self.logger.end('pre process cell image')
        return None, None

    def predict(self, src):
        self.logger.tick('model predict')
        value = self.predictor.predict(
            src, show_image_before_model_feed=self.show_image_before_model_feed)
        answer = np.argmax(value[0], axis=0) + 1
        self.logger.end('model predict')
        return answer

    def pre_process_cell_image_and_predict(self, cell):
        self.logger.tick('pre process cell _ image and predict')
        self.sudoku_cells[cell.position[0]
                          ][cell.position[1]] = self.predict(cell.image)
        self.logger.end('pre process cell _ image and predict')

    def pre_process_cells_images(self):
        self.logger.tick('pre process cells images')
        for cell_image in self.sudoku_cells_images:
            self.pre_process_cell_image_and_predict(cell_image)
        self.logger.end('pre process cells images')

    def show_result_image(self):
        if self.sudoku_board is not None:
            cv.imshow('sudoku board', self.sudoku_board)

    def feed(self, image, show_result_image=False, real_board=None, camera=False):
        self.logger.tick('frame time', force_print=True)
        self.color_source_image = image
        self.source_image = cv.cvtColor(
            self.color_source_image, cv.COLOR_BGR2GRAY)
        self.pre_process_source_image(camera)
        self.get_sudoku_board()

        has_board = self.get_sudoku_cells_images()
        if has_board:
            self.pre_process_cells_images()
            self.solve()

        self.logger.end('frame time', force_print=True)
        if self.debug and self.sudoku_cells is not None:
            utils.pretty_model_result(real_board, self.sudoku_cells)
        if show_result_image:
            self.show_result_image()

    def solve(self):
        self.solver = s(width=3, height=3, board=self.sudoku_cells.tolist())
        image_with_solution = utils.write_solution_on_image(self.color_sudoku_board.copy(), self.solver.solve().board,
                                                            self.sudoku_cells)
        result_sudoku = cv.warpPerspective(image_with_solution, self.perspective_array,
                                           (self.source_image.shape[1], self.source_image.shape[0]), flags=cv.WARP_INVERSE_MAP)
        result = np.where(result_sudoku.sum(axis=-1, keepdims=True)
                          != 0, result_sudoku, self.color_source_image)
        cv.imshow('result', result)
        self.video_images.append(result)

    def save_video(self):
        video_writer = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(
            *'MP4V'), 10.0, (self.video_images[0].shape[1], self.video_images[0].shape[0]))
        for i in range(len(self.video_images)):
            video_writer.write(self.video_images[i])
        video_writer.release()


if __name__ == '__main__':
    model = NeuralNetworkPredictor()
    cap = cv.VideoCapture("resources/sudoku2.mp4")
    sudoku = Sudoku(model, debug=False, show_image_before_model_feed=False)

    while True:
        ret, frame = cap.read()  # Read the frame
        if ret:
            frame = cv.resize(frame, (1280 * 3 // 3, 720 * 4 // 3))
            sudoku.feed(frame, show_result_image=False, camera=True)
            if cv.waitKey(1) & 0xFF == ord('q'):  # Hit q if you want to stop the camera
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
    # sudoku.save_video()


# if __name__ == '__main__':
#     board = [
#         [5, 3, 0, 0, 7, 0, 0, 0, 0],
#         [6, 0, 0, 1, 9, 5, 0, 0, 0],
#         [0, 9, 8, 0, 0, 0, 0, 6, 0],
#         [8, 0, 0, 0, 6, 0, 0, 0, 3],
#         [4, 0, 0, 8, 0, 3, 0, 0, 1],
#         [7, 0, 0, 0, 2, 0, 0, 0, 6],
#         [0, 6, 0, 0, 0, 0, 2, 8, 0],
#         [0, 0, 0, 4, 1, 9, 0, 0, 5],
#         [0, 0, 0, 0, 8, 0, 0, 7, 9]
#     ]
#     model = NeuralNetworkPredictor()
#     sudoku = Sudoku(model, debug=True, show_image_before_model_feed=False)
#     sudoku.feed(cv.imread('resources/sod6.jpg'),
#                 real_board=board, show_result_image=False)
#     cv.waitKey(0)
