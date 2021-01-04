#
# import cv2 as cv
# import numpy as np
# import time
#
#
# def gaussian_blur(src, kernel_size=(7, 7), sigmaX=0):
#     return cv.GaussianBlur(src, kernel_size, sigmaX)
#
#
# def adaptive_thresh(src):
#     return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
#                                 5, 2)
#
#
# def cross_dilation(src):
#     se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
#     return cv.dilate(src, se)
#
#
# def draw_lines(src):
#     edges = cv.Canny(src, 50, 150)
#     lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
#     for line in lines:
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return src
#
#
# def cross_closing(src):
#     se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
#     return cv.morphologyEx(src, cv.MORPH_CLOSE, se)
#
#
# def find_grid_with_floid(src):
#     # Using flood filling to find the biggest blob in the picture
#     outerbox = src
#     maxi = -1
#     maxpt = None
#     value = 10
#     height, width = np.shape(outerbox)
#     for y in range(height):
#         row = src[y]
#         for x in range(width):
#             if row[x] >= 128:
#                 area = cv.floodFill(outerbox, None, (x, y), 64)[0]
#                 if value > 0:
#                     # cv2.imwrite("StagesImages/5.jpg", outerbox)
#                     value -= 1
#                 if area > maxi:
#                     maxpt = (x, y)
#                     maxi = area
#
#     # Floodfill the biggest blob with white (Our sudoku board's outer grid)
#     cv.floodFill(outerbox, None, maxpt, (255, 255, 255))
#
#     # Floodfill the other blobs with black
#     for y in range(height):
#         row = src[y]
#         for x in range(width):
#             if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
#                 cv.floodFill(outerbox, None, (x, y), 0)
#
#     # Eroding it a bit to restore the image
#     kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
#     outerbox = cv.erode(outerbox, kernel)
#     return outerbox
#
#
# def find_grid_with_contours(src):
#     image = src.copy()
#     contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     if len(contours) != 0:
#         # the contours are drawn here
#         # cv2.drawContours(output, contours, -1, 255, 3)
#
#         # find the biggest area of the contour
#         c = max(contours, key=cv.contourArea)
#
#         x, y, w, h = cv.boundingRect(c)
#         # draw the 'human' contour (in green)
#         cv.rectangle(image, (x, y), (x + w, y + h), (255, 40, 40), 2)
#
#     return image
#
#
# def get_contours(src):
#     image = src.copy()
#     contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     if len(contours) != 0:
#         c = max(contours, key=cv.contourArea)
#         x, y, w, h = cv.boundingRect(c)
#         return [x, y], [x + w, y + h]
#
#
# def find_bigger_connected_object(src):
#     nb_components, output, stats, centroids = cv.connectedComponentsWithStats(src, connectivity=4)
#     sizes = stats[:, -1]
#
#     max_label = 1
#     max_size = sizes[1]
#     for i in range(2, nb_components):
#         if sizes[i] > max_size:
#             max_label = i
#             max_size = sizes[i]
#
#     img2 = np.zeros(output.shape, dtype=np.uint8)
#     img2[output == max_label] = 255
#     return img2
#
#
# def find_bigger_connected_object_as_array(src):
#     nb_components, output, stats, centroids = cv.connectedComponentsWithStats(src, connectivity=4)
#     sizes = stats[:, -1]
#
#     max_label = 1
#     max_size = sizes[1]
#     for i in range(2, nb_components):
#         if sizes[i] > max_size:
#             max_label = i
#             max_size = sizes[i]
#
#     img2 = np.zeros(output.shape, dtype=np.uint8)
#     img2[output == max_label] = 255
#     return img2
#
#
# def get_nearest_point(src, x, y):
#     closest_point = src[0, 0]
#     minDistance = 1000000
#     for i in range(src.shape[0]):
#         for j in range(src.shape[1]):
#             if src[i, j] == 0:
#                 continue
#             dx = np.abs(x - j)
#             dy = np.abs(y - i)
#             distance = dx + dy
#             if distance < minDistance:
#                 minDistance = distance
#                 closestPoint = [i, j]
#
#     return closest_point
#
#
# def get_corner_points(src, top_left, bottom_right):
#     min_x = top_left[0]
#     min_y = top_left[1]
#     max_x = bottom_right[0]
#     max_y = bottom_right[1]
#     # return [topLeft,topRight,bottomLeft,bottomRight]
#     return [
#         get_nearest_point(src, min_x, min_y),
#         get_nearest_point(src, max_x, min_y),
#         get_nearest_point(src, min_x, max_y),
#         get_nearest_point(src, max_x, max_y)
#     ]
#
#
# def pre_processing(src):
#     start_time = time.time()
#     gauss_blur = gaussian_blur(src)
#     gauss_blur = gaussian_blur(gauss_blur, kernel_size=(3, 3), sigmaX=1)
#     binary_image = adaptive_thresh(gauss_blur)
#     inverse_binary_image = cv.bitwise_not(binary_image)
#     closed_image = cross_closing(inverse_binary_image)
#     dilated_image = cross_dilation(closed_image)
#     bigger_connected_object = find_bigger_connected_object(dilated_image)
#     # bigger_connected_object_wraped_with_contours = find_grid_with_contours(bigger_connected_object)
#     top_left, bottom_right = get_contours(bigger_connected_object)
#
#     corners = get_corner_points(bigger_connected_object, top_left, bottom_right)
#     print(corners)
#
#     print("FPS:{} MS".format((time.time() - start_time) * 1000))
#     # cv.imshow('pre_processing_result', bigger_connected_object_wraped_with_contours)
#     cv.imshow('pre_processing_result', bigger_connected_object)
#     cv.waitKey(0)
#
#
# if __name__ == '__main__':
#     original_sudoku = cv.imread('resources/sudoku.jpg', 0)
#     pre_processing(original_sudoku)
#
#









# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import time
#
# #start_time = 0
#
# number_featuers = []
# sift = cv.SIFT_create()
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#
# winSize = (28, 28)
# blockSize = (4, 4)
# blockStride = (4, 4)
# cellSize = (4, 4)
# nbins = 9
# hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
#
# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=441))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# model.load_weights('resources/weights2.h5')
#
#
# def check_match(target):
#     pred = model.predict(target)
#     # print(pred)
#     temp = pred[0]
#     temp = np.delete(temp, 0)
#     # print(temp)
#     answer = np.argmax(temp, axis=0) + 1
#     return answer
#
#
# def prepare_numbers_features(count):
#     number_featuers.append(cv.imread("resources/Numbers/1.jpg", 0))
#     for i in range(1, count + 1):
#         temp = cv.imread("resources/Numbers2/" + str(i) + ".jpg", 0)
#         temp = np.uint8(temp)
#         temp = cv.resize(temp, (28, 28))
#         _, temp = cv.threshold(temp, 150, 255, cv.THRESH_BINARY)
#         h = hog.compute(temp, None, None)
#         h = np.asarray(h)
#         h = h.reshape((-1, 441))
#         pred = model.predict(h)
#         temp = pred[0]
#         temp = np.delete(temp, 0)
#         # print(temp)
#         answer = np.argmax(temp, axis=0) + 1
#         print(answer)
#
#
# def gaussian_blur(src, kernel_size=(5, 5), sigmaX=0):
#     return cv.GaussianBlur(src, kernel_size, sigmaX)
#
#
# def adaptive_thresh(src):
#     # return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#     # cv.THRESH_BINARY_INV,
#     # 5, 2)
#     return cv.adaptiveThreshold(src, 255, 1, 1, 11, 2)  # for threshold and inverse at once
#
#
# def adaptive_thresh2(src):
#     return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                 cv.THRESH_BINARY_INV,
#                                 5, 2)
#
#
# def cross_dilation(src, size=3):
#     se = cv.getStructuringElement(cv.MORPH_CROSS, (size, size))
#     return cv.dilate(src, se)
#
#
# def draw_lines(src):
#     edges = cv.Canny(src, 50, 150)
#     lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
#     for line in lines:
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return src
#
#
# def cross_closing(src):
#     se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
#     return cv.morphologyEx(src, cv.MORPH_CLOSE, se)
#
#
# def find_grid_with_floid(src):
#     # Using flood filling to find the biggest blob in the picture
#     outerbox = src
#     maxi = -1
#     maxpt = None
#     value = 10
#     height, width = np.shape(outerbox)
#     for y in range(height):
#         row = src[y]
#         for x in range(width):
#             if row[x] >= 128:
#                 area = cv.floodFill(outerbox, None, (x, y), 64)[0]
#                 if value > 0:
#                     # cv2.imwrite("StagesImages/5.jpg", outerbox)
#                     value -= 1
#                 if area > maxi:
#                     maxpt = (x, y)
#                     maxi = area
#
#     # Floodfill the biggest blob with white (Our sudoku board's outer grid)
#     cv.floodFill(outerbox, None, maxpt, (255, 255, 255))
#
#     # Floodfill the other blobs with black
#     for y in range(height):
#         row = src[y]
#         for x in range(width):
#             if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
#                 cv.floodFill(outerbox, None, (x, y), 0)
#
#     # Eroding it a bit to restore the image
#     kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
#     outerbox = cv.erode(outerbox, kernel)
#     return outerbox
#
#
# def find_grid_with_contours(src, edited, start_time):
#     image = edited.copy()
#     print(time.time()*1000)
#     contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     if len(contours) != 0:
#         # the contours are drawn here
#         # cv2.drawContours(output, contours, -1, 255, 3)
#         # Get 4 corners of the biggest contour
#         # j = 0
#         # contours = sorted(contours, key=lambda x: cv.contourArea(x))
#         # print(len(contours))
#         bom = src.copy()
#         # bom2 = src.copy()
#         max_area = 0
#         # biggest_contour = None
#         for c in contours:
#             area = cv.contourArea(c)
#             if area > max_area:
#                 max_area = area
#                 biggest_contour = c
#                 bom = cv.drawContours(bom, [c], 0, (0, 255, 0), 3)
#                 cv.imshow("cont", bom)
#
#         # cv.imshow("cont", bom)
#         # find the biggest area of the contour
#         c = max(contours, key=cv.contourArea)
#         # cnt = get_corners_from_contours(c, 4)
#         # bom2 = cv.drawContours(bom2, [c], 0, (0, 255, 0), 3)
#         # cv.imshow("cont2", bom2)
#         # x, y, w, h = cv.boundingRect(c)
#         # draw the 'human' contour (in green)
#         # cv.rectangle(image, (x, y), (x + w, y + h), (255, 40, 40), 2)
#
#         cnt = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
#         print(time.time() * 1000)
#         # hull = cv.convexHull(cnt, returnPoints=False)
#         # defects = cv.convexityDefects(cnt, hull)
#         # print(cnt)
#         rect = 0
#
#         if len(cnt) == 4:
#             image, rect, src, M = prespective_transformation(src, cnt, start_time)
#             crop_image(image, src, M)
#     return image, src
#
#
# def cordinats_sum(corner):
#     return corner[0] + corner[1]
#
#
# def prespective_transformation(img, cnt, start_time):
#     corners = np.zeros((4, 2), dtype="float32")
#     rect = np.zeros((4, 2), dtype="float32")
#     for i in range(4):
#         corners[i] = cnt[i][0]
#     # print(corners)
#
#     # Top left
#     rect[0] = min(corners, key=cordinats_sum)
#     for i in range(4):
#         if np.array_equal(corners[i], rect[0]):
#             corners = np.delete(corners, i, 0)
#             break
#     # Bottom right
#     rect[2] = max(corners, key=cordinats_sum)
#     for i in range(3):
#         if np.array_equal(corners[i], rect[2]):
#             corners = np.delete(corners, i, 0)
#             break
#     # The left two corners
#     if corners[0][0] > corners[1][0]:
#         rect[3] = corners[1]
#         rect[1] = corners[0]
#     else:
#         rect[3] = corners[0]
#         rect[1] = corners[1]
#     print("FPS:{} MS".format((time.time() - start_time) * 1000))
#     # print(rect)
#     # print(corners)
#     (tl, tr, br, bl) = rect
#     width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#
#     # the height of our Sudoku board
#     height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#
#     # take the maximum of the width and height values to reach
#     # our final dimensions
#     max_width = max(int(width_A), int(width_B))
#     max_height = max(int(height_A), int(height_B))
#     pts1 = np.float32([rect[0], rect[1], rect[2], rect[3]])
#     pts2 = np.float32(
#         [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]])  # TL -> TR -> BR -> BL
#     """
#     cv.line(img, tuple(pts1[0]), tuple(pts1[1]), (255, 0, 0), 2)
#     cv.line(img, tuple(pts1[1]), tuple(pts1[2]), (255, 0, 0), 2)
#     cv.line(img, tuple(pts1[2]), tuple(pts1[3]), (255, 0, 0), 2)
#     cv.line(img, tuple(pts1[3]), tuple(pts1[0]), (255, 0, 0), 2)
#     cv.circle(img, tuple(pts1[0]), 5, (0, 0, 0), 3)
#     cv.circle(img, tuple(pts1[1]), 5, (0, 255, 0), 3)
#     cv.circle(img, tuple(pts1[2]), 5, (255, 255, 0), 3)
#     cv.circle(img, tuple(pts1[3]), 5, (0, 255, 255), 3)
#     # cv.imshow("cont", img)
#     """
#     M = cv.getPerspectiveTransform(pts1, pts2)
#     dst = cv.warpPerspective(img, M, (max_width, max_height))
#     # dst = remove_side_lines(dst, 0.1)
#
#     return dst, rect, img, M
#
#
# def remove_side_lines(img, ratio):
#     """
#         Remove lines from image sides
#     """
#     while np.sum(img[0]) <= (1 - ratio) * img.shape[1] * 255:
#         img = img[1:]
#     # Bottom
#     while np.sum(img[:, -1]) <= (1 - ratio) * img.shape[1] * 255:
#         img = np.delete(img, -1, 1)
#     # Left
#     while np.sum(img[:, 0]) <= (1 - ratio) * img.shape[0] * 255:
#         img = np.delete(img, 0, 1)
#     # Right
#     while np.sum(img[-1]) <= (1 - ratio) * img.shape[0] * 255:
#         img = img[:-1]
#     return img
#
#
# def largest_connected_component(image):
#     image = image.astype('uint8')
#     nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
#     sizes = stats[:, -1]
#
#     if (len(sizes) <= 1):
#         blank_image = np.zeros(image.shape)
#         blank_image.fill(255)
#         return blank_image
#
#     max_label = 1
#     # Start from component 1 (not 0) because we want to leave out the background
#     max_size = sizes[1]
#
#     for i in range(2, nb_components):
#         if sizes[i] > max_size:
#             max_label = i
#             max_size = sizes[i]
#
#     img2 = np.zeros(output.shape)
#     img2.fill(255)
#     img2[output == max_label] = 0
#     return img2
#
#
# def get_best_shift(img):
#     # Calculate how to centralize the image using its center of mass
#     cy, cx = ndimage.measurements.center_of_mass(img)
#     rows, cols = img.shape
#     shiftx = np.round(cols / 2.0 - cx).astype(int)
#     shifty = np.round(rows / 2.0 - cy).astype(int)
#     return shiftx, shifty
#
#
# def shift(img, sx, sy):
#     rows, cols = img.shape
#     M = np.float32([[1, 0, sx], [0, 1, sy]])
#     shifted = cv.warpAffine(img, M, (cols, rows))
#     return shifted
#
#
# def write_solution_on_image(image, grid, user_grid):
#     # Write grid on image
#     SIZE = 9
#     width = image.shape[1] // 9
#     height = image.shape[0] // 9
#     for i in range(SIZE):
#         for j in range(SIZE):
#             if (user_grid[i][j] != 0):  # If user fill this cell
#                 continue  # Move on
#             text = str(user_grid[i][j])
#             off_set_x = width // 15
#             off_set_y = height // 15
#             font = cv.FONT_HERSHEY_SIMPLEX
#             (text_height, text_width), baseLine = cv.getTextSize(text, font, fontScale=1, thickness=3)
#             marginX = np.floor(width / 7)
#             marginY = np.floor(height / 7)
#
#             font_scale = 0.6 * min(width, height) / max(text_height, text_width)
#             text_height *= font_scale
#             text_width *= font_scale
#             bottom_left_corner_x = int(width * j + np.floor((width - text_width) / 2) + off_set_x)
#             bottom_left_corner_y = int(height * (i + 1) - np.floor((height - text_height) / 2) + off_set_y)
#             image = cv.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y),
#                                font, font_scale, (0, 255, 0), thickness=3, lineType=cv.LINE_AA)
#     return image
#
#
# def crop_image(img, src, M):
#     original = img.copy()
#     # cv.imshow("img1", img)
#     # img = cv.bitwise_not(img)
#     # cv.imshow("img", img)
#     # _, warp = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
#     # cv.imshow("warp", warp)
#     # mon = cv.erode(img, kernel, iterations=1)
#     # cv.imshow("mon", mon)
#
#     img = cv.resize(img, (500, 500))
#     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     cv.imshow("norm", img)
#     warp = cv.GaussianBlur(img, (3, 3), 0)
#     cv.imshow("gaws", warp)
#     warp = adaptive_thresh(warp)
#     cv.imshow("adpt", warp)
#     warp = cv.bitwise_not(warp)
#     cv.imshow("bitnot", warp)
#     _, warp = cv.threshold(warp, 150, 255, cv.THRESH_BINARY)
#     cv.imshow("warp", warp)
#     # warp = cv.resize(warp, (300, 300))
#
#     kernel = np.ones((3, 3), np.uint8)
#     blocks = []
#     userGrid = np.zeros((9, 9), np.uint8)
#     h = warp.shape[0] // 9
#     w = warp.shape[1] // 9
#     offset_w = np.math.floor(w / 10)  # Offset is used to get rid of the boundaries
#     offset_h = np.math.floor(h / 10)
#     # fig, axs = plt.subplots(9, 9)
#     for i in range(9):
#         for j in range(9):
#             userGrid[i][j] = 1
#             match = True
#             n = i * 9 + j
#             blocks.append(warp[h * i + offset_h:h * (i + 1) - offset_h, w * j + offset_w:w * (j + 1) - offset_w])
#             # blocks[n] = remove_side_lines(blocks[n], 0.6)
#             blocks[n] = cv.bitwise_not(blocks[n])
#             blocks[n] = largest_connected_component(blocks[n])
#
#             # Resize
#             digit_pic_size = 28
#
#             blocks[n] = cv.resize(blocks[n], (digit_pic_size, digit_pic_size))
#             _, blocks[n] = cv.threshold(blocks[n], 200, 255, cv.THRESH_BINARY)
#             # blocks[n] = cv.dilate(blocks[n], kernel, iterations=1)
#             # blocks[n] = cv.erode(blocks[n], kernel, iterations=1)
#             # blocks[n] = cross_closing(blocks[n])
#             # blocks[n] = cross_dilation(blocks[n], 3)
#             # Criteria 1 for detecting white cell:
#             # Has too little black pixels
#             if blocks[n].sum() >= digit_pic_size ** 2 * 255 - digit_pic_size * 1 * 255:
#                 blocks[n] = np.zeros((digit_pic_size, digit_pic_size))
#                 userGrid[i][j] = 0
#                 match = False
#             # Criteria 2 for detecting white cell
#             # Huge white area in the center
#             center_width = blocks[n].shape[1] // 2
#             center_height = blocks[n].shape[0] // 2
#             x_start = center_height // 2
#             x_end = center_height // 2 + center_height
#             y_start = center_width // 2
#             y_end = center_width // 2 + center_width
#             center_region = blocks[n][x_start:x_end, y_start:y_end]
#             if center_region.sum() >= center_width * center_height * 255 - 255:
#                 blocks[n] = np.zeros((digit_pic_size, digit_pic_size))
#                 userGrid[i][j] = 0
#                 match = False
#             # Centralize the image according to center of mass
#             # blocks[n] = cv.bitwise_not(blocks[n])
#             # shift_x, shift_y = get_best_shift(blocks[n])
#             # blocks[n] = shift(blocks[n], shift_x, shift_y)
#             # blocks[n] = cv.bitwise_not(blocks[n])
#             # blocks[n] = cv.resize(blocks[n], (50, 50))
#
#             if match:
#                 temp = np.uint8(blocks[n])
#                 histogram = hog.compute(temp, None, None)
#                 histogram = np.asarray(histogram)
#                 histogram = histogram.reshape((-1, 441))
#                 id = check_match(histogram)
#                 print(i, j, id)
#             # axs[i, j].imshow(blocks[n])
#             # += 300 / 9
#         # h += 300 / 9
#         # w = 0
#     # orb = cv.ORB_create()
#     # _, test = cv.threshold(blocks[7 * 9 + 0], 100, 255, cv.THRESH_BINARY)
#     # test = np.uint8(blocks[4 * 9 + 0])
#     # test = cv.resize(test, (100, 100))
#     # h = hog.compute(test, None, None)
#     # h = np.asarray(h)
#     # h = h.reshape((-1, 441))
#     # id = check_match(h)
#     # print(i, j, id)
#     # kp, des = sift.detectAndCompute(test, None)
#     # test = cv.drawKeypoints(test, kp, None)
#     # cv.imshow("2, 2", test)
#     test = np.uint8(blocks[1 * 9 + 4])
#     cv.imshow("2, 3", test)
#     test = np.uint8(blocks[1 * 9 + 5])
#     cv.imshow("2, 2", test)
#     # if len(kp) > 0:
#     # id = check_match(des)
#     # print(id)
#     # plt.show()
#     lol = write_solution_on_image(original, userGrid, userGrid)
#     result_sudoku = cv.warpPerspective(lol, M, (src.shape[1], src.shape[0])
#                                        , flags=cv.WARP_INVERSE_MAP)
#     result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, src)
#     cv.imshow("res", result)
#
#
# def pre_processing(src):
#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     # start here
#     start_time = time.time()
#     gauss_blur = gaussian_blur(gray)
#     # gauss_blur = gaussian_blur(src, kernel_size=(3, 3), sigmaX=1)
#     # cv.imshow("gauss_blur", gauss_blur)
#     inverse_binary_image = adaptive_thresh(gauss_blur)
#     # cv.imshow("inv_binary_image", inverse_binary_image)
#     # inverse_binary_image = cv.bitwise_not(binary_image)  # Not needed at first
#     # cv.imshow("inverse_bi", inverse_binary_image)
#     # closed_image = cross_closing(inverse_binary_image)  # Not needed at first
#     # cv.imshow("closed_image", closed_image)
#     # dilated_image = cross_dilation(closed_image)  # Not needed at first
#     # cv.imshow("dilated_image", dilated_image)
#     # with_line_image = draw_lines(dilated_image)
#     # grid = find_grid_with_floid(dilated_image)
#     grid, src = find_grid_with_contours(src, inverse_binary_image, start_time)
#     print("FPS:{} MS".format((time.time() - start_time) * 1000))
#     # end here
#     cv.imshow('base', grid)
#     # k = cv.waitKey(0)
#     # if k == 27:
#     # cv.destroyAllWindows()
#     return src
#
#
# if __name__ == '__main__':
#     original_sudoku = cv.imread('resources/sod8.jpg')
#     # prepare_numbers_features(9)
#     pre_processing(original_sudoku)
#     k = cv.waitKey(0)
#     if k == 27:
#         cv.destroyAllWindows()
#     # cap = cv.VideoCapture(0)
#     cap = cv.VideoCapture("resources/sudoku2.mp4")
#     cap.set(3, 1280)  # HD Camera
#     cap.set(4, 720)
#     old_sudoku = None
#
#     while (True):
#         ret, frame = cap.read()  # Read the frame
#         if ret == True:
#             # frame = pre_processing(frame)
#             # cv.imshow("lol", frame)
#             # sudoku_frame = RealTimeSudokuSolver.recognize_and_solve_sudoku(frame, model, old_sudoku)
#             # showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)  # Print the 'solved' image
#             if cv.waitKey(1) & 0xFF == ord('q'):  # Hit q if you want to stop the camera
#                 break
#         else:
#             break
#
#     cap.release()
#     # out.release()
#     cv.destroyAllWindows()
