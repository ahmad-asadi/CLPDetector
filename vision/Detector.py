import cv2 as cv
from cv2 import ml_KNearest as KNN
import numpy as np
import random as rand

import math

from image_processings.ImageProcessor import ImageProcessor


class Detector:
    averaging_region_count = 4
    lp_locate_winw = 200
    lp_locate_winh = 50
    lp_locate_stepw = 20
    lp_locate_steph = 10
    lp_search_cut_ratio = 5

    char_split_histh = 300
    char_split_avg_scale = 0.7

    @classmethod
    def detect_lp_location(cls, img, src_img, pyr_ratio, verbose=False):

        char_regs = []

        rows, cols = map(int, img.shape)

        lp_search_region = img[:, cols // cls.lp_search_cut_ratio:cols // cls.lp_search_cut_ratio * (
                cls.lp_search_cut_ratio - 1)]

        lp_search_offset = cols // cls.lp_search_cut_ratio

        vote_map = cls.create_vote_map(lp_search_region)

        _, lp_mask = cv.threshold(src=vote_map, thresh=0.5, maxval=255, type=cv.THRESH_BINARY)
        lp_reg = np.bitwise_and(lp_search_region, lp_mask.astype(np.uint8))

        h, w, x, y = cls.find_best_bounding_rect(lp_reg)
        if verbose:
            rate = w / h if h != 0 else 0
            print(w, ", ", h, ", ", rate)

        lp_reg = cls.cut_lp_bounding_box(h, lp_reg, lp_search_offset, pyr_ratio, src_img, w, x, y)
        if lp_reg is not None:
            char_regs = cls.guess_char_regions(lp_reg)
            lp_reg = cls.precise_lp(lp_reg)
            if 250 > w > 100 > h and 2 <= w // h < 7:
                return lp_reg, char_regs

        return None, char_regs

    @classmethod
    def create_vote_map(cls, lp_search_region):
        rows, cols = map(int, lp_search_region.shape)
        vote_map = np.zeros(shape=lp_search_region.shape, dtype=np.float)
        for r in range(0, rows - cls.lp_locate_winh, cls.lp_locate_steph):
            for c in range(0, cols - cls.lp_locate_winw, cls.lp_locate_stepw):
                srwin = lp_search_region[r:r + cls.lp_locate_winh, c: c + cls.lp_locate_winw]
                sobelx64f = cv.Sobel(srwin, cv.CV_64F, 1, 0, ksize=3)
                abs_sobel64f = np.absolute(sobelx64f)
                sobel_8u = np.uint8(abs_sobel64f)
                col_hist = np.sum(sobel_8u, axis=0)
                row_hist = np.sum(sobel_8u, axis=1)
                counts = np.bincount(row_hist.astype(np.int64))
                freq_in_row = np.argmax(counts) // 255
                counts = np.bincount(col_hist.astype(np.int64))
                freq_in_col = np.argmax(counts) // 255
                if np.mean(row_hist) // 255 > 13 and freq_in_row == 0 and freq_in_col == 0:
                    vote_map[r:r + cls.lp_locate_winh, c: c + cls.lp_locate_winw] += 0.05
        return vote_map

    @classmethod
    def cut_lp_bounding_box(cls, h, lp_reg, lp_search_offset, pyr_ratio, src_img, w, x, y):
        w += 50
        h += 20
        x -= 25
        y -= 10
        lp_reg = src_img[y * (2 ** pyr_ratio):(y + h) * (2 ** pyr_ratio), (lp_search_offset + x) * (2 ** pyr_ratio):
                                                                          (x + w + lp_search_offset) * (
                                                                                  2 ** pyr_ratio)]
        return lp_reg

    @classmethod
    def find_best_bounding_rect(cls, lp_reg):
        _, contours, hierarchy = cv.findContours(lp_reg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0, 0, 0
        # Find the index of the largest contour
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv.boundingRect(cnt)
        return h, w, x, y

    @classmethod
    def split_lp_char_regions(cls, img, verbose=False):
        img = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
        kernel = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
        img = cv.filter2D(img, -1, kernel)
        # img = cv.threshold(src=img, thresh=50, maxval=255, type=cv.THRESH_BINARY_INV)
        rows, cols = map(int, img.shape)
        hist = 1 - np.sum(img, axis=0) / (255 * rows)
        if np.max(hist) - np.average(hist) < 0.15:
            return None
        hist_size = len(hist)
        hist_img = np.zeros(shape=(cls.char_split_histh, cols, 3))
        split_points = [0]
        averaging_regions = []
        for i in range(cls.averaging_region_count):
            averaging_regions.append(range(int(i * hist_size / cls.averaging_region_count),
                                           int((i + 1) * hist_size / cls.averaging_region_count)))
        for i in range(1, hist_size - 1):
            split_point = False
            if (hist[i] < cls.char_split_avg_scale * np.average(
                    hist[averaging_regions[int(cls.averaging_region_count * i / hist_size)]])) \
                    and hist[i - 1] >= hist[i] and hist[i + 1] >= hist[i] and \
                    (len(split_points) == 0 or i - split_points[-1] > 10):
                split_points.append(i)
                split_point = True
                # print((hist[i] - hist[i-1]) * (hist[i+1] - hist[i]))

            if split_point:
                hist_img = cv.circle(hist_img, (i, int(cls.char_split_histh * (1 - hist[i]))), radius=2,
                                     color=(255, 0, 255), thickness=5)

            hist_img = cv.line(hist_img, (i - 1, int(cls.char_split_histh * (1 - hist[i - 1]))),
                               (i, int(cls.char_split_histh * (1 - hist[i]))), (255, 255, 255), 2)
            hist_img = cv.line(hist_img, (i - 1, int(cls.char_split_histh * (1 - cls.char_split_avg_scale * np.average(
                hist[averaging_regions[int(cls.averaging_region_count * i / hist_size)]])))),
                               (i, int(cls.char_split_histh * (1 - cls.char_split_avg_scale * np.average(
                                   hist[averaging_regions[int(cls.averaging_region_count * i / hist_size)]])))),
                               (255, 0, 0), 1)

        split_points.append(hist_size)

        if verbose:
            print("number of detected split points:", len(split_points))

        if len(split_points) < 9:
            return None

        if verbose:
            print(np.max(hist), np.min(hist), np.average(hist))

        ImageProcessor.show_image(img, win_name="lp", wait=1)
        ImageProcessor.show_image(hist_img, win_name="hist", wait=1, width=cols, height=cls.char_split_histh)

        splitted_chars = []
        for i in range(len(split_points)):
            if i == 0:
                prev_point = 0
            else:
                prev_point = split_points[i - 1]
            new_char = img[:, prev_point:split_points[i]]
            if 5 <= len(new_char[0]) <= 60:
                splitted_chars.append(new_char)
                # ImageProcessor.show_image(new_char)

        return splitted_chars

    @classmethod
    def _find_if_close(cls, cnt1, cnt2, threshold):
        dist = np.inf
        for i in range(len(cnt1)):
            for j in range(len(cnt2)):
                dist = min(dist, np.linalg.norm(cnt1[i] - cnt2[j]))
        return dist < threshold

    @classmethod
    def merge_nearby_contours(cls, contours, threshold=2):

        contours_length = len(contours)
        status = np.zeros((contours_length, 1))

        for i, cnt1 in enumerate(contours):
            x = i
            if i != contours_length - 1:
                for j, cnt2 in enumerate(contours[i + 1:]):
                    x = x + 1
                    if cls._find_if_close(cnt1, cnt2, threshold):
                        val = min(status[i], status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x] == status[i]:
                            status[x] = i + 1

        unified = []
        maximum = int(status.max()) + 1
        for i in range(maximum):
            pos = np.where(status == i)[0]
            if pos.size != 0:
                cont = np.vstack(contours[i] for i in pos)
                # hull = cv.convexHull(cont)
                unified.append(cont)
        return unified

    @classmethod
    def precise_lp(cls, lp_candid):
        gray = cv.cvtColor(lp_candid, cv.COLOR_BGR2GRAY)

        thresh = gray

        thresh = ImageProcessor.extract_edges(thresh)

        _, contours, h = cv.findContours(thresh, mode=cv.RETR_TREE, method=2)

        temp_img = np.zeros(lp_candid.shape)
        r = rand.Random()
        for i in range(len(contours)):
            color = [r.random(), r.random(), r.random()]
            cv.drawContours(temp_img, contours, i, color=color)

        max_rect = None
        max_rect_area = -1
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if 3 < w/h < 5:
                if w * h > max_rect_area:
                    max_rect = [x, y, w, h]
                    max_rect_area = w * h

        if max_rect is None or max_rect_area < 10000:
            return None

        x = max_rect[0]
        y = max_rect[1]
        w = max_rect[2]
        h = max_rect[3]
        # cv.rectangle(lp_candid, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ImageProcessor.show_image(lp_candid[y:y+h, x:x+w], "lp_candid", wait=1)
        # ImageProcessor.show_image(temp_img, "temp_img", wait=1)
        # ImageProcessor.show_image(thresh, "lp_candid_gray")
        return lp_candid[y:y+h, x:x+w]

    @classmethod
    def guess_char_regions(cls, lp_candid):
        char_regs = []

        # lp_candid = cv.cvtColor(lp_candid, cv.COLOR_BGR2GRAY)

        thresh = lp_candid
        # thresh = lp_candid

        thresh = ImageProcessor.extract_edges(thresh)

        _, contours, h = cv.findContours(thresh, mode=cv.RETR_TREE, method=2)

        if lp_candid is None:
            return char_regs
        temp_img = np.zeros(lp_candid.shape)
        r = rand.Random()
        for i in range(len(contours)):
            color = [r.random(), r.random(), r.random()]
            cv.drawContours(temp_img, contours, i, color=color)

        x = y = w = h = 0
        # print("---------------------")
        indices = np.zeros(shape=(len(contours), 4))
        idx = 0
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w > 0 and h > 0 and 2000 > w*h > 500 and float(h)/w > 0.3:
                indices[idx, :] = [x, y, w, h]
                idx += 1

        x_pad = 10
        prev_x = -x_pad
        indices = indices[~np.all(indices == 0, axis=1)]
        if len(indices) > 0:
            indices.view("i8,i8,i8,i8").sort(order=["f0"], axis=0)
            for ind in indices:
                x = int(ind[0])
                y = int(ind[1])
                w = int(ind[2])
                h = int(ind[3])
                if x - prev_x < x_pad:
                    prev_x = x
                    continue
                prev_x = x

                ch_reg = lp_candid[y:y + h, x:x + w]
                # _, ch_reg = cv.threshold(ch_reg, 40, 255, cv.THRESH_BINARY)
                char_regs.append(ch_reg)
                # ImageProcessor.show_image(ch_reg, "ch_reg")

        return char_regs
