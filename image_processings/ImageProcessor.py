import cv2 as cv
import numpy as np
from os.path import join


class ImageProcessor:
    pyr_down_times = 1

    @classmethod
    def load_image(cls, img_name):
        return cv.imread(img_name, cv.IMREAD_COLOR)

    @classmethod
    def show_image(cls, img, win_name="temp img", wait=None, width=None, height=None):
        if (width is not None) and (height is not None):
            cv.namedWindow(win_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(win_name, width, height)

        cv.imshow(win_name, img)

        if wait is None:
            cv.waitKey()
        else:
            cv.waitKey(wait)

    @classmethod
    def lp_locate_preprocess(cls, img):
        for i in range(cls.pyr_down_times):
            rows, cols, _ = map(int, img.shape)
            img = cv.pyrDown(src=img, dstsize=(cols // 2, rows // 2))

        mask = cls.white_filter(img)

        mask = cv.medianBlur(mask, ksize=3)

        return mask

    @classmethod
    def white_filter(cls, img):
        lower_white = np.array([100, 100, 100])
        upper_white = np.array([255, 255, 255])
        mask = cv.inRange(img, lower_white, upper_white)
        return mask

    @classmethod
    def writeFile(cls, img, file_name, output_dir="/tmp/clp"):
        cv.imwrite(join(output_dir, file_name), img)
