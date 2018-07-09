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

        # mask = cv.erode(mask, kernel=np.ones((5, 5), np.uint8))

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

    @classmethod
    def morphological_open(cls, img, iterations=1):
        return cv.morphologyEx(img, cv.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8), iterations=iterations)

    @classmethod
    def morphological_close(cls, img):
        return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

    @classmethod
    def cvt_to_gray(cls, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @classmethod
    def morphological_gradient(cls, img):
        return cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel=np.ones((3, 3), np.uint8))

    @classmethod
    def extract_edges(cls, img):
        return cv.Canny(img, 100, 300)

    @classmethod
    def find_corners(cls, img):
        lines = cv.HoughLines(img, 1, np.pi / 180, 10)
        if lines is None:
            return None
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv.imwrite('houghlines3.jpg', img)
        return img
