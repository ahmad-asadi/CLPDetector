import os
import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt


class ImagePreprocessor:
    @classmethod
    def load_and_preprocess(cls, data_dir, label2ind):
        images = dict()
        for label in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, label)):
                images[label2ind[label]] = cls.extract_and_resize_one_label(os.path.join(data_dir, label))
        return images

    @classmethod
    def extract_and_resize_one_label(cls, data_dir):
        images = []
        for file in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, file)):
                image_src = Image.open(os.path.join(data_dir, file))
                image_prep = cls.preprocess_single_pil_image(image_src)
                images.append(image_prep)
                # image_prep.close()
        return images

    @classmethod
    def preprocess_img(cls, image):
        image = image.resize((28, 28))
        # img = np.array(image)
        # image_prep = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # image.close()
        # return image_prep
        return image

    @classmethod
    def preprocess_np_array(cls, image):
        image = Image.fromarray(image)
        return cls.preprocess_single_pil_image(image)

    @classmethod
    def preprocess_single_pil_image(cls, image):
        image = cls.preprocess_img(image)
        image = cv.cvtColor(np.array(image), cv.COLOR_BGR2GRAY)
        image = image / float(np.max(image))
        return image.astype(dtype=np.float32)
