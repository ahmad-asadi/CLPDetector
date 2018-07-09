import os
import numpy as np
from PIL import Image


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
                image = image_src.resize((28, 28))
                images.append(np.array(image))
                image.close()
        return images
