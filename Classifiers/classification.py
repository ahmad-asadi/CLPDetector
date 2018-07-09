import os
from image_processings.preprocessors import ImagePreprocessor
from Classifiers.CNN import CNN


class Classifier:
    label2ind = dict()
    ind2label = dict()
    features = dict()

    @classmethod
    def train(cls, data_dir="/home/ahmad/Programs/Programming/CarLicensePlateDetection/Dataset/TrainingChars/"):
        cls.create_label_index(data_dir)
        cls.dataset = ImagePreprocessor.load_and_preprocess(data_dir=data_dir, label2ind=cls.label2ind)
        cls.cnn = CNN(dataset=cls.dataset, label2ind=cls.label2ind, ind2label=cls.ind2label)
        cls.cnn.train()

    @classmethod
    def create_label_index(cls, data_dir):
        ind = 0
        for file in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, file)):
                ind = cls.index_new_label(file, ind)

    @classmethod
    def index_new_label(cls, file, ind):
        cls.label2ind[file] = ind
        cls.ind2label[ind] = file
        ind += 1
        return ind


Classifier.train()
