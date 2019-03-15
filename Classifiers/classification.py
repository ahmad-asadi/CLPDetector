import os
from image_processings.preprocessors import ImagePreprocessor
from Classifiers.CNN import CNN


class Classifier:
    label2ind = dict()
    ind2label = dict()
    features = dict()
    cnn = None

    @classmethod
    def train(cls, data_dir="/home/ahmad/Programs/Programming/CarLicensePlateDetection/Dataset/TrainingChars_/",
              load_if_exists=True):
        cls.create_label_index(data_dir)
        cls.dataset = ImagePreprocessor.load_and_preprocess(data_dir=data_dir, label2ind=cls.label2ind)
        cls.cnn = CNN(dataset=cls.dataset, label2ind=cls.label2ind, ind2label=cls.ind2label)
        cls.cnn.train()

        # if not load_if_exists:
        #     cls.cnn.train()
            # cls.cnn.freeze()
        # else:
            # graph = cls.cnn.load_graph()
            #
            # # We can verify that we can access the list of operations in the graph
            # for op in graph.get_operations():
            #     print(op.name)
            #
            # # We access the input and output nodes
            # x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
            # y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')

            # cls.cnn.export_model()

    @classmethod
    def predict(cls, image):

        return cls.cnn.predict_single_image(image)

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


if __name__ == "main":
    Classifier.train()
