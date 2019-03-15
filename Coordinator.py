import sys
import os

from image_processings.ImageProcessor import ImageProcessor
from image_processings.preprocessors import ImagePreprocessor as Preprocessor
from vision.Detector import Detector
from Classifiers.classification import Classifier
import requests
import json

class Coordinator:
    def __init__(self):
        self.data_directory = None
        self.verbose = False
        self.show_images = False
        self.classifier = Classifier()
        self.classifier.train(data_dir="./clp", load_if_exists=True)
        # self.classifier.train(load_if_exists=False)
        self.get_input()

        self.url = "http://172.23.186.247/api/car-entered?"
        self.token = "3c461d6a343a40ea297180baebe5ebecc4d7b"

        self.start()

    def get_input(self):
        for i in range(len(sys.argv)):
            if i == 0:
                continue
            argv = sys.argv[i]
            if argv == "-d":
                self.data_directory = sys.argv[i + 1]
            elif argv == "--verbose":
                self.verbose = True
            elif argv == "--showimages":
                self.show_images = True
            elif argv == "--url":
                self.url = sys.argv[i + 1]
            elif argv == "--token":
                self.token = sys.argv[i + 1]

    def start(self):

        if self.verbose:
            print("opening input stream...")

        file_names = [file for file in os.listdir(self.data_directory) if file != "." or file != ".."]
        file_names.sort()

        if self.verbose:
            print("dataset is loaded")
            print("found ", len(file_names), " images in specified directory")

        for i in range(550, len(file_names)):
            fname = file_names[i]
            if self.verbose:
                print("processing image #", i, " in file:", fname)
            img = ImageProcessor.load_image(img_name=os.path.join(self.data_directory, fname))
            full_res_img = img.copy()

            if self.show_images:
                ImageProcessor.show_image(img=img, win_name="input", width=500, height=500, wait=1)

            img = ImageProcessor.lp_locate_preprocess(img)

            if self.show_images:
                ImageProcessor.show_image(img, win_name="prep", width=500, height=500, wait=1)

            img, char_imgs = Detector.detect_lp_location(img, src_img=full_res_img, pyr_ratio=ImageProcessor.pyr_down_times,
                                              verbose=self.verbose)
            if img is None:  # No LP detected
                continue

            # char_ind = 0
            # for char in char_imgs:
            #     ImageProcessor.show_image(char, "candid", wait=1)
            #     ImageProcessor.writeFile(char, file_name="char_"+str(char_ind)+"_frame_" + str(i) + ".png")
            #     char_ind += 1


            # char_imgs = Detector.split_lp_char_regions(img, self.verbose)
            # if char_imgs is None:  # Wrong LP detection
            #     continue

            # char_ind = 0
            # for char in char_imgs:
            #     ImageProcessor.writeFile(char, file_name="char_"+str(char_ind)+"_frame_" + str(i) + ".png")
            #     char_ind += 1

            print("***********************************")
            char_ind = 0
            lp = []
            for char in char_imgs:
                char = Preprocessor.preprocess_np_array(char)
                # ImageProcessor.show_image(img=char, win_name="predicting char ...")
                pred = self.classifier.predict(char)
                if pred != "_NOISE":
                    print(pred)
                    lp.append(pred)
                # ImageProcessor.writeFile(char, file_name=str(pred) + "/char_"+str(char_ind)+"_frame_" + str(i) + ".png")
                char_ind += 1
            print("***********************************")
            print("detected LP: ", lp)
            if len(lp) == 8:
                payload = {'plate_first': str(lp[0]) + str(lp[1]),
                           'plate_middle': str(lp[2]),
                           'plate_last' : str(lp[3]) + str(lp[4]) + str(lp[5]),
                           'plate_global': str(lp[6]) + str(lp[7]),
                           'token': self.token}
                headers = {'content-type': 'application/json'}
                r = requests.post(self.url, data=json.dumps(payload), headers=headers)
                print(r)



Coordinator()
