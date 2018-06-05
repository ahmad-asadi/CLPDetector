import sys
import os

from image_processings.ImageProcessor import ImageProcessor
from vision.Detector import Detector


class Coordinator:
    def __init__(self):
        self.data_directory = None
        self.verbose = False
        self.show_images = False
        self.get_input()
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

            img = Detector.detect_lp_location(img, src_img=full_res_img, pyr_ratio=ImageProcessor.pyr_down_times,
                                              verbose=self.verbose)
            if img is None:  # No LP detected
                continue

            char_imgs = Detector.split_lp_char_regions(img, self.verbose)
            if char_imgs is None:  # Wrong LP detection
                continue

            char_ind = 0
            for char in char_imgs:
                ImageProcessor.writeFile(char, file_name="char_"+str(char_ind)+"_frame_" + str(i) + ".png")
                char_ind += 1


Coordinator()
