from MRZ_Extraction import *
import glob
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_path", required=True, help="path to images directory")
ap.add_argument("-o", "--ocr", default="easyocr", required=False, help="ocr module")
ap.add_argument("-v", "--visualize", default=False, required=False, help="if we want to visulaize each image after "
                                                                         "darwing MRZ rectangle")
args = vars(ap.parse_args())




if __name__ == '__main__':
    mrz_extractor = MrzExtractor(ocr_module=args["ocr"])
    data_path = os.path.join(args["images_path"], '*g')
    images = glob.glob(data_path)

    # create results folder if it doesn't exist'
    isExist = os.path.exists("results")
    if not isExist:
        os.mkdir("results")

    for img_path in images:
        texts = mrz_extractor.extract_mrz(img_path, args["visualize"])
        txt_path = "results/" + img_path.split(".")[0].split("/")[-1].split("\\")[-1] + '.txt'

        file = open(txt_path, 'w')
        for txt in texts:
            file.write(txt + "\n")
        file.close()


