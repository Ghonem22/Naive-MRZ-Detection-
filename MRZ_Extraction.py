from imutils import paths
import numpy as np
import imutils
import cv2
import PIL
import easyocr
import pytesseract
import boto3
import yaml

with open("utilities/configs.yml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)


class OCR:
    def __init__(self, module='easyocr', AccessKeyID='', SecretAccessKey=''):
        print(f"Using ocr: {module}")
        self.module = module
        if self.module == 'easyocr':
            self.reader = easyocr.Reader(['en'])

        if self.module == 'aws':
            self.rekognition_client = boto3.client('rekognition',
                                                   aws_access_key_id=AccessKeyID,
                                                   aws_secret_access_key=SecretAccessKey,
                                                   region_name='us-east-1')

    def get_text_using_textract(self, img):
        # read image as byte object if the input is image path
        if type(img) == str:
            with open(img, 'rb') as image_file:
                image_bytes = image_file.read()

        # convert the image to byte object if the input is image
        elif type(img) != bytes:
            success, encoded_image = cv2.imencode('.png', img)
            image_bytes = encoded_image.tobytes()

        else:
            image_bytes = img
        # get text in image using AWs rekognition
        return self.rekognition_client.detect_text(Image={'Bytes': image_bytes})

    def get_txt(self, image):
        if self.module == 'pytesseract':
            txt = pytesseract.image_to_string(image, lang='eng')
            txt_lines = txt.split("\n")
            return [t for t in txt_lines if t]

        elif self.module == 'aws':
            txt = self.get_text_using_textract(image)
            return [t["DetectedText"] for t in txt["TextDetections"] if t["Type"] == "LINE"]

        else:
            # consider easyocr is the default one
            txt = self.reader.readtext(image)
            return [t[1] for t in txt]


class MrzExtractor:
    def __init__(self, sqkernel_sizes=None, ocr_module='easyocr'):
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

        if not sqkernel_sizes:
            sqkernel_sizes = [21, 13, 11]

        self.sqKernel_sizes = sqkernel_sizes
        self.ocr = OCR(module=ocr_module, AccessKeyID=config["AccessKeyID"], SecretAccessKey=config["SecretAccessKey"])

    def get_sorted_contours(self, gray, kernel_size):
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # smooth the image using a 3x3 Gaussian, then apply the blackhat
        # morphological operator to find dark regions on a light background
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)

        # compute the Scharr gradient of the blackhat image and scale the
        # result into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # apply a closing operation using the rectangular kernel to close
        # gaps in between letters -- then apply Otsu's thresholding method
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, self.rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform another closing operation, this time using the square
        # kernel to close gaps between lines of the MRZ, then perform a
        # series of erosions to break apart connected components
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.erode(thresh, None, iterations=4)

        # during thresholding, it's possible that border pixels were
        # included in the thresholding, so let's set 5% of the left and
        # right borders to zero
        p = int(gray.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, gray.shape[1] - p:] = 0

        # find contours in the thresholded image and sort them by their
        # size
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return sorted(cnts, key=cv2.contourArea, reverse=True)

    def get_rois(self, cnts, image):
        rois = []
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and use the contour to
            # compute the aspect ratio and coverage ratio of the bounding box
            # width to the width of the image
            # (x,y): top-left coordinate of the rectangle and (w,h): width and height.

            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            crWidth = w / float(image.shape[1])
            # check to see if the aspect ratio and coverage width are within
            # acceptable criteria
            if ar > 6 and crWidth > 0.3:
                # pad the bounding box since we applied erosions and now need
                # to re-grow it
                pX = int((x + w) * 0.05)
                pY = int((y + h) * 0.03)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))
                # extract the ROI from the image and draw a bounding box
                # surrounding the MRZ
                roi = image[y:y + h, x:x + w].copy()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rois.append(roi)


        # show the output images
        return rois

    def get_text_from_rois(self, rois):
        if len(rois) == 1:
            return self.ocr.get_txt(rois[0])
        else:
            lines = []
            for roi in rois:
                line = "".join(self.ocr.get_txt(roi))
                lines.append(line)
            return lines

    def extract_mrz(self, imagepath, visualize_mrz=False):

        for kernel_size in self.sqKernel_sizes:
            # load the image, resize it, and convert it to grayscale
            image = PIL.Image.open(imagepath)
            image = np.array(image)
            image = imutils.resize(image, height=600)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cnts = self.get_sorted_contours(gray, kernel_size)
            rois = self.get_rois(cnts, image)
            if len(rois) > 0:

                # to visulize image after drawing rectangle over region of interest
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # extract text using ocrr
                return self.get_text_from_rois(rois)
        return []

