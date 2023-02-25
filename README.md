# Naive-MRZ-Detection-

This is an implementation of extracting Machine-Readable Zones (MRZ) from an image based on image processing. The MRZ region contains text, which is used for machine-reading the identity of passports, visas, and other travel documents.


**The code performs the following steps to extract the MRZ region:**

1. Adjust image rotation so that the face in the image isn't rotated or flipped
2. Applies morphological operations (closing) on order to connect the MRZ text in one/ two blocks.
3. Gets contours from the processed image with different kernel sizes.
4. Filters out the contours based on their width, height, and aspect ratio.
5. Uses Optical Character Recognition (OCR) to extract the text from the filtered contour regions.
---


## How to run:
  **1. install required libraries**
  
        pip install -r requiremtns.txt
     
     
  **2. setup pytesseract:**
   
   a. for windows:
   
       follow this article: https://linuxhint.com/install-tesseract-windows/
      
   b. for linux:
   
       sudo apt install tesseract-ocr
      

  3. upload all the images you want to extract MRZ from them to "images" folder
  
  4. run this script:
  
            python run.py --images_path images --visualize True --ocr easyocr
        
        * images_path: the path of the direcotry that contains images
        * visualize: if we want to visualize each image 
        * ocr: the module we want to use to extract text after detecting MRZ region


 **we have three options for ocr argument:**
 * easyocr
 * pytesseract
 * aws: using Amazon Textract, but you need to add your aws creditnal to "utilities/configs.yml"
 
---

## Some notes:

1. The current pipeline for MRZ detection is based on classical image processing and is considered naive.
2. We can apply some optimization to this pipeline by first detecting the large square contour (passport) and then applying contour detection on this region. This may solve the problem of defining the right kernel size.
3. The OCR-based extraction part of the pipeline is not accurate. We can enhance it by using super resolution to increase image quality before using OCR.
4. There are papers that introduce better approaches for MRZ detection, but they depend on CNNs. Unfortunately, we don't have the dataset available for our case, so we had to rely on classical image processing. One such paper is "MRZ Code Extraction from Visa and Passport" (https://paperswithcode.com/paper/mrz-code-extraction-from-visa-and-passport/review/).

 
 
---
## Resources:

 * https://pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/
