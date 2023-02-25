# Naive-MRZ-Detection-

## install requirements:
  **1. install required libraries**
  
        pip install -r requiremtns.txt
     
     
  **2. setup pytesseract:**
   
   a. for windows:
   
       follow this article: https://linuxhint.com/install-tesseract-windows/
      
   b. for linux:
   
       sudo apt install tesseract-ocr
      
---

## How to run:
  1. upload all the images you want to extract MRZ from them to "images" folder
  
  3. run this script:
  
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

 * This is just naive pipeline using classical image processing
 * We can enhance MRZ detection using object detection model (if we have data).
 * we can optimize same pipeline by detecting the large squere contour first, and apply contours detection on that part, this may fix problem of defining the right kernel size.
 * The Ectraction part using the OCR isn't accurate, we can enhnce that by using super resolution to increase image quality before using OCR
 * There's diffrent papers that introduce better approaches, but they depend on CNN, So we need dataset which isn't avialble for our case, so I had to use classical image processing, for ex, https://paperswithcode.com/paper/mrz-code-extraction-from-visa-and-passport/review/
 
 
 
---
## Resources:

 * https://pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/
