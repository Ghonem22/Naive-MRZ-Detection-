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
 * We can enhance MRZ detection using object detection model (if we have data), or we can do much optimzation to same concept to detect contours in better way
 * The Ectraction part using the OCR isn't accurate, we can enhnce that by using super resolution to increase image quality before using OCR
 
---
## Resources:

 * https://pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/
