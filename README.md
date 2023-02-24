# Simple-MRZ-Detection-

## install requirements:
  1. install required libraries
        pip install -r requiremtns.txt
     
     
  2. setup pytesseract:
   
   a. for windows:
       follow this article: https://linuxhint.com/install-tesseract-windows/
      
   b. for linux:
       sudo apt install tesseract-ocr
      
## How to run:
  1. upload all the images you want to extract MRZ from it to "images" folder
  2. run this script:
        python run.py --images_path images --visualize True --ocr aws
        
        * images_path: the path of the direcotry that contains images
        * visualize: if we want to visualize each image 
        * ocr: the module we want to use to extract text after detecting MRZ region
 we have three options for the ocr:
    * easyocr
    * pytesseract
    * aws: using Amazon Textract, but you need to add your aws creditnal to "utilities/configs.yml"
