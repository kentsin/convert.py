#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-18 09:11:40
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

from importlib.resources import path
import cv2
import pytesseract

image = cv2.imread(path)

working = image.copy()

gray = cv2.cvColor(working, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 50))
dilate = cv2.dilate(thresh, kernal, iterations=1)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h > 200 and w > 250:
        roi = base_image[y:y+h, x:x+w]
        cv2.rectangle(image, (x,y), (x+w, y+h), (36, 255, 12), 2)

cv2.imwrite("temp/sample_boxes.png", image)
ocr_result_original = pytesseract.image_to_string(base_image)


import cv2


image = cv2.imread('data/sample_mgh_2.jpg')
base_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,25))
dilate = cv2.dilate(thresh, kernel, iterations=1)

# Find contours and draw rectangle
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
main_text = ""
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h > 200 and w > 250:
        roi = base_image[y:y+h, 0:x]
#         cv2.rectangle(image, (0, y), (x, 0 + h+20), (36,255,12), 2)
        
        constant= cv2.copyMakeBorder(roi.copy(),30,30,30,30,cv2.BORDER_CONSTANT,value=[255,255,255])
        ocr_result = pytesseract.image_to_string(constant)
        cv2.imwrite("temp/output.png", roi)
        
        print (ocr_result)
#         print (ocr_result)
# cv2.imwrite("temp/output.png", image)


import pytesseract
import cv2


image = cv2.imread('data/sample_mgh.jpg')
im_h, im_w, im_d = image.shape
base_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,10))
dilate = cv2.dilate(thresh, kernel, iterations=1)

# Find contours and draw rectangle
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h < 20 and w > 250:
        roi = base_image[0:y+h, 0:x+im_w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        
cv2.imwrite("temp/output.png", roi)