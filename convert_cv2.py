#!/usr/bin/env python  # https://blog.python.org/2011/07/python-launcher-for-windows_11.html
# -*- coding: utf-8 -*-

# https://blog.42mate.com/opencv-tesseract-is-a-powerful-combination/
import numpy as np
# pip install numpy

import pdf2image
# install https://github.com/oschwartz10612/poppler-windows/releases/
# pip install pdf2image
POPPERPATH=r"E:\Program Files (x86)\poppler-22.04.0\Library\bin"
# set to where you place popper in your system
import cv2
# pip install opencv_python

import matplotlib.pyplot as plt
# pip install --user matplotlib

import pytesseract
from pytesseract import Output

LANG = "chi_tra+por+eng"
config = r'--psm 6 --oem 3'

def convert_pdf_to_image(document, dpi):
    images = []
    images.extend(
        list(
            map(
                lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR), pdf2image.convert_from_path(document, dpi=dpi, poppler_path=POPPERPATH),
            )
        )
    )
    return images

def mark_boxes(img, d):
    n = len(d['text'])
    for i in range(n):
        if int(float(d['conf'][i])) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    return img

def tes_boxes(img):
    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    return img

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def pre_poss(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    
