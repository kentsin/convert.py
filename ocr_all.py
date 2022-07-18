#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-18 10:45:37
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

import os
# from unittest import skip
# from openpyxl import Workbook
# https://www.youtube.com/watch?v=ADV-AjAXHdc
# https://blog.42mate.com/opencv-tesseract-is-a-powerful-combination/

# poppler   https://github.com/oschwartz10612/poppler-windows/releases/
# pdf2image pip install pdf2image
#from binhex import BinHex
#from os import symlink
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pdf2image
poppler_path = r"E:\Program Files (x86)\poppler-22.04.0\Library\bin"

#
# https://blog.42mate.com/opencv-tesseract-is-a-powerful-combination/
# pip install numpy

# pip install opencv_python

# pip install --user matplotlib

# require pillow, numpy
# install tesseract add tesseract to PATH
# Margins, object fall off the margins will be ignored
MT = 120
MB = 20
ML = 20
MR = 20

MC = 10  # Maximum objects in a single page. More than that will consider fail
MA = 100  # Minimum object Area. Smaller objects will be ignored

DPI = 300

LANG = "chi_tra+por+eng"
config = r'--psm 6 --oem 3'

# useful functions


def load_imgs(path, dpi=DPI):
    imgs = []
    imgs.extend(list(map(lambda img: cv2.cvtColor(np.asarray(img), code=cv2.COLOR_RGB2BGR),
                pdf2image.convert_from_path(path, dpi=dpi, poppler_path=poppler_path), )))
    return imgs


def display(img):
    dpi = 80

    height, width = img.shape[:2]

    # what size does the figurer need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # hide spines, ticks, etc.
    ax.axis('off')

    # display the image
    ax.imshow(img, cmap='gray')

    plt.show()


def getSkewAngle(Img) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = Img.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    # cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center


def rotateImage(Img, angle: float):
    newImage = Img.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(Img):
    angle = getSkewAngle(Img)
    return rotateImage(Img, -1.0 * angle)

# Remove Borders


def remove_borders(img):
    contours, heiarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]
    return (crop)


def proc_page(file):
    imgs = load_imgs(file)
    i = 0
    for img in imgs:
        i = i+1
        work = img.copy()
    # deSkew
    # remove lines
    # find contours
    #
        # work = remove_borders(work)
        work = deskew(work)
        # copy work
        baseimg = work.copy()

        # find contours

        for s in [1, 2, 3, 4]:
            w = int(work.shape[1] / s)
            h = int(work.shape[0] / s)
            dim = (w, h)
            work = cv2.resize(work, dim, interpolation = cv2.INTER_AREA)
            work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            work = cv2.GaussianBlur(work, (7, 7), 0) # adjustable
            Work = cv2.threshold(
                work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13)) # adjustable
            work = cv2.dilate(work, kernal, iterations=1) # iterations adjustable?
        #    cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts)==2 else cnts[1]

            if len(cnts) < MC : break  # We are satisfied 
        
        #    result = cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        #    return image
        # mark_boxes
        cnts = sorted(cnts, key=lambda y: cv2.boundingRect(y)[1])
        # Now process images, remember x = x*s y = y*s
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            x = x*s
            y = y*s
            w = w*s
            h = h*s
                # Mark_boxes
            if x < ML: continue
            if y < MT: continue
            if x > MR: continue
            if y > MB: continue
                # Outside Margins
            cv2.rectangle(baseimg, (x,y), (x+w, y+h), (36, 255, 12), 2)
                # boxc = baseimg[y:y+h, x:x+w]
        cv2.imwrite(file[:-4]+'_'+str(i)+'.png', baseimg)
        
    return "yes"
 
if __name__ == "__main__":
    #   Do something here
    dir_files = [f for f in os.listdir(".") if os.path.isfile(os.path.join(".", f))]
    for file in dir_files: # look at every file in the current directory
        if file.endswith('.pdf'): # if it is a PDF, use it
            text = proc_page(file)

