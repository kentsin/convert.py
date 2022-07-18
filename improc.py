#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-16 05:46:59
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

# https://www.youtube.com/watch?v=ADV-AjAXHdc
# https://blog.42mate.com/opencv-tesseract-is-a-powerful-combination/

# poppler   https://github.com/oschwartz10612/poppler-windows/releases/
# pdf2image pip install pdf2image
from binhex import BinHex
from os import symlink
from pytesseract import Output
import pytesseract
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

# parameters for tesseract
LANG = "chi_tra+por+eng"
config = r'--psm 6 --oem 3'
# d = pytesseract.image_to_data(image, lang=LANG, config=config, output_type=Output.DICT)


def load_images(path, dpi=300):
    images = []
    images.extend(list(map(lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR),
                  pdf2image.convert_from_path(path, dpi=dpi, poppler_path=poppler_path), )))
    return images

# https://stackoverflow.com/questions/28816046/
# displaying-different-images-with-actual-size-in-matplotlib-subplot


def display(image):
    dpi = 80

    height, width = image.shape[:2]

    # what size does the figurer need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # hide spines, ticks, etc.
    ax.axis('off')

    # display the image
    ax.imshow(image, cmap='gray')

    plt.show()

# invert


def invert_image(img):
    return cv2.bitwise_not(img)

# Rescaling

# Binarization


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# thresh, im_bw = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
#

# noise removal


def noise_removal(image):
    kernal = np.ones((1, 1), np.unit8)
    image = cv2.dilate(image, kernal, iterations=1)
    kernel = np.ones((1, 1), np.unit8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


# Dilation adn Erosion
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.unit8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.unit8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

# Rotation / Deskewing
# https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df


def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
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
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center


def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

# Remove Borders


def remove_borders(image):
    contours, heiarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

# Missing Borders
# Tesseract is trained with images with border
# color = [255, 255, 255]
# top, bottom, left, right = [150]*4
# image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


# Transparency / alpha Channel


# Noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)
# noise = remove_noise(grey)

# Thresholding


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# thresh = thresholding(gray)

# Erosion (thin_font)


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# Morphology


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# opening = opening(gray)

# Template matching


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
# match = match_template(gray, gray)

# find contours


def find_contours(image):
    workimage = image.copy()
    # some doc need resize to make bounding boxes 
    t = 10 # Max of boxes in a page
    BW = 50 # Min of width of a box consider useful
    BH = 50 # Min of height of a box consider useful
    XM = 80 # margin of x : boxes outside this is ignored
    YM = 120 # Margin of y : boxes outside this is ignored
    for s in [1, 2, 4]:
        w = int(workimage.shape[1] / s)
        h = int(workimage.shape[0] / s)
        dim = (w, h)
        resized = cv2.resize(workimage, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0) # adjustable
        thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13)) # adjustable
        dilate = cv2.dilate(thresh, kernal, iterations=1) # iterations adjustable?
    #    cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts)==2 else cnts[1]

        if len(cnts) < t : break  # We are satisfied 
    
    #    result = cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    #    return image
    # mark_boxes
    cnts = sorted(cnts, key=lambda y: cv2.boundingRect(y)[1])
    # Now process images, remember x = x*s y = y*s
    x = x*s
    y = y*s
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Mark_boxes
        if w > BW and h > BH:
            cv2.rectangle(image, (x,y), (x+w, y+h), (36, 255, 12), 2)
        boxc = image[y:y+h, x:x+w]


def mark_boxes(image):
    h, w, c = image.shape
    boxes = pytesseract.image_to_box(image)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(image, int(
            b[1]), h-int(b[2]), int(b[3]), h-int(b[4]), (0, 255, 0), 2)


def mark_boxes_dict(image, d):
    keys = list(d.keys())
    n = len(d['text'])
    for i in range(n):
        if float(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top']
                            [i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
