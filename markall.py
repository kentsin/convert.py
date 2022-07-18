#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-18 14:12:11
# @Author  : kentsin (kentsin@gmail.com)
# @Link    : link
# @Version : 0.0.1

from pytesseract import Output
import pytesseract
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pdf2image
poppler_path = r"E:\Program Files (x86)\poppler-22.04.0\Library\bin"

MT = 120
MB = 20
ML = 20
MR = 20

MC = 100  # Maximum objects in a single page. More than that will consider fail
MA = 100  # Minimum object Area. Smaller objects will be ignored

DPI = 300

LANG = "chi_tra+por+eng"
config = r'--psm 6 --oem 3'


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

def load_images(path, dpi=300):
    images = []
    images.extend(list(map(lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR),
                  pdf2image.convert_from_path(path, dpi=dpi, poppler_path=poppler_path), )))
    return images

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
    # print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    # cv2.imwrite("temp/boxes.jpg", newImage)
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


def mark_img(img):
    # base_img = img.copy()

    newImage = img.copy()
    # Deskew
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
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
    # print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    # cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    print(angle)
    if angle!=90.0:
        newImage = img.copy() 
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(
            newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Deskew finished
    base_img = newImage.copy()

    workimage = newImage.copy()

    for s in [2, 3, 4, 5]:
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
        print(len(cnts))
        if len(cnts) < MC: break  # We are satisfied 
    
    #    result = cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    #    return image
    # mark_boxes
    print(s)
    cnts = sorted(cnts, key=lambda y: cv2.boundingRect(y)[1])
    # Now process images, remember x = x*s y = y*s
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x = x*s
        y = y*s
        w = w*s
        h = h*s
        # Mark_boxes
        if x<MR: continue
        if y<MT: continue
        if x>ML: continue
        if y>MB: continue

        cv2.rectangle(base_img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    return base_img
        