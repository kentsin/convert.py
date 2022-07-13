#!/usr/bin/env python  # https://blog.python.org/2011/07/python-launcher-for-windows_11.html
# -*- coding: utf-8 -*-

# https://blog.42mate.com/opencv-tesseract-is-a-powerful-combination/
import numpy as np
import pdf2image
import cv2

import matplotlib.pyplot as plt

def convert_pdf_to_image(document, dpi):
    images = []
    images.extend(
        list(
            map(
                lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR), pdf2image.convert_from_path(document, dpi=dpi),
            )
        )
    )
    return images