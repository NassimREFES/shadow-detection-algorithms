# -*- coding: utf-8 -*-

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import sys
import os

np.set_printoptions(threshold=numpy.nan)

def nsvdi(img):
    _index = []

    rows = img.shape[0]
    cols = img.shape[1]

    if len(img.shape) == 3L:
        h, s, v = cv2.split(img)

        for row in range(rows):
            _index.append([])
            for col in range(cols):
                if s[row][col] + v[row][col] == 0.0:
                    _index[row].append(0)
                else:
                    _index[row].append( (s[row][col] - v[row][col]) / (s[row][col] + v[row][col]) )

        return np.array(_index, dtype=np.float64)
    return img

def main():
    if len(sys.argv) == 2:
        img = cv2.imread(sys.argv[1])

        # RGB to HSV
        bgr_to_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bgr_to_hsv = bgr_to_hsv.astype(np.float64)

        cv2.imwrite('hsv.jpg', bgr_to_hsv)

        # normalize HSV in [0, 1]
        cv2.normalize(bgr_to_hsv, bgr_to_hsv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #bgr_to_hsv = bgr_to_hsv / 255
        
        # build index NSVDI
        _index = nsvdi(bgr_to_hsv)

        normlize_index = np.zeros((_index.shape[0], _index.shape[1]))
        cv2.normalize(_index, normlize_index, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normlize_index = np.array(np.round(normlize_index), dtype=np.uint8)
        otsu, res_otsu = cv2.threshold(normlize_index, 0, 255, type=cv2.THRESH_OTSU)

        bgr_to_hsv = bgr_to_hsv.astype(np.uint8)

        cv2.imwrite("original.jpg", img)
        cv2.imwrite('nsvdi.jpg', normlize_index)
        cv2.imwrite('res.jpg', res_otsu)

        #while True:
        #    cv2.imshow('img', img)
        #    cv2.imshow('bgr_to_hsv', bgr_to_hsv)
        #    cv2.imshow('nsvdi', normlize_index)
        #    cv2.imshow('res seg', res_otsu)
        #    k = cv2.waitKey(33)
        #    if k==27:
        #        break
        #    elif k==-1:
        #        continue
        #    else:
        #        print (k)

if __name__ == '__main__':
    main()
