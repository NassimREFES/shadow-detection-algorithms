# -*- coding: utf-8 -*-

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import sys
import os
from time import sleep
from threading import Thread
from multiprocessing import Process, Lock, Queue

def main():
    if len(sys.argv) == 2:
        img = cv2.imread(sys.argv[1])

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        seuil, res_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU) # ombre = 0, non_ombre = 1

        res_otsu1 = cv2.bitwise_not(res_otsu) # comme dans l'article | ombre = 1, non_ombre = 0

        # filtre en dehors de l'ombre
        elem_struct = np.ones((3,3), np.uint8)
        fermeture_morph = cv2.morphologyEx(res_otsu, cv2.MORPH_CLOSE, elem_struct)

        # filtre a l'interieur de l'ombre
        median = cv2.medianBlur(fermeture_morph, 5)


        # filtre en dehors de l'ombre
        elem_struct = np.ones((3,3), np.uint8)
        fermeture_morph1 = cv2.morphologyEx(res_otsu1, cv2.MORPH_CLOSE, elem_struct)

        # filtre a l'interieur de l'ombre
        median1 = cv2.medianBlur(fermeture_morph1, 5)

        cv2.imwrite("original.jpg", img)
        cv2.imwrite('otsu.jpg', res_otsu)
        cv2.imwrite('res.jpg', median)
        cv2.imwrite('otsu1.jpg', res_otsu1)
        cv2.imwrite('res1.jpg', median1)

        # while True:
        #     cv2.imshow('img', img)
        #     cv2.imshow('otsu', res_otsu)
        #     cv2.imshow('otsu1', res_otsu1)
        #     cv2.imshow('res', median)
        #     cv2.imshow('res1', median1)

        #     k = cv2.waitKey(33)
        #     if k==27:
        #         break
        #     elif k==-1:
        #         continue
        #     else:
        #         print (k)

if __name__ == '__main__':
    main()