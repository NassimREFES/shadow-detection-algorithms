#! /usr/bin/env python

import sys
from imageio import imread
from matplotlib import pyplot
import cv2

from SRM import SRM

import matplotlib.image as mpimg

from arevalo_algo_1 import *

q = int(sys.argv[1])
im = imread(sys.argv[2])

img = cv2.imread(sys.argv[2])

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_img_w = gray_img.shape[0]
gray_img_h = gray_img.shape[1]

seuil, res_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)

# -----------------------------------------
# --------- Pre processing stage ----------
# -----------------------------------------

res = bgr_to_c1c2c3(img)
bgr_to_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# C3 lisse par un noyau 3x3
noyau = np.ones((3,3), dtype=np.float32) / 9
c1, c2, c3 = cv2.split(res)
c3_lisse = cv2.filter2D(c3, -1, noyau)

# gradient Sobel 3x3
h, s, v = cv2.split(bgr_to_hsv)

# gradients horizontaux de V
v_sobelx = cv2.Sobel(v, cv2.CV_64F, dx=1, dy=0, ksize=3) 

# gradients verticaux de V
v_sobely = cv2.Sobel(v, cv2.CV_64F, dx=0, dy=1, ksize=3)

# amplitude de Gx | Gy
v_sobelxy = np.round(np.sqrt(v_sobelx**2 + v_sobely**2))
v_sobelxy = v_sobelxy.astype(np.uint8)

ssx = bgr_to_hsv.copy()
ssx[:, :, 0] = 0
ssx[:, :, 2] = 0

vvx = bgr_to_hsv.copy()
vvx[:, :, 0] = 0
vvx[:, :, 1] = 0

cv2.imwrite(str(sys.argv[3])+"_hsv.jpg", bgr_to_hsv)
cv2.imwrite(str(sys.argv[3])+"_s.jpg", ssx)
cv2.imwrite(str(sys.argv[3])+"_v.jpg", vvx)
cv2.imwrite(str(sys.argv[3])+"_amplitude_v.jpg", v_sobelxy)
cv2.imwrite(str(sys.argv[3])+"_c1c2c3.jpg", res)
cv2.imwrite(str(sys.argv[3])+"_c3.jpg", c3)
cv2.imwrite(str(sys.argv[3])+"_c3_lisse.jpg", c3_lisse)

ML = get_maxima_locaux(v_sobelxy, v_sobely, v_sobelx)

# -----------------------------------------
# ------- Shadow detection stage ----------
# -----------------------------------------

# la moyenne de l ensemble de l image c3
c3_lisse = c3_lisse.astype(np.float64)
rows = c3_lisse.shape[0]
cols = c3_lisse.shape[1]
c3_moy = sum(sum(c3_lisse))
c3_moy = c3_moy / (rows*cols)
c3_lisse = c3_lisse.astype(np.uint8)

# la selection des seeds
regions, seeds, seeds_window, Tv = selection_seeds(c3_lisse, c3_moy, v, s, [5, 5], ML)

srm = SRM(im, seeds, q, seuil/2)
segmented, seeds_res, fermeture_morph = srm.run()

mpimg.imsave(str(sys.argv[3])+"_original.jpg", im)
mpimg.imsave(str(sys.argv[3])+"_segmented.jpg", segmented/256)
cv2.imwrite(str(sys.argv[3])+"_seeds.jpg", seeds)
cv2.imwrite(str(sys.argv[3])+"_seeds_window.jpg", seeds_window*255)
cv2.imwrite(str(sys.argv[3])+"_fermeture_morph.jpg", fermeture_morph)





