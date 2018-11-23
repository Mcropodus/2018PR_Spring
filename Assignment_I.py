# pattern
import numpy as np
import sklearn
import cv2

img = cv2.imread('duck_data.jpg')  # open file
cv2.imshow('image', img)
cv2.waitKey(0)