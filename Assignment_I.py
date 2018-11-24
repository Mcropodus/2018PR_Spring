# pattern assignment
import numpy as np
import sklearn
import cv2

img = cv2.imread('duck_data.jpg')  # open file
# cv2.imshow('image', img)
# cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RBG to gray
# cv2.imshow('image', img)
# cv2.waitKey(0)

img_pixel = np.array(img)
# print(img_pixel)

pixel = img.shape
height = pixel[0]
width = pixel[1]
# print(x, y)

# remove the background
array = []  # save the data
for i in range(height):
    for j in range(width):
        if img_pixel[i][j] < 230:
            img_pixel[i][j] = 0
            array.append(0)
        else:
            array.append(1)
            continue

img = img_pixel.reshape(img_pixel.shape[0], img_pixel.shape[1])
cv2.imwrite('BackgroundRemove.jpg', img)
cv2.imshow('image', img)
cv2.waitKey(0)
array = np.reshape(array, (height, width))
print(array)


# array = []
# for i in range(height):
#     for j in range(width):
#         if img_pixel[i][j] == 0
#             array.append(0)
#         else:
#             array.append(1)
#
# print(array)

