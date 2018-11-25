# pattern assignment
import numpy as np
from sklearn import naive_bayes
from sklearn import model_selection as ms
import cv2
import matplotlib.pyplot as plt
import scipy

img = cv2.imread('test_duck.jpg')  # open file
non_duck_img = cv2.imread('non_duck.jpg')  # the data without duck

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RBG to gray
non_duck_img = cv2.cvtColor(non_duck_img, cv2.COLOR_BGR2GRAY)

img_pixel = np.array(img)
n_img_pixel = np.array(non_duck_img)

pixel = img.shape
height = pixel[0]
width = pixel[1]
non_duck_pixel = non_duck_img.shape
n_height = non_duck_pixel[0]
n_width = non_duck_pixel[1]

# remove the background
array = []  # save the label data
for i in range(height):
    for j in range(width):
        if img_pixel[i][j] == 0:
            array.append(0)
        else:
            array.append(1)
            continue

n_array = []  # remove the background without duck
for i in range(n_height):
    for j in range(n_width):
        if n_img_pixel[i][j] < 225:
            n_array.append(0)
        else:
            n_array.append(0)
            continue


ori_img = cv2.imread('test_img.jpg')
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
ori_img = np.reshape(ori_img, (height*width, 1))
ori_n_img = cv2.imread('non_duck.jpg')
ori_n_img = cv2.cvtColor(ori_n_img, cv2.COLOR_BGR2GRAY)
ori_n_img = np.reshape(ori_n_img, (n_height*n_width, 1))

x_train, x_test, y_train, y_test = ms.train_test_split(
    ori_img, array, test_size=0.1
)
model_naive = naive_bayes.GaussianNB()  # Gaussian model
model_naive.fit(x_train, y_train)
# model_naive.partial_fit(ori_n_img, n_array)
print('scoring the naive Bayes : ', model_naive.score(x_test, y_test))

sigma_d = model_naive.sigma_[0] ** 0.5
sigma_n = model_naive.sigma_[1] ** 0.5
mu_d = model_naive.theta_[0]
mu_n = model_naive.theta_[1]
# print(sigma, '\n', mu)
x_d = np.linspace(mu_d - 3 * sigma_d, mu_d + 3 * sigma_d, 100)
plt.plot(x_d, scipy.stats.norm.pdf(x_d, mu_d, sigma_d))
x_n = np.linspace(mu_n - 3 * sigma_n, mu_n + 3 * sigma_n, 100)
plt.plot(x_n, scipy.stats.norm.pdf(x_n, mu_n, sigma_n))
plt.show()  # Gaussian curve


# predict
test = cv2.imread('full_duck.jpg')
test = np.array(test)
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
Pixel = test.shape
t_height = Pixel[0]
t_width = Pixel[1]
test = np.reshape(test, (t_height*t_width, 1))
y_pred = model_naive.predict(test)
# for i in range(t_height*t_width):
#     if y_pred[i] == 1:
#         print(1)
result = np.reshape(y_pred, (t_height, t_width))
# print(result)
for i in range(t_height):
    for j in range(t_width):
        if result[i][j] > 0:
            result[i][j] = 255
        else:
            continue
cv2.imwrite('new_result.jpg', result)
# cv2.imshow('result', result)
# cv2.waitKey(0)
