from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio

plt.style.use('ggplot')

img=np.array(Image.open("Mandetory\Supplementary data\Jupiter1.png"))

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')


histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))

red_color = cv2.calcHist([JUP1], [2], None, [256], [0, 256])

plt.hist(img.ravel(),256,[0,256]); plt.show()
plt.plot(red_color, color = 'r')
plt.show()

""" f = np.fft.fft2()
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.show() """



img = cv2.cvtColor(JUP1, cv2.COLOR_BGR2RGB)
red_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
green_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
blue_hist = cv2.calcHist([img], [2], None, [256], [0, 255])

plt.plot(red_hist, color = 'r')
plt.plot(green_hist, color = 'g')
plt.plot(blue_hist, color = 'b')
plt.show()