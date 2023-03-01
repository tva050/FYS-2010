from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio
import scipy.ndimage as ndi

plt.style.use('ggplot')

JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png')

blue, green, red = cv2.split(JUP2)


def vertical_line_removingfilter(img):
    rows, cols = img.shape[:2]
    crow, ccol = rows//2, cols//2
    img = np.float32(img)
    f_img = np.fft.fft2(img)
    center_shift_img = np.fft.fftshift(f_img)
    magnitude_spectrum_img = np.log(np.abs(center_shift_img))
    center_shift_img[250:crow - 4, ccol - 1:ccol + 1] = 1
    center_shift_img[crow + 4:-250, ccol - 1:ccol + 1] = 1
    img = np.fft.ifftshift(center_shift_img)
    img = np.fft.ifft2(img)
    img = np.real(img)
    img = np.uint8(img)
    return img

blue = vertical_line_removingfilter(blue)
green = vertical_line_removingfilter(green)
red = vertical_line_removingfilter(red)


filtered = cv2.merge((blue, green, red))

plt.subplot(121)
plt.imshow(cv2.cvtColor(JUP2, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Orginal Jupiter2")
plt.subplot(122)
plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Filtered Jupiter2")
plt.show()











