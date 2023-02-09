""" 
Task 1:

The blurred images, will have different histograms. For the image on the left, 
it will have some higher peaks on the far right and left, but also some lower
values in the middle. 
The image on the right will have a higher peak in the middle than the image on the
left, but also high 
peaks on the far left and right.
"""

from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio


""" Task 2 """

img=np.array(Image.open("DIP3E_Original_Images_CH03\Fig0316(1)(top_left).tif"))
img1=np.array(Image.open("DIP3E_Original_Images_CH03\Fig0316(2)(2nd_from_top).tif"))
img2=np.array(Image.open("DIP3E_Original_Images_CH03\Fig0316(3)(third_from_top).tif"))
img3=np.array(Image.open("DIP3E_Original_Images_CH03\Fig0316(4)(bottom_left).tif"))

histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
histogram1, bin_edges1 = np.histogram(img1, bins=256, range=(0, 255))
histogram2, bin_edges2 = np.histogram(img2, bins=256, range=(0, 255))
histogram3, bin_edges3 = np.histogram(img3, bins=256, range=(0, 255))


plt.subplot(2, 4, 5)
plt.plot(bin_edges[0:-1], histogram, color='r')
plt.subplot(2, 4, 1)
plt.imshow(img, cmap='gray')
plt.xticks([])

plt.subplot(2, 4, 6)
plt.plot(bin_edges1[0:-1], histogram1, color='r')
plt.subplot(2, 4, 2)
plt.imshow(img1, cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 7)
plt.plot(bin_edges2[0:-1], histogram2, color='r')
plt.subplot(2, 4, 3)
plt.imshow(img2, cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 8)
plt.plot(bin_edges3[0:-1], histogram3, color='r')
plt.subplot(2, 4, 4)
plt.imshow(img3, cmap='gray')
plt.xticks([])
plt.yticks([])

plt.show()

# b)
# Histrogram equalization

img_eq = cv2.equalizeHist(img)

histogram_eq, bin_edges_eq = np.histogram(img_eq, bins=256, range=(0, 255))

""" plt.subplot(2,1,1)
plt.plot(bin_edges[0:-1], histogram, color='r')
plt.subplot(2,1,2)
plt.plot(bin_edges_eq[0:-1], histogram_eq, color='r')
plt.show() """

""" Task 3 """

# Riemann Zêta function
def zeta(s, N):
    return np.sum(1.0 / np.power(np.arange(1, N+1), s))

# program to calculate scalar product of two functions

def dirac_delta(x):
    return 1 if x == 0 else 0


""" Task 5 """

f = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]

# plot the magnitude and phase of f
fourier = np.fft.fft(f)
fourier_shifted = np.fft.fftshift(fourier)

# Plot for showing the difference between the shifted and unshifted fourier transform
plt.plot(fourier, label = "Fourier")
plt.plot(fourier_shifted, label = "Fourier shifted")
plt.text(0.27, 6, "As we can see, the shifted fourier\n transform has a peak in the \n middle which is the same as \n the original function")
plt.title("shifted and unshifted fourier transform")
plt.legend()
plt.show()

# Plot of the magnitude and phase of the fourier transform
plt.plot(np.abs(fourier_shifted), label = "Magnitude")
plt.plot(np.angle(fourier_shifted), label = "Phase")
plt.title("Magnitude and phase of the fourier transform")
plt.legend()
plt.show()















