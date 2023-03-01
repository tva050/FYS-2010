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

orginal = cv2.merge((blue, green, red))

def notch_filter_J2(shape, d0, u_k, v_k): 
    M, N = shape
    H = np.zeros((M, N))
    
    for u in range(0, M):
        for v in range(0, N):
            D_k = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
            D_mk = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
            
            if D_k <= d0 or D_mk <= d0 :
                H[u, v] = 0.0
            else: 
                H[u, v] = 1.0
    return H

def fourier_transform_J2(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    
    H1 = notch_filter_J2(img.shape, 2, 5, 0)

    img = fshift * H1
    img = np.fft.ifftshift(img)
    img = np.fft.ifft2(img)
    img = np.abs(img)
    img = np.uint8(img)
    return img

blue = fourier_transform_J2(blue)
green = fourier_transform_J2(green)
red = fourier_transform_J2(red)

nocth_filtered = cv2.merge((blue, green, red))

plt.subplot(121)
plt.imshow(cv2.cvtColor(orginal, cv2.COLOR_BGR2RGB))
plt.title("Jupiter2")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(cv2.cvtColor(nocth_filtered, cv2.COLOR_BGR2RGB))
plt.title("Filtered")
plt.xticks([])
plt.yticks([])
plt.show()

median_filtered = cv2.medianBlur(nocth_filtered, 3)

plt.subplot(121)
plt.imshow(cv2.cvtColor(nocth_filtered, cv2.COLOR_BGR2RGB))
plt.title("Jupiter2")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
plt.title("Filtered")
plt.xticks([])
plt.yticks([])
plt.show()

np.seterr(invalid="ignore") # To ignore the warning of division by zero
# Contrast Harmonic Mean filter, 
# written with help from https://stackabuse.com/introduction-to-image-processing-in-python-with-opencv/
def CHM_filter(img, Q):
    img = img.astype(np.float64)
    numirator = img**(Q+1)
    denumirator = img**(Q)
    kernel = np.full(shape = 3, fill_value= 1.0, dtype = np.float64)
    result = cv2.filter2D(numirator, -1, kernel) / cv2.filter2D(denumirator, -1, kernel)
    return result

CHM_filtered = CHM_filter(median_filtered, 4)
CHM_filtered = np.uint8(CHM_filtered)

plt.subplot(121)
plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
plt.title("Jupiter2")
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(cv2.cvtColor(CHM_filtered, cv2.COLOR_BGR2RGB))
plt.title("Filtered")
plt.xticks([])
plt.yticks([])
plt.show()