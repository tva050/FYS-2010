from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio
import scipy.ndimage as ndi
from PIL import Image

JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png') 

blue, green, red = cv2.split(JUP2)

orginal_jup2 = cv2.merge((blue, green, red))

blue_j2, green_j2, red_j2 = cv2.split(JUP2)


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

blue_j2 = fourier_transform_J2(blue_j2)
green_j2 = fourier_transform_J2(green_j2)
red_j2 = fourier_transform_J2(red_j2)
#-------------
f = np.fft.fft2(blue_j2)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))
H1 = notch_filter_J2(blue.shape, 2, 5, 0)
#-------------

notch_filtered_j2 = cv2.merge((blue_j2, green_j2, red_j2))
n_bluej2, n_greenj2, n_redj2 = cv2.split(notch_filtered_j2)
def homomorphic_filter(img, gL, gH, c, D0):
    img = img.astype(np.float64)
    img = np.log1p(img) # log transform to reduce the effect of dark pixels

    M, N = img.shape
    u, v = np.meshgrid(np.arange(0, N), np.arange(0, M))
    Duv = np.sqrt((u - N/2)**2 + (v - M/2)**2)
    highpass = (gH - gL) * (1 - np.exp(-(c * Duv**2) / (D0**2))) + gL       # GHPF (Gaussian High Pass Filter)
    #highpass = (1.0 - 1.0 / (1.0 + (D0 / Duv)**(2 * c))) * (gH - gL) + gL  # BHPF (Butterworth High Pass Filter)
    
    fft_img = np.fft.fftshift(np.fft.fft2(img)) # fft  
    filtering_img = fft_img * highpass # filtering image
    filtered_img_log = np.fft.ifft2(np.fft.ifftshift(filtering_img)) # ifft shift
    
    # convert back to the original scale 
    img_filtered = np.exp(filtered_img_log) - 1
    img_filtered = np.real(img_filtered)
    img_filtered = np.uint8(img_filtered * 255 / np.max(img_filtered)) 
    
    return img_filtered


n_bluej2  = homomorphic_filter(n_bluej2,  3., 0.2, 5, 300)
n_greenj2 = homomorphic_filter(n_greenj2, 3., 0.2, 5, 300)
n_redj2   = homomorphic_filter(n_redj2,   3., 0.2, 5, 300)

homomorphic_filtered_j2 = cv2.merge((n_bluej2, n_greenj2, n_redj2))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(notch_filtered_j2, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.title("Notch filtered")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(homomorphic_filtered_j2, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.title("Homomorphic filtered")
plt.show()

median_filter_j2 = cv2.medianBlur(homomorphic_filtered_j2, 3)

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


CHM_filtered_j2 = CHM_filter(median_filter_j2, 3)
CHM_filtered_j2 = np.uint8(CHM_filtered_j2)
blue, green, red = cv2.split(CHM_filtered_j2)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(median_filter_j2, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.title("Median filtered")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(CHM_filtered_j2, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.title("CHM filtered")
plt.show()

def contrast_stretching(img):
    img = img.astype(np.float64)
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img

contrast_stretching_j2 = contrast_stretching(CHM_filtered_j2)
contrast_stretching_j2 = np.uint8(contrast_stretching_j2)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(CHM_filtered_j2, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.title("CHM filtered")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(contrast_stretching_j2, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])
plt.title("Contrast stretching")
plt.show()


