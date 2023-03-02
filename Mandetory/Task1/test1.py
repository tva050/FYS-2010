from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio
import scipy.ndimage as ndi

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')

blue, green, red = cv2.split(JUP1)

orginal_jup1 = cv2.merge((blue, green, red))


""" 
___________________________FOURIER DOMAIN____________________________
"""

def notch_filter_J1(shape, d0, u_k, v_k): 
    M, N = shape
    H = np.zeros((M, N))
    
    for u in range(0, M):
        for v in range(0, N):
            D_k = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
            D_mk = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
            horizontal1 = u - M/2 + u_k 
            vertical1 =v - N/2 + v_k
            horizontal2 = u - M/2 - u_k 
            vertical2 = v - N/2 - v_k
        
            if D_k <= d0 or D_mk <= d0 or horizontal1 == 0 or vertical1 == 0 or horizontal2 == 0 or vertical2 == 0:
                H[u, v] = 0.0
            else: 
                H[u, v] = 1.0
    return H

f_J1 = np.fft.fft2(red) 
fshift_J1 = np.fft.fftshift(f_J1)
magnitude_spectrum_J1 = np.log(np.abs(fshift_J1))

H1_J1 = notch_filter_J1(red.shape, 3, 7, 7)

red = fshift_J1 * H1_J1
red = np.fft.ifftshift(red)
red = np.fft.ifft2(red)
red = np.abs(red)
red = np.uint8(red)

notch_filtered = cv2.merge((blue, green, red))
blue_, green_, red_ = cv2.split(notch_filtered)

def homomorphic_filter(img, gL, gH, c, D0):
    img = img.astype(np.float64)
    img = np.log1p(img) # log transform to reduce the effect of dark pixels

    M, N = img.shape
    u, v = np.meshgrid(np.arange(0, N), np.arange(0, M))
    Duv = np.sqrt((u - N/2)**2 + (v - M/2)**2)
    highpass = (gH - gL) * (1 - np.exp(-c * (Duv**2) / (D0**2))) + gL
    #highpass = (1.0 - 1.0 / (1.0 + (D0 / Duv)**(2 * c))) * (gH - gL) + gL 
    
    fft_img = np.fft.fftshift(np.fft.fft2(img)) # fft
    
    filtering_img = fft_img * highpass # filtering image
    filtered_img_log = np.fft.ifft2(np.fft.ifftshift(filtering_img)) # ifft shift
    
    # convert back to the original scale 
    img_filtered = np.exp(filtered_img_log) - 1
    img_filtered = np.real(img_filtered)
    img_filtered = np.uint8(img_filtered * 255 / np.max(img_filtered)) 
    
    return img_filtered
 
blue_profiler = homomorphic_filter (blue_,  1.4, 1.3, 5, 20)
green_profiler = homomorphic_filter(green_, 1.4, 1.3, 5, 20)
red_profiler = homomorphic_filter  (red_,   1.4, 1.3, 5, 20)
    
homomotphic_filtered = cv2.merge((blue_profiler, green_profiler, red_profiler))

plt.subplot(121)
plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
plt.title('Notch filtered')
plt.subplot(122)
plt.imshow(cv2.cvtColor(homomotphic_filtered, cv2.COLOR_BGR2RGB))
plt.title('Homomorphic filtered')
plt.show()


""" 
_________________________SPATIAL DOMAIN_________________________
"""

# __Median filter__

median_filter1 = ndi.median_filter(notch_filtered, size=3)
median_filter2 = ndi.median_filter(homomotphic_filtered, size=3)

# __Contrast Harmonic Mean filter__
np.seterr(invalid="ignore") # 

def CHM_filter(img, Q):
    img = img.astype(np.float64)
    numirator = img**(Q+1)
    denumirator = img**(Q)
    kernel = np.full(shape = 3, fill_value= 1.0, dtype = np.float64)
    result = cv2.filter2D(numirator, -1, kernel) / cv2.filter2D(denumirator, -1, kernel)
    return result

CHM_filtered_j1 = CHM_filter(median_filter1, -3)
CHM_filtered_j1 = np.uint8(CHM_filtered_j1)

chm_filtered = CHM_filter(median_filter2, -3)
chm_filtered = np.uint8(chm_filtered)

blue_chm, green_chm, red_chm = cv2.split(chm_filtered)

def contrast_stretching(img):
    img = img.astype(np.float64)
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img

blue_chm = contrast_stretching(blue_chm)
green_chm = contrast_stretching(green_chm)
red_chm = contrast_stretching(red_chm)

contrast_stretching = cv2.merge((blue_chm, green_chm, red_chm))
contrast_stretching = np.uint8(contrast_stretching)

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
plt.title('Notch filtered 1')
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(CHM_filtered_j1, cv2.COLOR_BGR2RGB))
plt.title('CHM filtered 1')
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(homomotphic_filtered, cv2.COLOR_BGR2RGB))
plt.title('Homomorphic filtered 2')
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(contrast_stretching, cv2.COLOR_BGR2RGB))
plt.title('CHM filtered 2')
plt.show() 



