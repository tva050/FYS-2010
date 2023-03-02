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
______________________________ FOURIER DOMAIN ______________________________
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

def homomorphic_filter(image, gL, gH, c, d0):
    # Convert the input image to the logarithmic scale
    img_log = np.log1p(np.array(image, dtype="float") / 255)

    # Create the high pass filter mask
    M, N, D = image.shape
    x, y = np.meshgrid(np.linspace(-N // 2, N // 2, N), np.linspace(-M // 2, M // 2, M))
    distance = np.sqrt(x**2 + y**2)
    highpass = (1.0 - 1.0 / (1.0 + (d0 / distance)**(2 * c))) * (gH - gL) + gL

    # Apply the high pass filter mask to the image in the frequency domain
    img_fft = np.fft.fftshift(np.fft.fft2(img_log))
    img_fft_filtered = np.zeros_like(img_fft)
    for i in range(D):
        img_fft_filtered[:, :, i] = img_fft[:, :, i] * highpass
    
    img_filtered_log = np.fft.ifft2(np.fft.ifftshift(img_fft_filtered))
    # Convert the filtered image back to the original scale
    img_filtered = np.exp(img_filtered_log) - 1
    img_filtered = np.uint8(img_filtered * 255 / np.max(img_filtered))

    return img_filtered

# Apply the homomorphic filter with the specified parameters
img_filtered = homomorphic_filter(notch_filtered, 3, 0.2, 2.5, 0)


# Display the original and filtered images
plt.subplot(121)
plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
plt.title("Homomorphic Filtered Image")
plt.xticks([])
plt.yticks([])

plt.show()

""" 
_________________________ SPATIAL FILTERS _________________________
"""

# __Median filter__

median_filter = ndi.median_filter(notch_filtered, size=3)
median_filter2 = ndi.median_filter(img_filtered, size=3)

# __Contrast Harmonic Mean filter__
np.seterr(invalid="ignore") # 

def CHM_filter(img, Q):
    img = img.astype(np.float64)
    numirator = img**(Q+1)
    denumirator = img**(Q)
    kernel = np.full(shape = 3, fill_value= 1.0, dtype = np.float64)
    result = cv2.filter2D(numirator, -1, kernel) / cv2.filter2D(denumirator, -1, kernel)
    return result

CHM_filtered_j1 = CHM_filter(median_filter, -2.)
CHM_filtered_j1 = np.uint8(CHM_filtered_j1)

chm_filtered = CHM_filter(median_filter2, -2.)
chm_filtered = np.uint8(chm_filtered)

blue_chm, green_chm, red_chm = cv2.split(chm_filtered)

def contrast_stretching(img):
    img = img.astype(np.float64)
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img

blue_ = contrast_stretching(blue_chm)
green_ = contrast_stretching(green_chm)
red_ = contrast_stretching(red_chm)

contrast_stretching = cv2.merge((blue_, green_, red_))
contrast_stretching = np.uint8(contrast_stretching)


def gamma_correction(img, gamma):
    img = img.astype(np.float64)
    img = img / 255
    img = img ** gamma
    img = img * 255
    return img

gamma_correction = gamma_correction(contrast_stretching, 0.5)
gamma_correction = np.uint8(gamma_correction)

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
plt.title("Org 1")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
plt.title("Notch filtered 1")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(CHM_filtered_j1, cv2.COLOR_BGR2RGB))
plt.title("CHM filtered 1")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
plt.title("org 2")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
plt.title("Homomorphic filtered 2")
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(chm_filtered, cv2.COLOR_BGR2RGB))
plt.title("CHM filtered 2")
plt.xticks([])
plt.yticks([])
plt.show()

plt.imshow(cv2.cvtColor(gamma_correction, cv2.COLOR_BGR2RGB))
plt.title("Gamma correction")
plt.xticks([])
plt.yticks([])
plt.show()




