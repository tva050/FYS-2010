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

""" rows , cols = JUP2.shape[:2]
crow,ccol = rows//2 , cols//2

blue = np.float32(blue)

f_blue1 = np.fft.fft2(blue)
center_shift_blue = np.fft.fftshift(f_blue1)
magnitude_spectrum_blue = np.log(np.abs(center_shift_blue))

center_shift_blue[250:crow - 4, ccol - 1:ccol + 1] = 1
center_shift_blue[crow + 5:-250, ccol - 1:ccol + 1] = 1


blue = np.fft.ifftshift(center_shift_blue)
blue = np.fft.ifft2(blue)
blue = np.real(blue)

blue = np.uint8(blue)

green = np.float32(green)

green = np.fft.fft2(green)
center_shift_green = np.fft.fftshift(green)
magnitude_spectrum_green = np.log(np.abs(center_shift_green))

center_shift_green[250:crow - 4, ccol - 1:ccol + 1] = 1
center_shift_green[crow + 5:-250, ccol - 1:ccol + 1] = 1

green = np.fft.ifftshift(center_shift_green)
green = np.fft.ifft2(green)
green = np.real(green)
green = np.uint8(green)

red = np.float32(red)

red = np.fft.fft2(red)
center_shift_red = np.fft.fftshift(red)
magnitude_spectrum_red = np.log(np.abs(center_shift_red))

center_shift_red[250:crow - 4, ccol - 1:ccol + 1] = 1
center_shift_red[crow + 5:-250, ccol - 1:ccol + 1] = 1

red = np.fft.ifftshift(center_shift_red)
red = np.fft.ifft2(red)
red = np.real(red)
red = np.uint8(red)
noise_removed = 20 * np.log(np.abs(center_shift_red))

plt.imshow(noise_removed, cmap = 'gray')
plt.title('Jupiter2')
plt.xticks([])
plt.yticks([])
plt.show()

notch = cv2.merge((blue, green, red))

plt.imshow(cv2.cvtColor(notch, cv2.COLOR_BGR2RGB))
plt.title("Jupiter2")
plt.xticks([])
plt.yticks([])
plt.show()

median_filter = ndi.median_filter(cv2.merge((blue, green, red)), size=3 )

plt.imshow(cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB))
plt.title("Jupiter2")
plt.xticks([])
plt.yticks([])
plt.show()
 """



""" def vertical_line_removingfilter(img):
    rows, cols = img.shape[:2]
    crow, ccol = rows//2, cols//2
    img = np.float32(img)
    f_img = np.fft.fft2(img)
    center_shift_img = np.fft.fftshift(f_img)
    magnitude_spectrum_img = np.log(np.abs(center_shift_img))
    center_shift_img[:crow - 4, ccol - 4:ccol + 4] = 1
    center_shift_img[crow + 4:, ccol - 4:ccol + 4] = 1
    img = np.fft.ifftshift(center_shift_img)
    img = np.fft.ifft2(img)
    img = np.real(img)
    img = np.uint8(img)
    return img

blue = vertical_line_removingfilter(blue)
green = vertical_line_removingfilter(green)
red = vertical_line_removingfilter(red)

notch = cv2.merge((blue, green, red))
plt.imshow(cv2.cvtColor(notch, cv2.COLOR_BGR2RGB))
plt.title("Jupiter2")
plt.xticks([])
plt.yticks([])
plt.show()


# __Median filter__



median_filter = ndi.median_filter(cv2.merge((blue, green, red)), size=3)

# __Contrast Harmonic Mean filter__
np.seterr(invalid="ignore") # To ignore the warning of division by zero
# Contrast Harmonic Mean filter, 
# written with help from https://stackabuse.com/introduction-to-image-processing-in-python-with-opencv/
def CHM_filter(img, Q):
    img = img.astype(np.float64)
    numirator = img**(Q+1)
    denumirator = (img)**(Q)
    kernel = np.full(shape = 3, fill_value= 1.0, dtype = np.float64)
    result = cv2.filter2D(numirator, -1, kernel) / cv2.filter2D(denumirator, -1, kernel)
    return result

CHM_filtered = CHM_filter(median_filter, -1.5)
CHM_filtered = np.uint8(CHM_filtered)


plt.imshow(cv2.cvtColor(CHM_filtered, cv2.COLOR_BGR2RGB))
plt.title("Jupiter2")
plt.xticks([])
plt.yticks([])
plt.show()
 """
 
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
    plt.imshow(magnitude_spectrum*H1, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Magnitude Spectrum")
    plt.show()
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


