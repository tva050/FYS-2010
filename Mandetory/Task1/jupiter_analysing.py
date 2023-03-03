import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as ndi
import scipy as sc
from scipy.signal import wiener

plt.style.use('ggplot')

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')
JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png')

gray_JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png',0)
blue, green, red = cv2.split(JUP1)


""" ---------------------------- Task 1b ----------------------------"""

def histogram(img):
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.title("Histogram of Jupiter1")
    plt.show()
    
    histogram, bin_edges = np.histogram(JUP2 , bins=256, range=(0, 255))
    plt.bar(bin_edges[0:-1], histogram, width = 0.7)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.title("Histogram of Jupiter2")
    plt.show()
    
def magnitude_spectrum(img): 
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title("Magnitude Spectrum", x=-0.1)
    plt.xticks([])
    plt.yticks([])
    plt.show()

#___________RUN FUNCTIONS___________ 
#histogram(JUP1)
#magnitude_spectrum(red)
#magnitude_spectrum(gray_JUP2)
#___________________________________


""" ---------------------------- Task 1c ----------------------------"""

orginal_jup1 = cv2.merge((blue, green, red))
# __Notch filter__
# Source: https://stackoverflow.com/questions/65483030/notch-reject-filtering-in-python
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

notch_filtered_j1 = cv2.merge((blue, green, red))
n_blue, n_green, n_red = cv2.split(notch_filtered_j1)

# __Plotting__
def plot_notch_filtered_J1img_mag():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = False
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Orginal Jupiter1")
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum_J1*H1_J1, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Eliminated bursts")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(notch_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered Jupiter1")
    plt.show()

# __Median filter__
median_filtered_j1 = ndi.median_filter(cv2.merge((blue, green, red)), size=3)

# __Contrast Harmonic Mean filter__
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

CHM_filtered_j1 = CHM_filter(median_filtered_j1, -2.)
CHM_filtered_j1 = np.uint8(CHM_filtered_j1)

def plot_median_CHM_filtered_J1img():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = False
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(notch_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered Jupiter1")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(median_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Median filtered")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(CHM_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("CHM filtered")
    plt.show()
    
def restored_JUP1_image():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Orginal Jupiter1")
    ax2.imshow(cv2.cvtColor(CHM_filtered_j1, cv2.COLOR_BGR2RGB))
    ax2.set_title("Restored Jupiter1")
    fig.suptitle("$\Longrightarrow$\n$\Longrightarrow$", x = 0.52, y = 0.45, fontsize = 16)
    fig.tight_layout(pad = 3.5)
    
    # remove the x and y ticks
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
    
    
#___________RUN FUNCTIONS___________
#plot_notch_filtered_J1img_mag()
#plot_median_CHM_filtered_J1img()
#restored_JUP1_image()
#___________________________________


""" ---------------------------- Task 1d ----------------------------"""

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
f = np.fft.fft2(blue)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))
H1 = notch_filter_J2(blue.shape, 2, 5, 0)

notch_filtered_j2 = cv2.merge((blue_j2, green_j2, red_j2))

median_filter_j2 = cv2.medianBlur(notch_filtered_j2, 3)

CHM_filtered_j2 = CHM_filter(median_filter_j2, 3)
CHM_filtered_j2 = np.uint8(CHM_filtered_j2)

def plot_notch_filtered_J2img_mag():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = False
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(JUP2, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Orginal Jupiter2")
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum*H1, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Eliminated bursts")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(notch_filtered_j2, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered")
    plt.show()
    
def plot_median_CHM_filtered_J2img():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = False
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(notch_filtered_j2, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered Jupiter2")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(median_filter_j2, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Median filtered")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(CHM_filtered_j2, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("CHM filtered")
    plt.show()

def restored_JUP2_image():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(JUP2, cv2.COLOR_BGR2RGB))
    ax1.set_title("Orginal Jupiter2")
    ax2.imshow(cv2.cvtColor(CHM_filtered_j2, cv2.COLOR_BGR2RGB))
    ax2.set_title("Restored Jupiter2")
    fig.suptitle("$\Longrightarrow$\n$\Longrightarrow$", x = 0.5, y = 0.45, fontsize = 16)
    fig.tight_layout(pad = 3.5)

    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

#___________RUN FUNCTIONS___________
#plot_notch_filtered_J2img_mag()
#plot_median_CHM_filtered_J2img()
#restored_JUP2_image()
#___________________________________


""" ---------------------------- Task 1e ----------------------------"""


""" 
___________________________FOURIER DOMAIN J1____________________________
"""
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

n_blue  = homomorphic_filter(n_blue,  0.95, 1.0, 2, 10)
n_green = homomorphic_filter(n_green, 0.95, 1.0, 2, 10)
n_red   = homomorphic_filter(n_red,   0.95, 1.0, 2, 10)

homomorphic_filtered_j1 = cv2.merge((n_blue, n_green, n_red))

def plot_notchfiltered_homomorphicfiltered_J1img():
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(notch_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.title("Notch filtered")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(homomorphic_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.title("Homomorphic filtered")
    plt.show()

""" 
___________________________SPATIAL DOMAIN J1____________________________
"""
median_filtered_homomorphic_j1 = ndi.median_filter(homomorphic_filtered_j1, size=3)

CHM_filtered_j1_2 = CHM_filter(median_filtered_homomorphic_j1, -2)
CHM_filtered_j1_2 = np.uint8(CHM_filtered_j1_2)

blue_chm, green_chm, red_chm = cv2.split(CHM_filtered_j1_2)

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

def plot_chm_filtered_contstretching_j1():
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(CHM_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.title("CHM filtered")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(contrast_stretching, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.title("Contrast stretching")
    plt.show()

def plot_enhanced_j1():
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(notch_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.title('Notch 1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(CHM_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.title('CHM  1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(orginal_jup1, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(homomorphic_filtered_j1, cv2.COLOR_BGR2RGB))
    plt.title('Homomorphic')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(contrast_stretching, cv2.COLOR_BGR2RGB))
    plt.title('contrast stretching')
    plt.xticks([]), plt.yticks([])
    plt.show()


""" 
___________________________FOURIER DOMAIN J1____________________________
"""





#___________RUN FUNCTIONS___________
#plot_notchfiltered_homomorphicfiltered_J1img()
#plot_chm_filtered_contstretching_j1()
#plot_enhanced_j1() 
#___________________________________

cv2.waitKey(0)  
cv2.destroyAllWindows()
