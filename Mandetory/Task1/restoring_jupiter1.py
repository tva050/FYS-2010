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

notch_filtered = cv2.merge((blue, green, red))

# __Median filter__
median_filter = ndi.median_filter(cv2.merge((blue, green, red)), size=3)

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

CHM_filtered_j1 = CHM_filter(median_filter, -2.)
CHM_filtered_j1 = np.uint8(CHM_filtered_j1)


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
    plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered Jupiter1")
    plt.show()

def plot_median_CHM_filtered_J1img():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = False
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.title("Notch filtered Jupiter1")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB))
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
    
    



