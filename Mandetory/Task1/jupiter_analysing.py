import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as ndi
import scipy as sc

plt.style.use('ggplot')

JUP1 = cv2.imread('Mandetory\Supplementary data\Jupiter1.png')
JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png')

gray_JUP2 = cv2.imread('Mandetory\Supplementary data\Jupiter2.png',0)
blue, green, red = cv2.split(JUP1)


""" Task 1b """

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

#histogram(JUP1)
magnitude_spectrum(red)
#magnitude_spectrum(gray_JUP2)

""" Task 1c """
orginal = cv2.merge((blue, green, red))

# Source: https://stackoverflow.com/questions/65483030/notch-reject-filtering-in-python
def notch_filter(shape, d0, u_k, v_k): 
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

f = np.fft.fft2(red) 
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift))

H1 = notch_filter(red.shape, 3, 7, 7)

red = fshift * H1
red = np.fft.ifftshift(red)
red = np.fft.ifft2(red)
red = np.abs(red)

red = np.uint8(red)

plt.imshow(magnitude_spectrum*H1, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

notch_filtered = cv2.merge((blue, green, red))

blue = ndi.median_filter(blue, size = 3)
#green = ndi.median_filter(green, size = 5)

""" def CHM_operator(roi,q):
    roi = roi.astype(np.float64) #Copy of the roi
    dividend= np.mean((roi)**(q+1)) # 
    divisor=np.mean((roi)**(q))
    quo= dividend/divisor
    return quo
def CHM_Algo(image,q):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            a=i-1
            b=j-1
            new_image[a,b] = CHM_operator(image[a:a+3,b:b+3],q)
    
    i_min=np.min(image) # min pixel
    i_max=np.max(image) # max pixel
    new_image = (new_image-i_min) * (255/i_max)
    new_image = new_image.astype(np.uint8)
    return new_image """

def CHM_filter(img, q):
    

median_filter = cv2.merge((blue, green, red))

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = False
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(orginal, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Orginal Jupiter1")
plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum*H1, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title("Eliminated bursts")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Notch filtered Jupiter1")
plt.show()

plt.subplot(121)
plt.imshow(cv2.cvtColor(orginal, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Orginal Jupiter1")
plt.subplot(122)
plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Notch filtered Jupiter1")


plt.subplot(121)
plt.imshow(cv2.cvtColor(notch_filtered, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Notch filtered Jupiter1")
plt.subplot(122)
plt.imshow(cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.title("Median filtered Jupiter1")
plt.show()


cv2.waitKey(0)  