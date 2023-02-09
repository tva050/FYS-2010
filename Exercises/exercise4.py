from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio
import scipy as sc
import pandas as pd

#plt.style.use('ggplot')


""" Task 1 """

IMG = np.array(Image.open("DIP3E_Original_Images_CH03\Fig0335(a)(ckt_board_saltpep_prob_pt05).tif"))
histogram, bin_edges = np.histogram(IMG, bins=256, range=(0, 255))

# Funcion for median filter
def med_filter():
    img_filtered= sc.ndimage.median_filter(IMG, size=3)
    histogram_mf, bin_edges_mf = np.histogram(img_filtered, bins=256, range=(0, 255))
    
    plt.subplot(2,2,1)
    plt.text(0.5, 1.1, 'Original Histogram', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.plot(bin_edges[0:-1], histogram, color='r', label = 'Original Histogram')
    plt.subplot(2,2,2)
    plt.text(0.5, 1.1, 'Original Image', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.imshow(IMG, cmap='gray', label = 'Original Image')
    plt.subplot(2,2,3)
    plt.plot(bin_edges_mf[0:-1], histogram_mf, color='r', label = 'Median Filtered Histogram')
    plt.text(0.5, -0.2, 'Median Filtered Histogram', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.subplot(2,2,4)
    plt.imshow(img_filtered, cmap='gray', label = 'Median Filtered Image')
    plt.text(0.5, -0.2, 'Median Filtered Image', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.show()
    

""" Task 2 """
# a) 



med_filter()


    

    
