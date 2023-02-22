from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio 
plt.style.use('ggplot')

JUP1 = np.array(Image.open("Mandetory\Supplementary data\Jupiter1.png"))
JUP2 = np.array(Image.open("Mandetory\Supplementary data\Jupiter2.png"))



def analyse_image():
    histogram1, bin_edges1 = np.histogram(JUP1, bins=256, range=(0, 255))
    histogram2, bin_edges2 = np.histogram(JUP2, bins=256, range=(0, 255))
    
    plt.plot(bin_edges1[0:-1], histogram1, color='r')
    plt.show()
    
analyse_image()
