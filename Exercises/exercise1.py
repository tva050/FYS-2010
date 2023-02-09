from PIL import Image
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as sio


""" Task 1 """
# a)
# matrix named A with only zero values of size 7×8
A = np.zeros((7,8)) 

# b)
# Replace the first column with value 1, the second with value 2, the third with
# 3, position (5,7) with value 5, and position (6,7) with value 8 in matrix A.
A[0, :] = 1
A[1, :] = 2
A[2, :] = 3
A[5, 7] = 5
A[6, 7] = 8

# c)
# Function wich finds the unique values in a matrix
#print(np.unique(A))

# d)
# Making a matrix of size 8×7 with random values between 0 and 10
B = np.random.randint(0, 10, (8, 7))

# e)
# Multiply matrix A with matrix B using matrix multiplication
#print(np.matmul(A, B))

# f)
# First flip A matrix up-down, then flip it left-right, flip it 90 degrees to the end flip it 37 degrees
A_up_down = np.flipud(A)
A_left_right = np.fliplr(A)
A_flip_90 = np.rot90(A)
A_flip_37 = skimage.transform.rotate(A, 37)
#print(A_up_down)
#print(A_left_right)
#print(A_flip_90)
#print(A_flip_37)
# When we rotate a matrix, it means that we are rotating the image.

# g) 

#sio.savemat('A.mat', {'A': A}) Vet ikke hvordan jeg skal bruge denne

""" Task 2 """

Img=Image.open("Fig0207(a)(gray level band).tif")
#Img.show()


""" Task 3 """
# a)b)

yhea = np.array(Img)

plt.plot(yhea[100])
plt.show()

uniq_val = np.unique(yhea)
print(uniq_val)  #printing the unique values in the image
print(yhea.shape)  #printing the shape of the image

# c)

color_pic = colors.ListedColormap(["green", "red", "orange", "yellow", "pink", "black"])

plt.imshow(yhea, cmap=color_pic)
plt.show()


