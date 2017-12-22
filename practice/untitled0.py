# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:38:41 2017

@author: Haoyu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Read in an image and grayscale it
image = mpimg.imread('color-shadow-example.jpg')


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = img
    if orient=='x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    if orient=='y':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    grad_binary = sxbinary # Remove this line
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(sobelx*sobelx+sobely*sobely)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    #scaled_grad_dir = np.uint8(255*grad_dir/np.max(grad_dir))
    scaled_grad_dir = grad_dir
    binary_output = np.zeros_like(scaled_grad_dir)
    binary_output[(scaled_grad_dir >= thresh[0]) & (scaled_grad_dir <= thresh[1])] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    #enalbe the following line for .png images
    #s = np.uint8(255*s/np.max(s))
    binary_output = np.zeros_like(s)
    binary_output[(s>thresh[0])&(s<=thresh[1])] = 1
    return binary_output

# Choose a Sobel kernel size
ksize = 5 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
image_r = image[:,:,0]
gradx = abs_sobel_thresh(image_r, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
s_binary = hls_select(image,thresh=(170,255))
#stack. first will be green, second will be blue
color_binary = np.dstack(( np.zeros_like(gradx), gradx, s_binary)) * 255

combined = np.zeros_like(s_binary)
combined[(s_binary==1)|(gradx==1)] = 1

# Plot the result
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(gradx, cmap='gray')
ax2.set_title('Gradient Thresh', fontsize=20)
ax3.imshow(s_binary, cmap='gray')
ax3.set_title('S Channel Thresh', fontsize=20)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 9))
f.tight_layout()
ax1.imshow(color_binary)
ax1.set_title('color_binary', fontsize=20)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Combined S channel & gradient', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

histogram = np.sum(combined[combined.shape[0]//2:,:], axis=0)
plt.plot(histogram)