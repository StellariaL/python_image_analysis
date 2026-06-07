import cv2 as cv
import numpy as np

def blur(image,kernel_size):
    """
    Apply Gaussian blur to the image.
    blurred_img=blur(img,k)
    
    Parameters:
        image: image object.
        kernel_size int: kernel size, larger->more blurry
    
    Returns:
        image object same size as input
    """
    if kernel_size//2==0:
        kernel_size+=1
    blurred_img = cv.GaussianBlur(image, (kernel_size,kernel_size), 0)
    return blurred_img

def global_threshold(image, threshold_value):
    """
    Apply global binary thresholding
    Can segment outline of zona pellucida
    th=global_threshold(img,t)
    
    Parameters:
        image: image object
        threshold_value double: 0~255 threshold value, <=/0, >/255
    
    Returns:
        binary image object same size as input
    """
    ret, thresh = cv.threshold(image, threshold_value, 255, cv.THRESH_BINARY)
    return thresh

def adaptive_threshold(image,block_size=51,C=-2):
    """
    Apply adaptive thresholding
    Can segment notochord (and some somites) of embryo
    th=adaptive_threshold(img,b)

    Parameters:
        image: image object
        block_size int: neighbourhood size, larger=less details
        C double: default = -2, makes embryo 1/background 0
    
    Returns:
        binary image object same size as input
    """
    adthreshimg = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=block_size, C=C)
    return adthreshimg

def otsu_threshold(image):
    """
    Apply otsu thresholding
    Can segment outline of embryo
    th=otsu_threshold(img)

    Parameters:
        image: image object

    Returns:
        binary image object same size as input
    """
    ret3,th = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th

