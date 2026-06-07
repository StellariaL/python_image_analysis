import cv2 as cv
import numpy as np
from collections import defaultdict

def map_to_original(points, y_start, x_start):
    """Map (y,x) point coordinates in a ROI back to the original image"""
    mapped_y = []
    mapped_x=[]
    for (y_roi, x_roi) in points:
        # Map the ROI coordinates back to the original image
        x_orig = x_roi + x_start
        y_orig = y_roi + y_start
        mapped_y.append(y_orig)
        mapped_x.append(x_orig)
    return mapped_y,mapped_x

def opening(image,ker):
    """
    Conducts morphological opening (erosion followed by dilation)
    open_img=opening(img,ker)

    Parameters:
        image: image object
        ker int: kernel size. Features smaller than this will be removed.

    Returns:
        opened_img: image object same size as input
    """
    kernel = np.ones((ker,ker),np.uint8)
    opened_img=cv.morphologyEx(image,cv.MORPH_OPEN,kernel)
    return opened_img

def closing(image,ker):
    """
    Conducts morphological closing (dilation followed by erosion)
    closed_img=closing(img,ker)

    Parameters:
        image: image object
        ker int: kernel size. Features smaller than this will be removed.

    Returns:
        closed_img: image object same size as input
    """
    kernel = np.ones((ker,ker),np.uint8)
    closed_img=cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)
    return closed_img

def average_duplicates(y_coords,x_coords):
    """averages y in paired y,x so that all x are unique"""
    # Dictionary to hold sum and count of y-values for each x
    y_sum = defaultdict(float)
    y_count = defaultdict(int)
    
    # Sum up y-values for each x and count occurrences
    for x, y in zip(x_coords, y_coords):
        y_sum[x] += y
        y_count[x] += 1
    
    # Create unique x-coordinates and averaged y-coordinates
    unique_x_coords = np.array(list(y_sum.keys()))
    averaged_y_coords = np.array([y_sum[x] / y_count[x] for x in unique_x_coords])
    
    return averaged_y_coords, unique_x_coords