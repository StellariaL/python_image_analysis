import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from scipy.interpolate import splrep, splev
from body_axis.utils import average_duplicates


def find_x_midline(contour,windowsize):
    """
    Finds midline along x direction of input contour, with preliminary sliding-window smoothening
    y,x=find_x_midline(contour,w)

    Parameters:
        contour: contour object
        w double: smoothening window size, larger=smoother

    Returns:
        y: array of y coordinates
        x: array of x coordinates
    """
    midline_y = []
    midline_x=[]
    
    # Get the bounding box of the contour
    x_start, y_start, width, height = cv.boundingRect(contour)
    x_end = x_start + width  # Calculate y_end
    
    for x in range(x_start, x_end):  # Iterate through the y-range of the contour
        col_pixels = np.where((contour[:,0,0]>=(x-windowsize))&(contour[:,0,0]<=(x+windowsize)))[0]  # Get non-zero pixels (i.e., contour points)
        if len(col_pixels) > 1:  # Ensure there are contour points in this row
            y_coords=contour[col_pixels, 0, 1]
            y_top = np.min(y_coords)  # topmost y
            y_bot = np.max(y_coords)  # bottommost y
            y_mid = (y_top+y_bot) // 2  # Compute the midpoint
            midline_y.append(y_mid)  # Append the (y_mid, x) coordinates
            midline_x.append(x)

    return midline_y,midline_x

def find_midaxis(image):
    """
    Finds middle axes of white areas on a binary image using skeletonization
    y,x=find_midaxis(img)

    Parameters:
        image: image object

    Returns:
        y: array of y coordinates
        x: array of x coordinates
    """
    skeleton = skeletonize(image)
    # Find the indices of non-zero (white) pixels
    y_coords, x_coords = np.where(skeleton > 0)
    return y_coords,x_coords

def fit_b_spline(y_points, x_points,s):
    """
    Fit b-spline with smoothening factor s, interpolate every unit length
    y,x=fit_b_spline(y_points,x_points,s)
    *when there are multiple points on same x, their average is used

    Parameters:
        y_points [y1,y2,...]: array of y coordinates
        x_points [x1,x2,...]: array of x coordinates
        s float: non-negative smoothing factor, larger=smoother & less restraint on passing through points

    Returns:
        y: array of y coordinates interpolated by b-spline
        x: array of x coordinates, sorted and evenly spaced by unit length
    """
    # Convert lists to numpy arrays
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    y_points_averaged,x_points_unique = average_duplicates(y_points, x_points)

    # Sort the points based on x-coordinates
    sorted_indices = np.argsort(x_points_unique)
    x_points_sorted = x_points_unique[sorted_indices]
    y_points_sorted = y_points_averaged[sorted_indices]

    # Fit a B-spline curve
    tck = splrep(x_points_sorted, y_points_sorted, s=s)

    # Generate smooth points
    x_smooth = np.arange(min(x_points_sorted), max(x_points_sorted))
    y_smooth = splev(x_smooth, tck)

    return y_smooth, x_smooth