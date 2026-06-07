import cv2 as cv
import numpy as np
from .utils import opening

def find_contours(image):
    """
    Find all contours in image and map them
    contours, mask=find_contours(image)

    Parameters:
        image: image object

    Returns:
        contours: array of contours
        mask: binary image same size as input, with all recognized contours filled white
    """
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv.drawContours(mask, contours, -1, (255), thickness=cv.FILLED)
    return contours, mask

def filter_contour_width(contours,min_width):
    """
    Find contours wider than min_width
    filtered=filter_contour_width(c,minw)

    Parameters:
        contours [contour1,contour2,...]: array of contours
        min_width double: minimum width (x direction), exclusive

    Returns:
        array of contours

    """
    large_contours_by_width = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w > min_width:
            large_contours_by_width.append(contour)

    return large_contours_by_width

def filter_contour_ratio(contours,min_ratio):
    """
    Find contours with large aspect ratio (width/height)
    filtered=filter_contour_ratio(c,minr)

    Parameters:
        contours [contour1,contour2,...]: array of contours
        min_ratio double: minimum aspect ratio (x/y direction), exclusive

    Returns:
        array of contours

    """
    filtered = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if (w/h) > min_ratio:
            filtered.append(contour)

    return filtered

def find_centermost_contour(image, contours):
    """
    Find the first contour closest to the center of image
    contour=find_centermost_contour(img,contours)

    Parameters:
        image: image object
        contours: array of contours

    Returns:
        contour object
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Function to calculate the centroid of a contour
    def contour_centroid(contour):
        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0  # In case the contour area is zero
        return (cx, cy)

    # Initialize variables to track the contour closest to the center
    closest_contour = None
    min_distance = float('inf')

    # Loop through each contour
    for contour in contours:
        # Get the centroid of the current contour
        cx, cy = contour_centroid(contour)
        
        # Calculate the Euclidean distance from the image center
        distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        
        # Update the closest contour if this one is closer
        if distance < min_distance:
            closest_contour = contour
            min_distance = distance

    return closest_contour

def find_thinnest_contour(contours):
    """
    Finds the first contour with highest width/height ratio
    contour=find_thinnest_contour(contours)

    Parameters:
        contours [contour1,contour2,...]: array of contours

    Returns:
        contour object
    """
    # initialization
    thinnest_contour = None
    ratio=0
    for i,contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        if (w/h)>ratio:
            ratio=w/h
            thinnest_contour=contour
    return thinnest_contour

def extend(image,mask,direction,lo=5,up=5,ker=7):
    '''
    Extend existing mask by floodfill
    extendedmask=extend(img,msk,dir,lo,up,ker)

    Parameters:
    image: image object
    mask: binary mask to be extended, same shape as image
    direction 'left'/'right': direction of extension
    lo float: boundary of negative difference for floodfill, larger=more pixels included
    up float: boundary of positive difference for floodfill
    ker float: kernel size of opening operation after floodfill, features smaller than ker will be discarded.

    Returns:
    combined: binary mask same size as input
    '''
    floodfill_img=image.copy()
    floodfill_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    coords = np.argwhere(mask > 0)
    if direction=='right':
        point = coords[np.argmax(coords[:, 1])]
        seed_point = tuple(point[::-1])
    elif direction=='left':
        point=coords[np.argmin(coords[:,1])]
        seed_point=tuple(point[::-1])
    cv.floodFill(floodfill_img, floodfill_mask, seed_point, 255, lo, up, flags=4)
    extended_area = floodfill_mask[1:-1, 1:-1]  # Remove the padding
    extended_area = (extended_area > 0).astype(np.uint8) * 255  # Convert to binary
    extended_area=opening(extended_area,ker)
    combined=cv.bitwise_or(mask,extended_area)
    return combined

