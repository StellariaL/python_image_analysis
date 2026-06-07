import cv2
import numpy as np
from scipy.interpolate import splrep, splev
from collections import defaultdict

def blur(image,ker):
    blurred_img = cv2.GaussianBlur(image, (ker,ker), 0)
    return blurred_img

def threshold_image(image, threshold_value):
    ret, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def adaptive_threshold(image,block_size):
    adthreshimg = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=block_size, C=-2)
    return adthreshimg

def contour_width_filter(contours,min_width):
    large_contours_by_width = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_width:
            large_contours_by_width.append(contour)

    return large_contours_by_width

def find_contour_closest_to_center(image, contours):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Function to calculate the centroid of a contour
    def contour_centroid(contour):
        M = cv2.moments(contour)
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

def find_outline(image,direction,startpoint,roi_height,roi_width,lo,up,close_ker,smooth_ker):
    if direction=='ant':
        roi = image[(startpoint[1]-roi_height):(startpoint[1]+roi_height), (startpoint[0]-roi_width):startpoint[0]]
        seed=((roi_width-1),roi_height)
    else:
        roi = image[(startpoint[1]-roi_height):(startpoint[1]+roi_height), startpoint[0]:(startpoint[0]+roi_width)]
        seed=(0,roi_height)

    mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), np.uint8)
    cv2.floodFill(roi, mask, startpoint, newVal=255, loDiff=(lo,lo,lo), upDiff=(up,up,up),flags=cv2.FLOODFILL_FIXED_RANGE)

    # smooth out floodfill mask
    kernel = np.ones((close_ker,close_ker), np.uint8)
    # fill holes
    closed_mask = cv2.morphologyEx(mask[1:-1, 1:-1], cv2.MORPH_CLOSE, kernel)
    # blur edges
    blurred_mask = cv2.GaussianBlur(closed_mask, (smooth_ker,smooth_ker),0)

    # find largest contour
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(contours, key=cv2.contourArea)

    return outline

def find_midline(contour,windowsize):
    midline_points = []
    
    # Get the bounding box of the contour
    x_start, y_start, width, height = cv2.boundingRect(contour)
    x_end = x_start + width  # Calculate y_end
    
    for x in range(x_start, x_end):  # Iterate through the y-range of the contour
        col_pixels = np.where((contour[:,0,0]>=(x-windowsize))&(contour[:,0,0]<=(x+windowsize)))[0]  # Get non-zero pixels (i.e., contour points)
        if len(col_pixels) > 1:  # Ensure there are contour points in this row
            y_coords=contour[col_pixels, 0, 1]
            y_top = np.min(y_coords)  # topmost y
            y_bot = np.max(y_coords)  # bottommost x
            y_mid = (y_top+y_bot) // 2  # Compute the midpoint
            midline_points.append((y_mid, x))  # Append the (y_mid, x) coordinates
    
    return midline_points

def map_to_original(midline_points, x_start, y_start):
    mapped_points = []
    for (y_roi, x_roi) in midline_points:
        # Map the ROI coordinates back to the original image
        x_orig = x_roi + x_start
        y_orig = y_roi + y_start
        mapped_points.append((y_orig, x_orig))
    return mapped_points

def map_midline(outline,option,startpoint,roi_height,roi_width,windowsize):
    midline=find_midline(outline,windowsize)
    if option=='ant':
        mapped=map_to_original(midline, startpoint[0]-roi_width, startpoint[1]-roi_height)
    else:
        mapped=map_to_original(midline, startpoint[0], startpoint[1]-roi_height)

def average_duplicates(x_coords, y_coords):
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
    
    return unique_x_coords, averaged_y_coords

def fit_b_spline(x_points, y_points,s):
    # Convert lists to numpy arrays
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    x_points_unique, y_points_averaged = average_duplicates(x_points, y_points)

    # Sort the points based on x-coordinates
    sorted_indices = np.argsort(x_points_unique)
    x_points_sorted = x_points_unique[sorted_indices]
    y_points_sorted = y_points_averaged[sorted_indices]

    # Fit a B-spline curve
    tck = splrep(x_points_sorted, y_points_sorted, s=s)

    # Generate smooth points
    x_smooth = np.linspace(min(x_points_sorted), max(x_points_sorted), 1000)
    y_smooth = splev(x_smooth, tck)

    return x_smooth, y_smooth

def find_axis(image, config):
    # Step 0: Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: segment zona pellucida
    # Gaussian Blur
    blurred_img = blur(gray_img,config['zp_gauss_kernel'])
    # Thresholding
    zp_thresh=threshold_image(blurred_img,config['zp_thresh_value'])
    # Find contours
    contours, _ = cv2.findContours(zp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assume largest contour correspond to zona pellucida
    largest_contour = max(contours, key=cv2.contourArea)
    # Create mask
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    # Apply mask to image to remove anything outside
    masked_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    

    # Step 2: segment middle part of body axis
    # Gaussian blur
    blurred_masked_img=blur(masked_img,config['em_gauss_kernel'])
    # Adaptive thresholding
    ad_thresh=adaptive_threshold(blurred_masked_img,config['block_size'])
    # Find contours
    contours, _ = cv2.findContours(ad_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours on width
    large_contours=contour_width_filter(contours,config['min_width'])
    # Sort contours by area to discard largest contour (zona pellucida)
    large_contours_sorted = sorted(large_contours, key=cv2.contourArea, reverse=True)[1:]
    midaxis_contour=find_contour_closest_to_center(image,large_contours_sorted)
    midaxis=find_midline(midaxis_contour,windowsize=config['Windowsize'])

    # Step 3: find anterior part of body axis
    # Find anterior end of middle part
    leftmost_point = tuple(midaxis_contour[midaxis_contour[:, :, 0].argmin()][0])
    # Find anterior outline
    ant_outline=find_outline(gray_img,
                             direction='ant',
                             startpoint=leftmost_point,
                             roi_height=config['ant_roi_height'],roi_width=config['ant_roi_width'],
                             lo=config['ant_lo'],up=config['ant_up'],
                             close_ker=config['ant_close_ker'],smooth_ker=config['ant_smooth_ker'])
    # Find anterior midline
    ant_mapped=map_midline(ant_outline,'ant',startpoint=leftmost_point,roi_height=config['ant_roi_height'],roi_width=config['ant_roi_width'],windowsize=config['Windowsize'])

    # Step 4: find posterior part of body axis
    # Find posterior end of middle part
    rightmost_point = tuple(midaxis_contour[midaxis_contour[:, :, 0].argmax()][0])
    # Find anterior outline
    post_outline=find_outline(gray_img,
                             direction='pos',
                             startpoint=rightmost_point,
                             roi_height=config['post_roi_height'],roi_width=config['post_roi_width'],
                             lo=config['post_lo'],up=config['post_up'],
                             close_ker=config['post_close_ker'],smooth_ker=config['post_smooth_ker'])
    # Find anterior midline
    post_mapped=map_midline(post_outline,'pos',startpoint=rightmost_point,roi_height=config['post_roi_height'],roi_width=config['post_roi_width'],windowsize=config['Windowsize'])

    # Step 5: assemble and fit into smooth b-spline curve
    whole_midline=ant_mapped+midaxis+post_mapped
    y_coords, x_coords = zip(*whole_midline)
    x_smooth, y_smooth = fit_b_spline(x_coords, y_coords,s=config['spline_smoothing_factor'])

    return x_smooth, y_smooth

# Main processing routine
if __name__ == "__main__":
    # Load the image (replace 'your_image.jpg' with the actual path)
    image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Process the image with the current configuration
    axis_x,axis_y = find_axis(image, config)
    
    # Show the result
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray')
    plt.plot(axis_x, axis_y, '-r', lw=2)
    plt.show()