# Configuration Section for Parameters
config = {
    # parameters for segmenting zona pellucida
    'zp_gauss_kernel': 13, # kernel size for Gaussian blur
    'zp_thresh_value': 75, # threshold value

    # parameters for segmenting embryo mid-axis
    'em_gauss_kernel': 13, # kernel size for Gaussian blur
    'block_size': 51, # block size for adaptive thresholding (smaller = more details)
    'min_width': 100, # width threshold for putative mid-axis

    # parameters for segmenting embryo anterior outline
    'ant_roi_height': 250, # HALF height of ROI
    'ant_roi_width': 800, # width of ROI
    'ant_lo': 4, # floodfill lower limit, Higher value = higher sensitivity = larger filled region
    'ant_up':40, # floodfill upper limit
    'ant_close_ker': 5, # kernel size for closing contour
    'ant_smooth_ker': 9, # kernel size for smoothing contour

    # parameters for segmenting embryo posterior outline
    'post_roi_height': 300, # HALF height of ROI
    'post_roi_width': 800, # width of ROI
    'post_lo': 20, # floodfill lower limit
    'post_up': 20, # floodfill upper limit
    'post_close_ker': 5, # kernel size for closing contour
    'post_smooth_ker': 9, # kernel size for smoothing contour

    # parameters for finding anterior & posterior midlines
    'Windowsize': 10,  # for each x coordinate, midpoint is taken as 1/2 maximum contour height in 2xWindowsize stripe

    # parameters for fitting smooth b-spline
    'spline_smoothing_factor': 70000, # smoothing factor s
}