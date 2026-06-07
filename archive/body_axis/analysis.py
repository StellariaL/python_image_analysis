import numpy as np

def compute_curvature(x,y):
    """
    Computes the curvature at each point along the skeleton.
    curv=compute_curvature(x,y)
    
    Parameters:
        x [x1,x2,...]: sorted array of x coordinates
        y [y1,y2,...]: array of y coordinates
    
    Returns:
        curvatures: Array of local curvature values for each x
    """
    # Compute finite differences (forward, central, backward)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Calculate curvature using the formula
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5
    curvatures = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    return curvatures