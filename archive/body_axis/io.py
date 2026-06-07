import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_image(filepath):
    """
    Load an image from a file.
    
    Parameters:
        filepath "path\\to\\image"
    
    Returns:
        image object img
    """
    return cv.imread(filepath, cv.IMREAD_GRAYSCALE)

def show_images(images,titles=None):
    """
    Show images in tiled subplots with titles.
    
    Parameters:
        images [img1,img2,...]: list of image objects.
        titles ["title1","title2",...]: array of titles.
    
    Returns:
        matplotlib image window
    """
    n = len(images)  # Number of images
    if titles is None:
        titles = [None] * n
    elif len(titles)<n:
        titles = titles + [None] * (n - len(titles))
    # Calculate the number of rows and columns needed
    cols = int(np.ceil(np.sqrt(n)))  # Approximate square layout
    rows = int(np.ceil(n / cols))
    
    # Create a figure and axes with a dynamic layout
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    for i in range(n):
        plt.subplot(rows,cols,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i]), plt.xticks([]), plt.yticks([])
    
    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def write_coords(x,y,fname,type='.csv'):
    dataframe=pd.DataFrame({'x':x,'y':y})
    name=fname+type
    if type=='.csv':
        dataframe.to_csv(name,index=False,sep=',')
    elif type=='.xlsx':
        dataframe.to_excel(name,sheet_name='Sheet1',index=False)
    else:
        print('wrong file type for writing coords')
        return 1
    return 0