import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def plot_feature(ax,x,y, values,lim=None,colour_lim=None,title=None):
    """
    Visualize a feature value as colour overlaid on coordinates.

    Parameters:
        ax: matplotlib axes object
        x [x1,x2,...]: array of x coordinates
        y [y1,y2,...]: array of y coordinates
        values [f1,f2,...]: array of values corresponding to each coordinate
        lim [xmax,ymax]: limits of x and y axis. If given, axis limit will be set as 0-x/ymax.
        colour_lim [min,max]: limits of colour mapping.
        title "title": default none, title of plot

    Returns:
        matplotlib image window
    """
    if colour_lim:
        vmin=colour_lim[0]
        vmax=colour_lim[1]
    else:
        vmin=np.min(values)
        vmax=np.max(values)
    preview=ax.scatter(x, y, c=values, cmap='coolwarm', s=5,vmin=vmin,vmax=vmax)
    cbar=plt.colorbar(preview,ax=ax)
    if title:
        ax.set_title(title)
    if lim:
        ax.set_xlim(0,lim[0])
        ax.set_ylim(0,lim[1])
    ax.invert_yaxis()
    return ax

def summarize_feature(values):
    """
    Summarizes a feature's mean, std and max for analysis and comparison.
    
    Parameters:
        values [f1,f2,...]: Array of feature values.
    
    Returns:
        summary (dict): mean, std and max(absolute) of curvature.
    """
    summary = {
        'mean': np.mean(values),
        'std': np.std(values),
        'absmax': np.max(np.abs(values))  # Absolute value for maximum bend
    }
    return summary

def plot_boxplots(data, columns_to_plot, x_column='treatment',option='show',out_name='summary.png'):
    """
    Generate boxplots for specified columns against a given x-axis column.
    plot_boxplots(df,['a','b','c'])

    Parameters:
        data (pandas dataframe): data for plotting
        columns_to_plot (list of str): List of column names to plot against `x_column`.
        x_column (str): Name of the column to use as the x-axis (default is 'treatment').

    Returns:
        None
    """
    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with subplots for each column
    num_plots = len(columns_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=False)
    
    # If there's only one column, wrap axes in a list for consistency
    if num_plots == 1:
        axes = [axes]
    
    # Generate a boxplot for each column
    for i, column in enumerate(columns_to_plot):
        sns.boxplot(x=x_column, y=column, data=data, ax=axes[i])
        axes[i].set_title(f'{column} vs {x_column.capitalize()}')
        axes[i].set_xlabel(x_column.capitalize())
        axes[i].set_ylabel(column)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    if option=='show':
        plt.show()
    elif option=='save':
        plt.savefig(out_name, bbox_inches='tight', dpi=300)
        plt.close()