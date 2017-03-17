import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt

def save_bar_chart_figure(data, filepath):
    # Plot bar chart and save figure to file
    # --------------------------------------
    # Saving plot to file

    # Convert to pandas Series for easy plotting
    pred_series = pd.Series.from_array(data)

    cgfont = {'fontname':'Century Gothic'}
    x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Set plotting parameters
    plt.figure(figsize=(7, 4))
    ax = pred_series.plot(kind='bar', rot=0, grid=None, fontsize=13, color="#0099DF", width=0.7)
    ax.set_xticklabels(x_labels,**cgfont)
    ax.set_facecolor('white')

    # Remove all axes for clean
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    rects = ax.patches
    labels = data

    for rect, label in zip(rects, data):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 1, label, ha='center', va='bottom', fontsize=16, **cgfont)

    plt.yticks([])

    savefig(filepath, dpi=None, facecolor='w', edgecolor=None,
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.0,
        frameon=None)
