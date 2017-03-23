import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
import datetime

def save_bar_chart_figure(data, x_labels):
    """ Saves bar chart as png file

    INPUT:
    data: list of input data for bar chart
    RETURNS:
    outpath: str for png filepath
    """
    # Plot bar chart and save figure to file
    # --------------------------------------
    # Saving plot to file

    # Convert to pandas Series for easy plotting
    pred_series = pd.Series.from_array(data)

    cgfont = {'fontname':'Century Gothic'}

    # Set plotting parameters
    plt.figure(figsize=(6, 4))
    ax = pred_series.plot(kind='bar', rot=0, grid=None, fontsize=10, color="#0099DF", width=0.8, alpha=0.5)
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
        ax.text(rect.get_x() + rect.get_width()/2, height + 1, label, ha='center', va='bottom', fontsize=14, **cgfont)

    plt.yticks([])

    datestring = '{:04d}{:02d}{:02d}{:02d}{:02d}'.format(datetime.date.today().year,
                                                         datetime.date.today().month,
                                                         datetime.date.today().day,
                                                         datetime.datetime.today().hour,
                                                         datetime.datetime.today().minute)

    print "Saving figure..."
    out_filepath = 'static/monthly_income.png'.format(datestring)
    savefig(out_filepath, dpi=200, facecolor='w', edgecolor=None,
        orientation='portrait', papertype=None, format=None,
        transparent=False,bbox_inches='tight', pad_inches=0.0,
        frameon=None)

    return out_filepath
