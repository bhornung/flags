from imageio import imread
import numpy as np

from histo_utils import calculate_colour_histogram
from histo_utils import ImageEncoder
    
    
def plot_colour_histogram(colours, counts, ax, n_show = None):
    """
    Plots bar histogram of colour histogram.
    Parameters:
        colors (np.ndarray[n_colours, 3]) : list of colours
        counts (np.ndarray[n_colours] : counts of colours
        ax (plt.axis) : axis to draw on
        n_show ({int, None}) : how many colours to draw. If None, all colours are plotted. Default: None.
    """
    
    
    if n_show is None:
        i_max = colours.size
    else:
        i_max = min(colours.size, n_show)
    
    xvals = np.arange(i_max)
    yvals = counts[:i_max]
    
     
    colours_ = colours / 256.0
     
    ax.set_yscale('log')
    ax.set_facecolor('#f0f0f0')
    ax.set_xticks([])
    ax.set_xlabel('Colours')
    ax.set_ylabel('P(colour)')
    ax.bar(xvals, yvals, color = colours_)
    
    
def plot_flag_with_histo(path_to_file, ax1, ax2):
    """
    Plots histogram along with the associated flag.
    """
    
    image = imread(path_to_file)
    ax1.axis('off')
    ax1.imshow(image)
    
    encoded = ImageEncoder.encode(image)
    histogram = calculate_colour_histogram(encoded)
    
    plot_colour_histogram(histogram.colours, histogram.counts, ax2, n_show = 50)