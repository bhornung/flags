from imageio import imread
import numpy as np

def calculate_colour_histogram(image):
    """
    Creates a histogram of colours of an image. The colours are encoded with a single integer
    in base256.
    """
    
    encoder = np.array([65536, 256, 1])
    vals, cnts = np.unique(np.dot(image[:,:,:3], encoder).flat, return_counts = True)
    cnts = cnts / cnts.sum()
    
    idcs = np.argsort(cnts)[::-1]
    histo = vals[idcs], cnts[idcs]

    return histo


def encode_colour(colour):
    """
    Converts a base256 encoded colour triplet to a single integer representation.
    """
    
    encoded = colour[0] * 65536 + colour[1] * 256 + colour[2]
    
    return encoded


def decode_colour(colour):
    """
    Converts a single base256 encoded colour triplet to a triplet of [0,255] integers
    """
    
    r = colour // 65536
    g = (colour - r * 65536) // 256
    b = colour % 256
     
    decoded = np.array((r, g, b))
    
    return decoded


def convert_code_to_padded_hex(x):
    
    
    string = hex(x)[2:]
    if x < 16:
        string = "0" + string
    
    return string

    
def code_to_hex(code):
    """
    Converts a single integer encoded coour triplet to a hexadecimal string.
    """
    
    decoded = decode_colour(code)
    hex_string = '#' + "".join([convert_code_to_padded_hex(x) for x in decoded])

    return hex_string

    
def plot_colour_histogram(histo, ax, n_show = None):
    
    vals, cnts = histo
    
    if n_show is None:
        i_max = vals.size
    else:
        i_max = min(vals.size, n_show)
    
    xvals = np.arange(i_max)
    yvals = cnts[:i_max]
    
    colours = np.array([code_to_hex(x) for x in vals[:i_max]])
    ax.set_yscale('log')
    ax.set_facecolor('#f0f0f0')
    ax.set_xticks([])
    ax.set_xlabel('Colours')
    ax.set_ylabel('P(colour)')
    ax.bar(xvals, yvals, color = colours)
    
    
def plot_flag_with_histo(path_to_file, ax1, ax2):
    
    im = imread(path_to_file)
    ax1.axis('off')
    ax1.imshow(im)
    plot_colour_histogram(calculate_colour_histogram(im) , ax2, n_show = 50)