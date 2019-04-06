from imageio import imread
import numpy as np


class ColourEncoder:

    @staticmethod
    def encode_colour(colour):
        """
        Converts a base256 encoded colour triplet to a single integer representation.
        """
    
        encoded = colour[0] * 65536 + colour[1] * 256 + colour[2]
    
        return encoded


    @staticmethod
    def decode_colour(colour):
        """
        Converts a single base256 encoded colour triplet to a triplet of [0,255] integers
        """
    
        r = colour // 65536
        g = (colour - r * 65536) // 256
        b = colour % 256
     
        decoded = np.array((r, g, b))
    
        return decoded
        

class ImageEncoder:
    
    @staticmethod
    def encode(X):
        """
        The colours are encoded with a single integer in base256.
        """
        
        if X.ndim != 3:
            raise ValueError("Only 3D images (H,W,(R,G,B,A)) are accepted. Got: {0}".format(X.shape))
    
        encoder = np.array([65536, 256, 1])
        encoded_image = np.dot(X[:,:,:3], encoder)

        return encoded_image
    
    
    @staticmethod
    def decode(X):
        """
        #RRGGBB 256-level image to (R,G,B) image converter
        Parameters:
            X (np.ndarray(height, width) of int): image
        Returns:
            decoded_image (np.ndarray([height, width, 3])) : (R,G,B) image
        """
        
        if X.shape != 2:
            raise ValueError("Image must be of shape 2. Got: {0}".format(X.shape))
        
        decoded_image = np.zeros(np.concatenate([X.shape, np.array[3]]),
                                 dtype = np.int)
         
        decoded_image[:,:,0] = X // 65536
        decoded_image[:,:,2] = X % 256
        decoded_image[:,:,1] = (X - decoded_image[:,:,0] * 65536) // 256
        
        return decoded
    

class ColourHistogram:
    
    @property
    def colours(self):
        return self._colours
    
    @property
    def counts(self):
        return self._counts
    
    @property
    def codes(self):
        return self._codes
    
    
    def __init__(self, colours, counts, codes):
        
        self._colours = colours
        
        self._counts = counts
        
        self._codes = codes
        
            
def calculate_colour_histogram(encoded_image):
    """
    Creates a histogram of the colours.
    """
    
    if encoded_image.ndim != 2:
        raise ValueError("Image must be 2D")
    
    # histogram colours
    codes, counts = np.unique(encoded_image.flat, return_counts = True)
    counts = counts / counts.sum()
    
    # sort to descending order
    idcs = np.argsort(counts)[::-1]
    codes = codes[idcs]
    counts = counts[idcs]
    
    # convert encoded colours to RGB [0--255] triplets
    colours = np.array([ColourEncoder.decode_colour(x) for x in codes])
    
    histogram = ColourHistogram(colours, counts, codes)
    
    return histogram





class ImageCompressor:
    
    @staticmethod  
    def compress(X):
        """
        Compresses an 2D image by marking the regions of identical colour.
        """
           
        compressed = []
        
        if image.shape != 2:
            raise ValueError("Image must be a 2D array")
        n_row, ncol = image.shape
            
        # iterator over doublets of pixels
        gen1, gen2 = tee(X.flat)
        
        i_start = 0
        for idx, (x1, x2) in enumerate(zip(gen1, gen2)):
            if x1 != x2:
                compressed.append([x1, i_start, idx -1])
                i_start = idx
        
        # last colour
        compressed.append([x2, i_start, idx])
                                  
        # calculate row and coloumn indices
        compressed = np.array(compressed)
        
        # bundle colours and 
        compressed = np.hstack(
                [
                    compressed[:,0],
                    compressed[:,1] // n_col, 
                    compressed[:,1] % n_col,
                    compressed[:,2] // n_col, 
                    compressed[:,3] % n_col 
                ])
        
        return compressed
                
    @staticmethod
    def decompress(compressed):
        """
        Parameters:
            compressed (np.ndarray) :
                each row [colour, row_start, col_start, row_end, col_end]
        Returns:
            image (np.ndarray[height, width, 3] of int) : decompressed image.
        """
        
        # create blank image
        image = np.zeros((compressed[-1,2], compressed[-1, 3], 3), dtype = np.int)
        
        for colour, row_start, col_start, row_end, col_end in compressed:
            
            rgb = ColourEncoder.decode(colour)
            
            image[row_start, col_start:, :] = rgb
            
            if row_end > row_start + 1:
                image[row_start + 1 : row_end - 1, :, :] = rgb
            
            image[row_end, :col_end, :] = rgb