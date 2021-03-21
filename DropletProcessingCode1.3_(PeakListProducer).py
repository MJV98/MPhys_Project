"""
Created on Thu Nov 26 12:41:30 2020

@author: milov
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import transform, exposure, util, color, data
from skimage.util import compare_images
from skimage.util import img_as_ubyte
from skimage.util import img_as_uint 
from scipy.optimize import curve_fit
import glob
import skimage.io
import skimage.filters
from skimage.color import rgb2gray
import skimage.viewer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

#%%

"""
Logarithmic Gauss Function
"""
def LogGaussfunction(x,a,b,c):
    loggauss = 10**(a*(np.exp(-((x-b)**2)/2*(c**2))))
    return loggauss

#a = 4.3
#b = 127
#c =0.03


#%%
"""
File reader
"""

fluorofiles = glob.glob(r'DROPLET IMAGES2\Droplet_344_COLOUR\*.png', recursive = False)

fluorofiles.sort()

"""
Loop to run the Log Gaussian fit to a series of histograms and append each peak 
position to a list peaks[].
"""

peaks = []

for i in range(len(fluorofiles)):
    im1 = skimage.io.imread(fluorofiles[i], as_gray = True)
    image = rgb2gray(im1) 
    
    nrows, ncols = image.shape
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = nrows / 2, ncols / 2
    outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (nrows / 2)**2)
    image[outer_disk_mask] = 0
    
    nbins= 400
    x_axis = (0.001, 1)
    ax, binedges = np.histogram(image, range = x_axis, bins = nbins)
    bins = np.delete(binedges, nbins)
    image = skimage.io.imread(fluorofiles[i])
    initial_guess = [3, 0.4, 10]
    popt, pcov = curve_fit(LogGaussfunction, bins, ax, p0=initial_guess)
    peaks.append(popt[1])

print(peaks)

"""
Saves the list as a text file 
"""

with open('section7heatmap20.txt', 'w') as filehandle:
    for listitem in peaks:
        filehandle.write('%s\n' % listitem)


