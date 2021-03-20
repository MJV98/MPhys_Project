# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:50:17 2021

@author: s1637221
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


"""
File reader
"""
# Use glob to get all of the images
fluorofiles = glob.glob(r'DROPLET IMAGES1\TEST_DROPS\Droplet_171_COLOUR\*.png', recursive = False)

fluorofiles.sort()

im1 = skimage.io.imread(fluorofiles[0], as_gray=True)
im1 = rgb2gray(im1)

#plt.imshow(im1)
#%%
"""
Masking function
"""
nrows, ncols = im1.shape
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (nrows / 2)**2)
im1[outer_disk_mask] = 0


#plt.imshow(im1)
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
Histogram attempt 1
"""

nbins= 400
x_axis = (0.001, 1)
plot, binedges = np.histogram(im1, range = x_axis, bins = nbins)
bins = np.delete(binedges, nbins)

"""
Model Gaussian Fit
"""

initial_guess = [3, 0.4, 10]
popt, pcov = curve_fit(LogGaussfunction, bins, plot, p0=initial_guess)
print(popt)

#%%

#plt.xlim(300, 600)
plt.ylim(1, 4000)
plt.yscale('log')
plt.xlabel('Normalised pixel value')
plt.ylabel('Number of observations')
plt.bar(bins, plot, width =0.005, bottom = None, align = 'center', zorder=1)
x_axis = np.arange(0, 1, 0.0001)
plt.plot(x_axis, LogGaussfunction(x_axis,popt[0],popt[1],popt[2]), "r", zorder=2)