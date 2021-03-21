# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:05:01 2021

@author: s1637221
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
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
fluorofiles = glob.glob('DROPLET IMAGES1/TEST_DROPS/Droplet_253/253t54.tif', recursive = False)
fluorofiles2 = glob.glob('DROPLET IMAGES1/TEST_DROPS/Droplet_217/17t54.tif', recursive = False)

fluorofiles.sort()
fluorofiles2.sort()

im1 = skimage.io.imread(fluorofiles[0])
im2 = skimage.io.imread(fluorofiles2[0])

#plt.imshow(im1)
#%%

"""
Image crop
"""

nrows, ncols = im1.shape
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (nrows / 2)**2)
im1[outer_disk_mask] = 0

nrows, ncols = im2.shape
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (nrows / 2)**2)
im2[outer_disk_mask] = 0
#plt.imshow(im1)
#%%
"""
Logarithmic Gauss Function
"""
def LogGaussfunction(x,a,b,c):
    loggauss = 10**(a*(np.exp(-((x-b)**2)/2*(c**2))))
    return loggauss

#a = 2.9
#b = 435
#c =0.04

#%%

"""
Histogram plotter
"""
nbins= 400
x_axis = (350, 600)
ax, binedges = np.histogram(im1, range = x_axis, bins = nbins)
ax2, binedges = np.histogram(im2, range = x_axis, bins = nbins)
bins = np.delete(binedges, nbins)
diff = ax2 - ax


initial_guess = [3, 432, 0.033]
popt, pcov = curve_fit(LogGaussfunction, bins, ax, p0=initial_guess)
print(bins[160])
print(bins[350])
integral = sum(diff[160:350])
print(integral)
#%%

plt.xlim(350, 700)
plt.ylim(1, 2000)
plt.yscale('log')
plt.xlabel('Pixel intensity value')
plt.ylabel('Number of observations')
plt.bar(bins, ax, width =1.5, bottom = None, align = 'center', zorder=1)
plt.bar(bins, ax2, width =1.5, bottom = None, align = 'center', zorder=2)
#x_axis = np.arange(300, 600, 0.001)
#plt.plot(x_axis, LogGaussfunction(x_axis,2.9,433,0.04), "r", zorder=2)
#plt.plot(x_axis, LogGaussfunction(x_axis,popt[0],popt[1],popt[2]), "r", zorder=2)


"""
Histogram attempt 2

counts, bins = skimage.exposure.histogram(im1, normalize = False)
plt.yscale('log')
plt.xlim([300, 600])
plt.ylim([1, 1200])
plt.bar(bins, counts)
plt.xlabel('pixel value')
plt.ylabel('number of observations')
"""

"""
Histogram attempt 3

histogram, bin_edges = np.histogram(im1, bins=200, range=(300, 600))
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.yscale('log')
plt.xlim([300, 600])
plt.plot(bin_edges[0:-1], histogram)
plt.show()
"""


#%%
