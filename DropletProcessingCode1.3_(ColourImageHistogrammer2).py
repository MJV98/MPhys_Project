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
fluorofiles = glob.glob('DROPLET IMAGES/tests/*.png', recursive = False)
fluorofiles2 = glob.glob('DROPLET IMAGES/Droplet_171/171t*.tif', recursive = False)

fluorofiles.sort()
fluorofiles2.sort()

im1 = skimage.io.imread(fluorofiles[0], as_gray = True)
im2 = skimage.io.imread(fluorofiles2[0])

#print(im1)

"""
Histogram attempt 2
"""
# Generate the image histogram
counts, bins = skimage.exposure.histogram(im1)

# Plot it as a bar plot
plt.bar(bins, counts)

# Add appropriate labels.
plt.xlabel('pixel value')
plt.ylim(1, 10000)
plt.yscale('log')
plt.ylabel('number of observations')







