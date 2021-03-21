
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
fluorofiles = glob.glob('DROPLET IMAGES2/Droplet_344/*.tif', recursive = False)

fluorofiles.sort()

for i in range(len(fluorofiles)): 
    image = skimage.io.imread(fluorofiles[i])
    fig = plt.imshow(image)
    plt.axis('off')
    plt.savefig(r'C:\Users\s1637221\.spyder-py3\DROPLET IMAGES2\Droplet_344_COLOUR\344t' +str(i) +'colour.png', bbox_inches='tight', pad_inches = 0)
    
