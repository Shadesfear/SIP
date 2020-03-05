# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:09:02 2020

@author: Dalle
"""

# Importing packages
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib
from matplotlib.pyplot import axis, imshow, subplot, savefig, figure, close, gca
from skimage import io
from skimage import color
from matplotlib import cm
from skimage.morphology import opening
from skimage.morphology import closing
from scipy.ndimage import black_tophat
from skimage.morphology import binary_erosion
from skimage.morphology import binary_dilation
from skimage.color import gray2rgb
from skimage.draw import line
import os
os.chdir(r"C:\Users\Dalle\Dropbox\skole\SIP\Assignments\GroupWork\src")

from util import plotImage, savefig1, time_function

imageLoadingFolder = r'''C:\Users\Dalle\Dropbox\skole\SIP\Assignments\Week 4'''
imageSavingFolder = r"C:\Users\Dalle\Dropbox\skole\SIP\Assignments\Groupwork\src\images\A4_images"

#%%


# Exercise 1.1

A = io.imread(imageLoadingFolder + "\cells_binary.png")

fig_d = [1,3]

plt.figure()
plt.subplot(fig_d[0], fig_d[1], 1)
plotImage(gca(), A[250:350, 350:450], 'Original zoomed', gray=True)

plt.subplot(fig_d[0], fig_d[1], 2)
plotImage(gca(), opening(opening(A))[250:350, 350:450], 'Opening', gray=True)

plt.subplot(fig_d[0], fig_d[1], 3)
plotImage(gca(), closing(closing(A)[250:350, 350:450]), 'Closing', gray=True)


savefig(imageSavingFolder + r"\1-1.png")
close()





#%%

# Exercise 1.3
from skimage.morphology import medial_axis
from scipy.ndimage.morphology import binary_hit_or_miss

def Normalize(I):
    '''
    Scales image intensities to range between 0 and 255

    Parameters
    ----------
    Image : n,m dimensional numpy array
        DESCRIPTION.

    Returns
    -------
    Normalized numpy array

    '''
    return(I/I.max())
    
def X_marksTspot(I, S_hit, S_miss):
    #make a color image and insert the binary image in one channel.
    I2 = Normalize(I)
    Iout = np.dstack([I2, I2, I2])
    
    #get hits from hit or miss
    hits = binary_hit_or_miss(I, structure1 = S_hit, structure2 = S_miss)#, origin1 = (n//2, m//2), origin2 = (n//2, m//2))
    loc = np.array(np.where(hits == True))

    #draw x in red at locations for hits
    for i in range(loc.shape[1]):
        x, y = loc[:,i]
        rr, cc = line(x-10, y-8, x+10, y+8)
        Iout[rr, cc, :] = [1,0,0]
        rr, cc = line(x+10, y-8, x-10, y+8)
        Iout[rr, cc, :] = [1,0,0]
    
    return(Iout)




A = io.imread(imageLoadingFolder + "\digits_binary_inv.png")
A2 = np.where(A < A.max()/8, 1, 0)

# #direct cut out letter x
# F1 = A[0:40,60:90] 
# F1_miss = binary_dilation(F1)

F2 = binary_erosion(A2[0:40,60:90])
F2_miss = np.where(binary_dilation(binary_dilation(binary_dilation(F2))) == True, 0,1)
#closing on cutout
# F2 = medial_axis(closing(F1/np.max(F1)))


# H, _ = np.histogram(A/A.max()*256, bins = 256, range = (-0.5,255.5))

fdim = [25,8]

fig = plt.figure(figsize = (fdim[0], fdim[1]), constrained_layout = True) #
gs = fig.add_gridspec(fdim[0],fdim[1])

plt.subplot(gs[0:8,0:4]) #fig_d[0], fig_d[1], 1)
plotImage(gca(), A2, 'Binary Image', gray=True)


plt.subplot(gs[0:8,5:6]) #fig_d[0], fig_d[1], 3)
plotImage(gca(), F2, 'Binary cutout (SE)', gray=True)

plt.subplot(gs[0:8,7:8]) #fig_d[0], fig_d[1], 4)
plotImage(gca(), F2_miss, 'Dilation on SE (SE_miss)', gray=True)


plt.subplot(gs[9:,:]) #fig_d[0], fig_d[1], 5)
ax = gca()
ax.imshow(X_marksTspot(A2, F2, F2_miss), interpolation='nearest', aspect = 'equal')
ax.set_title('Hit or Miss')
ax.axis('off')

plt.tight_layout()
savefig(imageSavingFolder + r"\1-3.png")
close()






#%%



















