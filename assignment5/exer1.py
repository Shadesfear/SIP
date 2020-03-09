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
imageSavingFolder = r"C:\Users\Dalle\Dropbox\skole\SIP\Assignments\Groupwork\assignment5\images"

#%
def exer1():
    cells_binary = color.rgb2gray(io.imread('../Week_4_export/cells_binary.png'))

    # cells_binary = np.max(cells_binary) - cells_binary
    ball = disk(2)

    opened = opening(cells_binary, ball)
    closed = closing(cells_binary, ball)

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(cells_binary, cmap='gray')
    # ax[1].imshow(closed, cmap='gray')
    # ax[2].imshow(opened, cmap='gray')
    # plt.show()
    io.imsave('ball.png', ball)
    io.imsave('opened.png', opened)
    io.imsave('closed.png', closed)



    blobs_inv = io.imread('../Week_4_export/blobs_inv.png')
    blobs_inv = np.max(blobs_inv) - blobs_inv
    se0 = np.array([[1, 1, 1, 1, 1]])
    se1 = disk(2)
    print()
    print(se1)
    se2 = np.array([[0,0,0],
                   [1,1,0],
                   [1,1,0]])

    hitormiss0 = binary_hit_or_miss(blobs_inv, se0, origin1=[0, 2])
    hitormiss1 = binary_hit_or_miss(blobs_inv, se1, 1-disk(12))
    hitormiss2 = binary_hit_or_miss(blobs_inv, se2, 1-se2)


    top_hat0 = white_tophat(blobs_inv, se0)
    top_hat1 = white_tophat(blobs_inv, se1)
    top_hat2 = white_tophat(blobs_inv, se2)

    btop_hat0 = black_tophat(blobs_inv, se0)
    btop_hat1 = black_tophat(blobs_inv, se1)
    btop_hat2 = black_tophat(blobs_inv, se2)


    plt.imshow(hitormiss2, cmap = 'gray')
    plt.savefig('hitormiss2pdf')


# Exercise 1.1.1
def Exer111():
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

#% hit operation using black tophat

def Exer114():
    '''
    NOT USED

    Returns
    -------
    None.

    '''
    A = io.imread(imageLoadingFolder + "\cells_binary.png")
    
    def LargeOpening(I, n):
        
        for i in range(n):
            I = binary_erosion(I)
        for i in range(n):
            I = binary_dilation(I)
            
        return(I)
    
    
    fig_d = [1,3]
    
    SE = A[367:395, 217:245]
    
    plt.figure()
    plt.subplot(fig_d[0], fig_d[1], 1)
    plotImage(gca(), opening(opening(SE)), 'SE_hit', gray=True)
    
    plt.subplot(fig_d[0], fig_d[1], 2)
    plotImage(gca(), black_tophat(opening(opening(A))[250:350, 350:450], ), 'Opening', gray=True)
    
    plt.subplot(fig_d[0], fig_d[1], 3)
    plotImage(gca(), closing(closing(A)[250:350, 350:450]), 'Closing', gray=True)
    
    
    # savefig(imageSavingFolder + r"\1-1.png")
    # close()







#%

# Exercise 1.3
    
def Exer13():
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
    
    #direct cut out of letter x from A2
    F1 = A2[0:40,60:90] 
    F1_miss = np.where(F1 == True, 0,1)
    
    F2 = binary_erosion(A2[0:40,60:90])
    F2_miss = np.where(binary_dilation(binary_dilation(binary_dilation(F2))) == True, 0,1)
    #closing on cutout
    # F2 = medial_axis(closing(F1/np.max(F1)))
    
    
    # H, _ = np.histogram(A/A.max()*256, bins = 256, range = (-0.5,255.5))
    
    fdim = [25,8]
    
    fig = plt.figure(figsize = (fdim[0], fdim[1]), constrained_layout = True) #
    gs = fig.add_gridspec(fdim[0],fdim[1])
    
    # plt.subplot(gs[0:8,0:4]) #fig_d[0], fig_d[1], 1)
    # plotImage(gca(), A2, 'Binary Image', gray=True)
    
    
    plt.subplot(gs[0:8,0:4]) #fig_d[0], fig_d[1], 3)
    plotImage(gca(), F2, 'Binary cutout (SE)', gray=True)
    
    plt.subplot(gs[0:8,4:8]) #fig_d[0], fig_d[1], 4)
    plotImage(gca(), F2_miss, 'Dilation on SE (SE_miss)', gray=True)
    
    
    plt.subplot(gs[9:,:]) #fig_d[0], fig_d[1], 5)
    ax = gca()
    ax.imshow(X_marksTspot(A2, F2, F2_miss), interpolation='nearest', aspect = 'equal')
    ax.set_title('Hit or Miss')
    ax.axis('off')
    
    plt.tight_layout()
    savefig(imageSavingFolder + r"\1-3.png")
    close()
    
    
    
    
    # maybe plot a figure of the naive implementation.. but what is the naive implementation? binary img and filter without erosion and dilation?
    plt.figure()
    ax = gca()
    ax.imshow(X_marksTspot(A2, F1, F1_miss), interpolation='nearest', aspect = 'equal')
    ax.set_title('Naive Hit or Miss')
    ax.axis('off')
    savefig(imageSavingFolder + r"\1-3_naiveMatch.png")
    close()




def exer1_4():
    pf = color.rgb2gray(io.imread('../Week_4_export/bokeh_purpleflowers.jpg'))
    filterd = filters.gaussian(pf, sigma = 50)
    plt.imshow(filterd, cmap='gray')
    plt.savefig('filterd.pdf')
    minus = pf - filterd
    plt.imshow(minus, cmap='gray')
    plt.savefig('minus.pdf')

    gradient = minus - erosion(minus)
    plt.imshow(gradient, cmap='gray')
    plt.savefig('gradient.pdf')

    threshold = gradient > 0.1
    plt.imshow(threshold, cmap='gray')
    plt.savefig('threshold.pdf')

    dilated = threshold.copy()
    for i in range(15):
        dilated = dilation(dilated)
    plt.imshow(dilated, cmap='gray')
    plt.savefig('dilated.pdf')

    closed = dilated.copy()
    for i in range(100):

        closed = closing(closed)
    plt.imshow(closed, cmap='gray')
    plt.savefig('closed.pdf')

    eroded = closed
    for i in range(10):
        eroded = erosion(eroded)

    plt.imshow(eroded, cmap='gray')
    plt.savefig('eroded.pdf')

    plt.imshow(pf * eroded)
    plt.savefig('pf_eroded_final.pdf')
    # plt.show()
