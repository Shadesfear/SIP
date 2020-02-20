# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:56:27 2020

@author: Multiple
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



def powerSpectrum(f):

    # Take Fourier transform
    f_fft = fftpack.fft2(f)
    img_fft_shifted = fftpack.fftshift(f_fft)
    power = np.abs(img_fft_shifted) ** 2
    img_logtrans = 100*np.log(1+power)
    return img_logtrans
    


def plotImage(ax, img, title, titleSize = 10, axesOn = False, gray = False):

    if gray == False:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')
    
    gca().set_title(title, fontsize=titleSize)
   
    if axesOn: 
        axis('on')
    else: 
        axis('off')
    return ax


def savefig1(filename):
    savefig(filename, dpi=500 , bbox_inches = 'tight')

def exer1():
    #  calculate the power spectrum of trui.png
    f = np.array(color.rgb2gray(io.imread("trui.png").astype(float)))
    
    img_power = powerSpectrum(f)
    fig1 = figure()
    fig1.tight_layout()
    
    subplot(1,2,1)
    plotImage(gca(),f,"Original",gray=True)
    
    subplot(1,2,2)
    plotImage(gca(),img_power,"Power spectrum",gray=True)
    
    
    savefig1('2-1.png')
    close()
    
    #  Apply the function scipy.fftpack.fftshift


if __name__== "__main__":
    exer1()

