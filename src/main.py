#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:33:48 2020

@author: Multiple
"""

import SIP3_practical as sip3
from scipy import fftpack, signal
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib
from matplotlib.pyplot import axis, imshow, subplot, savefig, figure, close, gca
from skimage import io
from skimage import color
from matplotlib import cm


def pf(statement):
    if False:
        print(statement)


def exer1(path):
    #  calculate the power spectrum of trui.png
    f = np.array(color.rgb2gray(io.imread("trui.png").astype(float)))
    
    img_power = sip3.powerSpectrum(f)
    fig1 = figure()
    fig1.tight_layout()
    
    subplot(1,2,1)
    sip3.plotImage(gca(),f,"Original",gray=True)
    
    subplot(1,2,2)
    sip3.plotImage(gca(),img_power,"Power spectrum",gray=True)
    
    
    sip3.savefig1(path + '2-1.png')
    close()
    
    #  Apply the function scipy.fftpack.fftshift


   
    
class boxFilter():
    '''
    The function to apply filter around a pixel
    
    Parameters
    ----------
    filter : TYPE, square ndarray
        DESCRIPTION. the filter to be applied
    Returns
    -------
    value : TYPE, float
        DESCRIPTION. pixel value as result of filter
    '''
    
    
    def __init__(self, boxfilter):
        self.boxfilter = boxfilter
        r, c  = boxfilter.shape
        self.halfwidth = int(r // 2)
        
        
        if r%2 == 0:
            raise("accept only filter with uneven sides")

        #print("halfwidth{}".format(self.halfwidth))
    
    
    def apply(self, img, x, y):
    
        acum = 0.0

        
        for i in range(- self.halfwidth, self.halfwidth + 1):
            #print("i:{}".format(i))
            for j in range(- self.halfwidth, self.halfwidth + 1):
                pf("i,j : {},{}".format(i,j))
                I = img[x+i][y+j] 
                pf("pixelvalue : {}".format(I))
                
                f = self.boxfilter[self.halfwidth + i, self.halfwidth + j]
                pf("filtervalue : {}".format(f))
                
                acum = acum + I * f
                pf("acumulated value : {}".format(acum))
                
        
        #print("acum: {}, I: {}, f: {}".format(acum,I,f))
        return acum
    



def convolve(img, fFilter):
    '''
    The convolution function convolves the image with the filter flipped around the horizontal and the veritcal axes
    
    Parameters
    ----------
    img: TYPE 2dArray
        DESCRIPTION the img to be convolved.
    filter : TYPE, square ndarray
        DESCRIPTION. the filter to be applied
    Returns
    -------
    img: TYPE 2dArray
        DESCRIPTION the result of the convolution
    '''
    
    row, column =  img.shape
    
    r, _ = fFilter.shape

    filteredImage= np.zeros_like(img)
    
    border = int(r/2)
    
    
    funcFilter = boxFilter(np.flip(fFilter))
    
    for i in range(border, row-border):
        for j in range(border,column - border):
            filteredImage[i][j] = funcFilter.apply(img,i,j)
    
    return filteredImage


def fastFourierConvolve(img, fFilter):
    
    # Take Fourier transform
    imgtransform = fftpack.fft2(img)
    # Make fftshift
    imgtransform = fftpack.fftshift(imgtransform)
    
    # Apply filter
    result = np.multiply(fFilter, imgtransform)
    
    # reverse fftshift
    resultOfConvolution = fftpack.ifftshift(result)
    
    # Inverse Fourier transform
    res = fftpack.ifft2(resultOfConvolution)

    return res



def exer2(path):
    f = np.ones((3,3)) / 9
    #print(f)
    
    img = np.array(color.rgb2gray(io.imread("trui.png").astype(float)))
        
    convolvede = convolve(img,f)
    
    
    row, column = img.shape
    gaussianrow = signal.gaussian(row, 1)
    gaussiancolumn = signal.gaussian(column, 1)
    
    gaussianFilter = np.outer(gaussianrow,gaussiancolumn)
    
    fcon = fastFourierConvolve(img,gaussianFilter)
    
    row, column = img.shape
    
    meanfilter = np.ones_like(img)/ (row * column)
    
    fconMean = fastFourierConvolve(img,meanfilter)
    
    
    def lp_filter(rad, img):
        fil = np.zeros(img.shape)
        n, m = img.shape
        for i in range(n):
            for j in range(m):
                r = np.linalg.norm([i-n/2, j-m/2])
                if r < rad:
                    fil[i,j] = 1
            
        return fil
    
    circleFilter = lp_filter(50, img)
    
    circle=fastFourierConvolve(img,circleFilter)
    
    plt.figure
    subplot(4,1,1)
    sip3.plotImage(gca(), img, "original", gray=True)
    subplot(4,1,2)
    sip3.plotImage(gca(), convolvede, "convolved with mean filter 3x3", gray=True)
    
    subplot(4,1,3)
    sip3.plotImage(gca(), np.real(fcon), "fast fourie convolution with a gaussian filter", gray=True)
    
    subplot(4,1,4)
    sip3.plotImage(gca(), np.real(circle), "fast fourie convolution with a circle", gray=True)
    
    
    sip3.savefig1(path + '2-2-1.png')
    close()
    
    
      
    plt.figure
    subplot(3,2,1)
    sip3.plotImage(gca(), np.real(fcon), "fast fourie convolution with a gaussian filter", gray=True)
    subplot(3,2,2)
    sip3.plotImage(gca(), gaussianFilter, "gaussian filter", gray=True)
    
    subplot(3,2,3)
    sip3.plotImage(gca(), np.real(circle), "fast fourie convolution with a circle", gray=True)
    
    subplot(3,2,4)
    sip3.plotImage(gca(), circleFilter, "circle", gray=True)
    
    subplot(3,2,5)
    sip3.plotImage(gca(), np.real(fconMean), "fast fourie convolution with a mean filter", gray=True)
    
    subplot(3,2,6)
    sip3.plotImage(gca(), meanfilter, "meanfitler", gray=True)
    
    
    sip3.savefig1(path + '2-2-2.png')
    close()

    
    

#%%
if __name__== "__main__":
    
    imagefolder = "./images/"
    
    #exer1(imagefolder)

    exer2(imagefolder)