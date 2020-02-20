#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:33:48 2020

@author: Multiple
"""

import SIP3_practical as sip3
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib
from matplotlib.pyplot import axis, imshow, subplot, savefig, figure, close, gca
from skimage import io
from skimage import color
from matplotlib import cm


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







if __name__== "__main__":
    
    imagefolder = "./images/"
    
    exer1(imagefolder)

