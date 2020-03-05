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
import os
os.chdir(r"C:\Users\Dalle\Dropbox\skole\SIP\Assignments\Individual_Handin")

from utilitaries import powerSpectrum, plotImage, savefig1, convolve, fastFourierConvolve, fastForierApplyFilter

imageLoadingFolder = r'''C:\Users\Dalle\Dropbox\skole\SIP\Assignments\Week 1'''
imageSavingFolder = r"C:\Users\Dalle\Dropbox\skole\SIP\Assignments\Individual_Handin\images"

