#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:21:11 2020

This file holds the solution to section 1 for assingment 6 SIP
"""

from  skimage.feature import canny, corner_harris, corner_peaks

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.pyplot import axis, imshow, subplot, savefig, figure, close, gca
import numpy as np


def exer11(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Include an illustration showing the results of the different set-
        tings and explain what the effect is of 
        each of the parameters based on these results.
    """  

    img = io.imread(testImageFolder + "hand.tiff")
    
    im = normalize(img, norm='max')
    

        
    
    i = 0
    j = 0
    k = 0 
    

    
    sigma = [1,2,4]
    lthresholds = [10,15]
    low_threshold = list(map(lambda x: x/100, lthresholds))
    hthresholds = [20,25]
    high_threshold = list(map(lambda x: x/100, hthresholds))

    for i in range(len(sigma)): 
        for j in range(len(low_threshold)):
            for k in range(len(high_threshold)):
                res = canny(im, sigma = sigma[i], low_threshold = low_threshold[j], high_threshold= high_threshold[k])
                
                fig = plt.figure()
                ax = plt.subplot(1,1,1)
                ax.imshow(res, cmap=plt.cm.gray)
                ax.axis('off')
                #ax.set_title(r'$\sigma={},low_t={},high_t={}$'.format(sigma[i], low_threshold[j], high_threshold[k]), fontsize=11)
                
                fig.tight_layout()
                
                filename = "exer11-sigma={}_low_t={}_high_t={}".format(sigma[i], lthresholds[j], hthresholds[k])

                savefig(saveImageFolder + filename)
                plt.close()

        
     
    



def exer12(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Include an illustration showing the results of the different set- 
        tings and explain what the effect is of 
        each of the parameters based on these results.
    """
   

    img =  io.imread(testImageFolder + "modelhouses.png")
    
    im = normalize(img, norm='max')
    

    i = 0
    j = 0 
    
    sigma = [1,2,4]
    Kthresholds = [0,5,20]
    K = list(map(lambda x: x/1e2, Kthresholds))



    for i in range(len(sigma)): 
        for j in range(len(K)):
                res = corner_harris(im, sigma = sigma[i], k = K[j], method='k')
                
                fig = plt.figure()
                ax = plt.subplot(1,1,1)
                ax.imshow(res, cmap=plt.cm.gray)
                ax.axis('off')
                # title = r'$\sigma={},k={}$'.format(sigma[i], Kthresholds[j])
                # ax.set_title(title, fontsize=11)
                
                fig.tight_layout()
                
                filename = "exer12-" + 'sigma={},k={}'.format(sigma[i], Kthresholds[j])
        
                plt.savefig(saveImageFolder + filename)    
                plt.close()
    
    eps = [1e-7,1,10]
    
    for j in range(len(eps)):
            res = corner_harris(im, sigma = 2, eps = eps[j], method='eps')
            
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.imshow(res, cmap=plt.cm.gray)
            ax.axis('off')
            # title = r'$\sigma={},eps={}$'.format(sigma[i], eps[j])
            # ax.set_title(title, fontsize=11)
            
            fig.tight_layout()
            
            filename = "exer12-" + 'sigma={}_eps={}.png'.format(2, eps[j])
    
            plt.savefig(saveImageFolder + filename)    
            plt.close()
        
    


def exer13(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Include in the report the code for your function. 
        Apply this function to the modelhouses.png image and 
        create a figure of the resulting corner points overlaid 
        on the modelhouses.png image. 
        Remember to indicate your choice of parameter settings 
        in the caption of the figure.
    """
    
    
    img = plt.imread(testImageFolder + "modelhouses.png")
    
    im = normalize(img, norm='max')
    

    sigma = 2
    Kthresholds = [0,1,15,20]
    K = list(map(lambda x: x/1e2, Kthresholds))

    for j in range(len(K)):
            res = corner_harris(im, sigma = sigma, k = K[j], method='k')
            peaks = corner_peaks(res*-1)
            
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            ax.plot(peaks[:,1],peaks[:,0], '.b')
            ax.imshow(img, cmap=plt.cm.gray)
    
            ax.axis('off')
            # title = r'Harris_corner $\sigma={},k={}$'.format(2,K[j])
            # ax.set_title(title, fontsize=11)
            
            fig.tight_layout()
            
            filename = "exer13-" + 'k={}_sigma={}_inverted'.format(Kthresholds[j],sigma)
    
            plt.savefig(saveImageFolder + filename)    
            plt.close()
    



def main():
    
    testImageFolder = "./Week 6/"

    saveImageFolder = "./exer1Images/"
    
    # exer11(testImageFolder,saveImageFolder)
    # exer12(testImageFolder,saveImageFolder)
    exer13(testImageFolder,saveImageFolder)

if __name__ == "__main__":
    
    main()