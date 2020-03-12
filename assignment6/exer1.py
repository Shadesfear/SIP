#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:21:11 2020

This file holds the solution to section 1 for assingment 6 SIP
"""

from  skimage.feature import canny
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt




def exer11(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Include an illustration showing the results of the different set-
        tings and explain what the effect is of 
        each of the parameters based on these results.
    """  
    
    
    # small test to see if the folders where correct
    img = plt.imread(testImageFolder + "hand.tiff")
    
    im = normalize(img, norm='max')
    
    # plots BGR colours should be fixed. 
    #plt.imsave(saveImageFolder + "hand.tiff", img, cmap=plt.cm.gray)
        
    
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
                ax = plt.subplot(1,2,1)
                ax.imshow(img, cmap=plt.cm.gray)
                ax = plt.subplot(1,2,2)
                ax.set_title(r'hands', fontsize=11)
                ax.imshow(res, cmap=plt.cm.gray)
                ax.axis('off')
                ax.set_title(r'$\sigma={},low_t= {} high_t={}$'.format(sigma[i], low_threshold[j], high_threshold[k]), fontsize=11)
                
                fig.tight_layout()
                
                filename = "exer11-$\sigma={},low_t= {} high_t={}$".format(sigma[i], lthresholds[j], hthresholds[k])
        
                plt.savefig(saveImageFolder + filename)    
        
        
    pass



def exer12(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Include an illustration showing the results of the different set- 
        tings and explain what the effect is of 
        each of the parameters based on these results.
    """

    pass


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
    
    pass


def main():
    
    testImageFolder = "./Week 6/"

    saveImageFolder = "./imageResults/"
    
    exer11(testImageFolder,saveImageFolder)
    exer12(testImageFolder,saveImageFolder)
    exer13(testImageFolder,saveImageFolder)

if __name__ == "__main__":
    
    main()