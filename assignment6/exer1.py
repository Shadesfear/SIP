#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:21:11 2020

This file holds the solution to section 1 for assingment 6 SIP
"""

from  skimage.feature import canny, corner_harris, corner_peaks

from PIL import Image
import numpy as np
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
    img = plt.imread(testImageFolder + r"hand.tiff")
    # img = Image.open(testImageFolder + r"hand.tiff")
    
    #  # Map PIL mode to numpy dtype (note this may need to be extended)
    # # dtype =np.uint8 # np.float32 #np.uint8
    
    # img = np.array(list(img.getdata()), dtype='uint8')
    
    im = normalize(img, norm='max') #wouldnt this function normalize per column?
    
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
                ax = plt.subplot(1,1,1)
                ax.imshow(res, cmap=plt.cm.gray)
                ax.axis('off')
                ax.set_title(r'$\sigma={},low_t= {} high_t={}$'.format(sigma[i], low_threshold[j], high_threshold[k]), fontsize=11)
                
                fig.tight_layout()
                
                filename = "exer11-$\sigma={},low_t= {} high_t={}$".format(sigma[i], lthresholds[j], hthresholds[k])
        
                plt.savefig(saveImageFolder + filename)    
        
     
    



def exer12(testImageFolder,saveImageFolder):
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
    
    sigma = [1,2,4,8]
    Kthresholds = [5,10,15,20]
    K = list(map(lambda x: x/1e2, Kthresholds))
    
    
    # the values of eps choosen produces garbage. need to ask for better values
    epsthreshold = [10,50,100]
    eps = list(map(lambda x: x/1e7, epsthreshold))

    for i in range(len(sigma)): 
        for j in range(len(K)):
                res = corner_harris(im, sigma = sigma[i], k = K[j], method='k')
                
                fig = plt.figure()
                ax = plt.subplot(1,1,1)
                ax.imshow(res, cmap=plt.cm.gray)
                ax.axis('off')
                title = r'$\sigma={},k={}$'.format(sigma[i], Kthresholds[j])
                ax.set_title(title, fontsize=11)
                
                fig.tight_layout()
                
                filename = "exer12-" + title
        
                plt.savefig(saveImageFolder + filename)    
        
    
    for i in range(len(sigma)): 
        for j in range(len(eps)):
                res = corner_harris(im, sigma = sigma[i], eps = eps[j], method='eps')
                
                fig = plt.figure()
                ax = plt.subplot(1,1,1)
                ax.imshow(res, cmap=plt.cm.gray)
                ax.axis('off')
                title = r'$\sigma={},eps={}$'.format(sigma[i], eps[j])
                ax.set_title(title, fontsize=11)
                
                fig.tight_layout()
                
                filename = "exer12-" + title
        
                plt.savefig(saveImageFolder + filename)    
        
        
    


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
    
    
    
    # small test to see if the folders where correct
    img = plt.imread(testImageFolder + "modelhouses.png")
    
    im = normalize(img, norm='max')
    
    # plots BGR colours should be fixed. 
    #plt.imsave(saveImageFolder + "hand.tiff", img, cmap=plt.cm.gray)
    
    
    def localMax(x, min_distance=1):
        
        return corner_peaks(x, min_distance=min_distance)
    
    
    i = 0
    j = 0 
    
    sigma = [2,4,8]
    Kthresholds = [15]
    K = list(map(lambda x: x/1e2, Kthresholds))
    
    h = 0
    min_distance = [1]
    

    
    # the values of eps choosen produces garbage. need to ask for better values
    epsthreshold = [10,50,100]
    eps = list(map(lambda x: x/1e7, epsthreshold))

    for i in range(len(sigma)): 
        for j in range(len(K)):
                res = corner_harris(im, sigma = sigma[i], k = K[j], method='k')
                peaks = localMax(res, min_distance[h])
                
                fig = plt.figure()
                ax = plt.subplot(1,1,1)
                ax.plot(peaks[:,0],peaks[:,1], 'ob')
                ax.imshow(img, cmap=plt.cm.gray)

                ax.axis('off')
                title = r'Harris_corner $\sigma={},k={}$, Corner_Peaks $mindist={}$'.format(sigma[i], Kthresholds[j], min_distance[h ])
                ax.set_title(title, fontsize=11)
                
                fig.tight_layout()
                
                filename = "exer13-" + title
        
                plt.savefig(saveImageFolder + filename)    
        
    
    pass


def main():
    
    testImageFolder = "./Week 6/"

    saveImageFolder = "./exer1Images/"
    
    exer11(testImageFolder,saveImageFolder)
    # exer12(testImageFolder,saveImageFolder)
    # exer13(testImageFolder,saveImageFolder)

if __name__ == "__main__":
    
    main()