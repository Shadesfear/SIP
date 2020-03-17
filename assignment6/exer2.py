#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:21:11 2020

This file holds the solution to section 1 for assingment 6 SIP
"""

import numpy as np
from  skimage.feature import canny
import matplotlib.pyplot as plt
from skimage import io, filters
from scipy.signal.windows import gaussian
from scipy.signal import convolve2d, convolve

from skimage.feature import blob_dog, blob_log, blob_doh
from scipy.ndimage.filters import laplace, gaussian_filter, sobel
from tqdm import tqdm




def exer21(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Provide a code snippet and explanation of your solution as well
        as illustrations of your solution.
    """  
    shape = (512, 512)
    sigma_blob = 10.0
    
    im = np.ones(shape)
    G = gaussian(shape[0], std = sigma_blob)
    GG = np.outer(G,G)
    blob = im* GG
    
    
    sigma_scl = [0., 1., 2., 3., 4., 5., 8., 10.]
    sigma_scale = list(map(lambda x: x**2, sigma_scl))
    
    for i in tqdm(range(len(sigma_scale))):
        
        #create gaussian for scalespace sampling
        Gs = gaussian(50, std = sigma_scale[i])
        GGs = np.outer(Gs,Gs)
        
        blob_scaled = convolve2d(blob, GGs)
        
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        ax.imshow(blob_scaled, cmap=plt.cm.gray)
    
        ax.axis('off')
        title = r'blob w. $\sigma={}$ scalespace w. $\sigma={}$'.format(sigma_blob, sigma_scale[i])
        ax.set_title(title, fontsize=11)
        
        fig.tight_layout()
        
        filename = "exer21-" + 'blob w sigma={}, scale w sigma={}'.format(sigma_blob, sigma_scale[i])
    
        plt.savefig(saveImageFolder + filename + '.png')    
        plt.close

    
    pass




def exer23iii(testImageFolder,saveImageFolder):
    """
     Deliverables: 
         Confirm your result in Python by plotting H(0,0,τ) 
         as a function of τ using the expression from 2.3.i.
    """
    pi=22/7
    sigma = 1
    h = lambda t: -1/(pi**2*(sigma**2+t**2)**2)
    x = np.linspace(0, 10, 10000)

    plt.plot(x, h(x))
    plt.xlabel('t')
    plt.ylabel('H')
    # plt.savefig(saveImageFolder + 'exer23iii.pdf')

    plt.show()
    plt.close()
def exer23iv(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        An illustration of your results and code snippets 
        showing essential steps in your implementation. 
        What image structure does maxima of eq. (6) represent, 
        and what image structure does minima represent?
    """

    img = io.imread(testImageFolder + "sunflower.tiff")
    img = img/255
    sigs = np.linspace(0, 30, 10)
    img_pyramide = [gaussian_filter(img, sig) for sig in sigs]
    img_pyramide_np = np.stack(img_pyramide, axis = 0)

    def detect_all_the_blobs(image, sigma):
        co_ordinates = [] #to store co ordinates
        (s,h,w) = image.shape
        for i in range(1,h):
            for j in range(1,w):
                slice_img = image[:,i-1:i+2,j-1:j+2]
                result = np.amax(slice_img)
                if result >= 0.1:
                    z,x,y = np.unravel_index(slice_img.argmax(),slice_img.shape)
                    co_ordinates.append((i+x-1,j+y-1,sigma[z])) #finding co-rdinates
        return co_ordinates

    coordinates = list(set(detect_all_the_blobs(laplace(img_pyramide_np), sigs)))

    blobs_log = blob_log(np.max(img)-img, max_sigma = 30, num_sigma=10, threshold=.03)
    blobs_log[:,2] = blobs_log[:,2] * np.sqrt(2)

    fig, axes = plt.subplots()
    axes.imshow(img, cmap='gray')
    for blob in blobs_log:
        y,x,r = blob
        c = plt.Circle((x, y), r, color='red' ,linewidth=1.3, fill=False)
        axes.add_patch(c)
    plt.savefig(saveImageFolder + 'blobs_scipy.pdf')


    fig, axes = plt.subplots()
    axes.imshow(img, cmap='gray')
    for blob in coordinates:
        y,x,r = blob
        c = plt.Circle((x, y), r, color='blue' ,linewidth=1.3, fill=False)
        axes.add_patch(c)
    plt.savefig(saveImageFolder + 'blobs_own.pdf')

def exer24(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Provide a code snippet and explanation of your solution 
        as well as illustrations of your solution.
    """
    
    pass



def exer25iii(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Confirm your result in Python by plotting ∥∇J(0,0,τ)∥2 
        as a function of τ using the expression from 2.5.i.
    """
    
    pass




def exer25iv(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        An illustration of your results and code snippets 
        showing essential steps in your implementation.
    """
    
    pass


def main():
    
    testImageFolder = "./Week 6/"

    saveImageFolder = "./exer2Images/"
    
    # exer21(testImageFolder,saveImageFolder)
    # exer23iii(testImageFolder,saveImageFolder)
    exer23iv(testImageFolder,saveImageFolder)
    # exer24(testImageFolder,saveImageFolder)
    # exer25iii(testImageFolder,saveImageFolder)
    # exer25iv(testImageFolder,saveImageFolder)

if __name__ == "__main__":
    
    main()
