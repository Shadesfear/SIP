#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:21:11 2020

This file holds the solution to section 1 for assingment 6 SIP
"""

from  skimage.feature import canny
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
    ## plots BGR colours should be fixed. 
    plt.imsave(saveImageFolder + "hand.tiff", img)
    
    

    
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