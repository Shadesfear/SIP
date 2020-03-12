#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:21:11 2020

This file holds the solution to section 1 for assingment 6 SIP
"""

from  skimage.feature import canny
import matplotlib.pyplot as plt




def exer21(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        Provide a code snippet and explanation of your solution as well
        as illustrations of your solution.
    """  
    
    # small test to see if the folders where correct
    img = plt.imread(testImageFolder + "hand.tiff")
    ## plots BGR colours should be fixed. 
    plt.imsave(saveImageFolder + "hand.tiff", img)
    
    

    
    pass




def exer23iii(testImageFolder,saveImageFolder):
    """
     Deliverables: 
         Confirm your result in Python by plotting H(0,0,τ) 
         as a function of τ using the expression from 2.3.i.
    """

    pass


def exer23iv(testImageFolder,saveImageFolder):
    """
    Deliverables: 
        An illustration of your results and code snippets 
        showing essential steps in your implementation. 
        What image structure does maxima of eq. (6) represent, 
        and what image structure does minima represent?
    """
    
    pass


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

    saveImageFolder = "./imageResults/"
    
    exer21(testImageFolder,saveImageFolder)
    exer23iii(testImageFolder,saveImageFolder)
    exer23iv(testImageFolder,saveImageFolder)
    exer24(testImageFolder,saveImageFolder)
    exer25iii(testImageFolder,saveImageFolder)
    exer25iv(testImageFolder,saveImageFolder)

if __name__ == "__main__":
    
    main()