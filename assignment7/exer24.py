#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file to answer question two in group assingment 7
"""


from skimage import io
import numpy as np
import matplotlib.pyplot as plt


def exer2_4():
    patches = np.load("exer23.npy")
        
    _,classes =    patches.shape 
    
    def dice(predictions, thruth):
        r,c = thruth.shape
        
        intersection = np.equal(predictions, thruth)
        intersect = np.sum(intersection)
        
        dice = 2 * intersect/ (r*c*2)
    
        return dice
    test_seg = io.imread("./Week_7_export/test_images/seg/1003_3_seg.png")
    
    value = dice(patches,test_seg)
    print("dice value{}".format(value))

if __name__ == "__main__":
    exer2_4()