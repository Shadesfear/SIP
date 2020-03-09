#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:49:58 2020

@author: Magnus
"""

from torchvision import models
import torch

import torch as tc
import numpy as np
import matplotlib.pyplot as plt
#with open('./week 5/SIPdiatomsTrain.txt') as f:
#
#    array = []
#    cnt=0;
#    for line in f: # read  lines
#        array.append([float(x) for x in line.split()])
# 

def printf(statement:str):
    if False:
        printf(statement)

  

data = np.loadtxt('./week 5/SIPdiatomsTrain.txt', delimiter=',')
  
xdata = data[:,0:179:2]
ydata = data[:,1:180:2]

len(ydata)

xdata2 = data[:,0:len(data)-1:2]
ydata2 = data[:,1:len(data):2]



def procrastinate(data,reference):
    """
    procrastiante function
    
    Parameters
    data: TYPE two dimensional
    """
    
    if not((data.shape == reference).all):
        raise "procrastinate: Data and referce must have the same shape"

    arrX = data[:,0]
    arrY = data[:,1]
    
    refX = reference[:,0]
    refY = reference[:,1]
    
    ## t = mean(Y) - mean(x)
    t = np.subtract(np.mean(reference), np.mean(data))
    
    printf("ref{}\ndata{}".format(np.mean(reference),  np.mean(data)))
    printf("t{}".format(t))
    
    #step one update
    translated_data = data + t
    
    printf("test1{}".format((np.equal(reference, translated_data).all())))
    
    #
    ##Step update 
    #arrX = np.subtract(arrX,step1xdata)
    #arrY = np.subtract(arrY,step1ydata)
    
    #step2 scaleing
    
    nominator = np.sum(np.dot(np.transpose(data), reference))
    denominator = np.sum(np.dot(np.transpose(reference), reference))
    
    s = nominator/denominator
    
    #arrX = s * arrX
    #arrY = s * arrY
    
    scaled_data = s * translated_data
    
    printf("test2{}".format((np.equal(reference, scaled_data).all())))
    
    
    # Rotation
    
    
    #ones = np.ones_like(arrX)
    
#    inhomoarr = np.dstack((arrX, arrY))[0]
#    
#    inhomoref = np.dstack((refX, refY))[0]
#    
    
    productXY = np.dot(scaled_data, np.transpose(reference))
    
    printf(productXY.shape)
    
    # singular value decomposition
    u, _, v = np.linalg.svd(productXY)
    
    # matix which minimizes the total squared error. 
    r = np.dot(u, v)
    # step rotate 
    
    rotated_data = np.dot(r, scaled_data)
    
    printf("test3{}".format((np.allclose(reference, rotated_data))))
    printf((np.allclose(reference, rotated_data)))
    
    return rotated_data



def ProcrastinateArray(data, referenceelement:int):
       
    r, c = data.shape
    
    xdata = data[:,0:c-1:2]
    ydata = data[:,1:c:2]


    res = data.copy()
    
    reference = data[referenceelement,:]


    reference = np.dstack((xdata[referenceelement],ydata[referenceelement]))[0]

    for i in range(r):
        
        data = np.dstack((xdata[i],ydata[i]))[0]
        
        output = procrastinate(data,reference)

        res[i,0:c-1:2] =  output[:,0]
        res[i,1:c:2] = output[:,1]
        
    return res 
    


def test1():
    # used as the index to procrastinate around
    base = 0
    index = 1
    
    
    refX = xdata[base,:]
    refY = ydata[base,:]
    
    arrX = xdata[index,:]
    arrY = ydata[index,:]
    
    reference = np.dstack((refX,refY))[0]
    data = np.dstack((arrX, arrY))[0]
    
    
    output = procrastinate(reference, reference)
    
    
    isEqual = np.allclose(output, reference)

    print(isEqual)




def test2():
    # used as the index to procrastinate around
    
    refX = xdata[base,:]
    refY = ydata[base,:]
    
    
    
    reference = np.dstack((refX,refY))[0]
    
    data = np.dstack((reference, reference))[0]
    
    
    
    output = ProcrastinateArray(data, 0)
    
    
    isEqual = np.allclose(output, data)

    print(isEqual)




def main():
    
    test1()
    test2()
    print("done")

if __name__ == "__main__":
    main()
