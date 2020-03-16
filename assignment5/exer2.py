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
from scipy.spatial import procrustes
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
    
    #step2 scaleing
    
    nominator = np.sum(np.dot(np.transpose(data), reference))
    denominator = np.sum(np.dot(np.transpose(reference), reference))
    
    s = nominator/denominator
    
    scaled_data = s * translated_data
    
    # Rotation
    
    productXY = np.dot(scaled_data, np.transpose(reference))
    
    # singular value decomposition
    u, _, v = np.linalg.svd(productXY)
    
    # matix which minimizes the total squared error. 
    r = np.dot(u, v)
    # step rotate 
    
    rotated_data = np.dot(r, scaled_data)
    
    return 1 , rotated_data, {}





def procrustes_mine(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform
    

def ProcrastinateArray(data, referenceelement:int):

    r, c = data.shape

    xdata = data[:,0:c-1:2]
    ydata = data[:,1:c:2]


    res = data.copy()

    reference = data[referenceelement,:]


    reference = np.dstack((xdata[referenceelement],ydata[referenceelement]))[0]

    for i in range(r):

        data = np.dstack((xdata[i],ydata[i]))[0]


        
        a, output, d = procrastinate(data,reference)
                
        #a, output, d = procrustes_mine(data,reference)
    
        res[i,0:c-1:2] =  output[:,0]
        res[i,1:c:2] = output[:,1]

    return res

def test1():
    # used as the index to procrastinate around
    base = 0
    index = 2
    
    
    refX = xdata[base,:]
    refY = ydata[base,:]
    print('refx', refX.shape)
    
    arrX = xdata[index,:]
    arrY = ydata[index,:]
    
    reference = np.dstack((refX,refY))[0]
    data = np.dstack((arrX, arrY))[0]
    
    
    output = procrastinate(reference, reference)
    print('Output', output.shape)
    # plt.plot(output[:,0])
    # plt.plot(output[:,1])
    plt.plot(refX, refY)
    plt.plot(arrX, arrY)
    plt.plot(output[:,0],output[:,1])
    plt.show()
    
    
    isEqual = np.allclose(output, reference)

    print(isEqual)




def test2():
    # used as the index to procrastinate around

    base = 0
    refX = xdata[base,:]
    refY = ydata[base,:]
    
    
    
    reference = np.dstack((refX,refY))[0]
    
    data = np.dstack((reference, reference))[0]
    
    
    
    output = ProcrastinateArray(data, 0)
    
    
    isEqual = np.allclose(output, data)

    print(isEqual)

def exer21(base, index):
   
    
    
    data = np.loadtxt('./Week 5/SIPdiatomsTrain.txt', delimiter=',')
      
    xdata = data[:,0:179:2]
    ydata = data[:,1:180:2]
    
    
    xdata2 = data[:,0:len(data)-1:2]
    ydata2 = data[:,1:len(data):2]
    



    refX = xdata[base,:]
    refY = ydata[base,:]
    print('refx', refX.shape)

    arrX = xdata[index,:]
    arrY = ydata[index,:]

    reference = np.dstack((refX,refY))[0]
    data = np.dstack((arrX, arrY))[0]


    output, out2, d = procrustes_mine(reference, data)
    # output = procrastinate(reference, reference)
    # print('Output', output.shape)
    fig, ax = plt.subplots()
    ax.plot(arrX, arrY, 'b--', label='Input')
    ax.plot(refX, refY, 'c--', label='Reference')
    ax.plot(out2[:,0], out2[:,1], 'r--', label='Output')
    ax.legend()
    plt.savefig('diatom_transform4.pdf')

    # plt.plot(output[:,0])
    print(output.shape)
    # plt.plot(output[:,1])
    # plt.plot(arrX, arrY)
    # plt.plot(output[:,0],output[:,1])
    # plt.plot(out2[:,0],out2[:,1])

    plt.show()


def exer22():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    XTrain = np.loadtxt('./Week 5/SIPdiatomsTrain.txt', delimiter=',')
    XTest = np.loadtxt('./Week 5/SIPdiatomsTest.txt', delimiter=',')
    XTrainL = np.loadtxt('./Week 5/SIPdiatomsTrain_classes.txt', delimiter=',')
    XTestL = np.loadtxt('./Week 5/SIPdiatomsTest_classes.txt', delimiter=',')


    knn = KNeighborsClassifier()
    knn.fit(XTrain, XTrainL)
    pred_labels = knn.predict(XTest)
    acc = sum(pred_labels == XTestL) / len(XTestL)

    knn2 = KNeighborsClassifier()
    XTrain_Aligned = ProcrastinateArray(XTrain, 0)
    knn2.fit(XTrain_Aligned, XTrainL)
    pred_labels_2 = knn2.predict(XTest)
    acc_2 = sum(pred_labels_2 == XTestL) / len(XTestL)

    print(acc, acc_2)
    print(accuracy_score(XTestL, pred_labels_2))

def main():
    
    #test1()
    #test2()
    drawProcrastinationIndex(0,3)
    exer22()
    # print("done")




if __name__ == "__main__":
    main()

