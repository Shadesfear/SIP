#!/usr/bin/env python3

import timeit
import numpy as np
import matplotlib
from matplotlib.pyplot import axis, imshow, subplot, savefig, figure, close, gca


def time_function(func, static_inp, inp_list, times=10, verbose=False):

    iterations = len(inp_list)
    results = np.zeros(iterations)

    for i in range(iterations):
        end_time = 0
        if verbose:
            print("Running input: {}".format(inp_list[i]))

        for j in range(times):
            start_time = timeit.default_timer()
            func(static_inp, inp_list[i])
            end_time += timeit.default_timer() - start_time
        results[i] = end_time
    return {"cumres":  results, "avgres": (results / times)}


if __name__ == "__main__":
    def firkant(x):
        return x**2


    print(time_function(firkant, [10**i for i in range(190,200)], times = 1000, verbose=True))




def plotImage(ax, img, title, titleSize = 10, axesOn = False, gray = False):

    if gray == False:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')
    
    gca().set_title(title, fontsize=titleSize)
   
    if axesOn: 
        axis('on')
    else: 
        axis('off')
    return ax






def savefig1(filename):
    savefig(filename, dpi=500 , bbox_inches = 'tight')





def pf(statement):
    if False:
        print(statement)