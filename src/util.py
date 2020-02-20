#!/usr/bin/env python3

import timeit
import numpy as np


def time_function(func, inp, times=10, verbose=False):

    iterations = len(inp)
    results = np.zeros(iterations)

    for i in range(iterations):
        end_time = 0
        if verbose:
            print("Running input: {}".format(inp[i]))

        for j in range(times):
            start_time = timeit.default_timer()
            func(inp[i])
            end_time += timeit.default_timer() - start_time
        results[i] = end_time
    return {"cumres":  results, "avgres": (results / times)}


if __name__ == "__main__":
    def firkant(x):
        return x**2


    print(time_function(firkant, [10**i for i in range(190,200)], times = 1000, verbose=True))
