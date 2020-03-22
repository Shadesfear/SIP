#!/usr/bin/env python3
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def exer1():
    def hough_line(image):
        """
        Does the hough transform on an image,  the image
        should already be a binary image.
        If not do the canny edge detector

        Parameters
        ----------
        image : 2d np.array

        Returns
        -------
        lines: 2d np.array

        """

        # All possible angles and rhos
        thetas = np.deg2rad(np.arange(-90.0, 90.0))
        w, h = image.shape
        max_dist = int(np.ceil(np.sqrt(w**2 + h**2)))
        rhos = np.linspace(-max_dist, max_dist, max_dist*2)
        print(rhos.shape, thetas.shape)

        # Save shit for later
        theta_cos = np.cos(thetas)
        theta_sin = np.sin(thetas)
        num_thetas = len(thetas)

        result = np.zeros((2 * max_dist, num_thetas), dtype=np.uint64)

        # Only get the non zero indexes of the image
        yidx, xidx = np.nonzero(image)

        for i in range(len(xidx)):
            x = xidx[i]
            y = yidx[i]

            for theta_idx in range(num_thetas):
                rho = int(round(x * theta_cos[theta_idx] +
                                y * theta_sin[theta_idx]) +
                          max_dist)
                result[rho, theta_idx] += 1

        return result, thetas, rhos

    cross = io.imread('Week_7_export/cross.png')
    r, t, d = hough_line(cross)
    origin = np.array((0, cross.shape[1]))

    ind = np.argpartition(r.flatten(), -2)[-2:]
    dist = d[ind // r.shape[1]]
    theta = t[ind % r.shape[1]]

    fontsize = 16

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(r, cmap = 'gray')
    ax[0].set_xlabel('Angle', fontsize = fontsize)
    ax[0].set_ylabel('Distance', fontsize = fontsize)
    ax[0].set_title('Hough Transform', fontsize = fontsize)
    for i in range(len(ind)):
        y0, y1 = (dist[i] - origin * np.cos(theta[i])) / np.sin(theta[i])
        ax[1].plot(origin, (y0, y1), '-r')
    ax[1].imshow(cross, cmap='gray')
    ax[1].axis('off')

    ax[1].set_title('Detected lines', fontsize = fontsize)
    plt.savefig('images/hough_cross.pdf')


exer1()
