#!/usr/bin/env python3
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.draw import circle_perimeter

from skimage.transform import hough_line, hough_circle, hough_line_peaks, hough_circle_peaks
from skimage import data, color

from skimage.feature import canny


def exer1():
    def hough_line_own(image):
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
    r, t, d = hough_line_own(cross)
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
    plt.close()


    # Nom compare to scikit image's implementation

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(cross, theta=tested_angles)
    fig, axes = plt.subplots(1, 2)
    ax = axes.ravel()

    ax[0].imshow(np.log(1 + h),
                extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                cmap=cm.gray, aspect=1/1.5)
    ax[0].set_title('Hough transform')
    ax[0].set_xlabel('Angles (degrees)')
    ax[0].set_ylabel('Distance (pixels)')
    ax[0].axis('image')

    ax[1].imshow(cross, cmap=cm.gray)
    origin = np.array((0, cross.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[1].plot(origin, (y0, y1), '-r')
    ax[1].set_xlim(origin)
    ax[1].set_ylim((cross.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.savefig('images/hough_cross_skimage.pdf')
    plt.close()



    coins = io.imread('Week_7_export/coins.png')
    # Now do the hough circle with skimage first applying canny
    edges = canny(coins, sigma=5, low_threshold=20, high_threshold=50)

    # Detect two radii
    hough_radii = np.arange(25, 70, 3)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=10)

    # plt.imshow(edges)

    # Draw them

    fig, ax = plt.subplots(ncols=2, nrows=1)
    ax[0].imshow(edges, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Canny Edges', fontsize=fontsize)

    image = color.gray2rgb(coins)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (220, 20, 20)
    ax[1].imshow(image, cmap=plt.cm.gray)
    ax[1].set_title('Detected Circles', fontsize = fontsize)

    ax[1].axis('off')
    plt.savefig('images/hough_circle_skimage.png', quality=100)
    # plt.show()
    plt.close()




exer1()
