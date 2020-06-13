import numpy as np
import matplotlib.pyplot as plt

from utils import *
import pickle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import trimesh

def select_points(dir1, dir2, k):
    colorimg_L = plt.imread(dir1)
    colorimg_R = plt.imread(dir2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax.imshow(colorimg_L)
    ax2.imshow(colorimg_R)
    spointsL = select_k_points(ax, k)
    spointsR = select_k_points(ax2, k)

    points1 = np.array([spointsR.xs, spointsR.ys])
    points2 = np.array([spointsL.xs, spointsL.ys])

    return points1, points2

def find_match(points1, points2, pts2L0, pts2L1, pts30, pts31):
    result1 = []
    result2 = []
    for p in points1.T:
        result = np.argmin(np.sum((pts2L0.T - p) ** 2, axis=1))
        result1.append(result)

    print(points1.T, pts30.T[result1])

    print('the second grab')
    for p in points2.T:
        result = np.argmin(np.sum((pts2L1.T - p) ** 2, axis=1))
        result2.append(result)
    print(points2.T, pts31.T[result2])

    return pts30.T[result1], pts31.T[result2]


def align_residuals(pts30, pts31, params):
    new_pts3 = pts31 @ makerotation(params[0], params[1], params[2]) + params[3:]
    residuals = (new_pts3 - pts30)**2
    return residuals.flatten()


def optimize_align(pts30, pts31,params_init):
    efun = lambda params: align_residuals(pts30, pts31, params)
    popt,_ = scipy.optimize.leastsq(efun,params_init)

    return popt

def remesh(new_params, pts30, pts31, tri0, tri1):
    transform_pts3 = pts31.T @ makerotation(new_params[0], new_params[1], new_params[2]) + new_params[3:]

    tri_merge = np.vstack((tri0, tri1))
    tri_merge = np.unique(tri_merge, axis=1)
    print(tri_merge.shape)
    pts3_merge = np.hstack((pts30, transform_pts3.T))
    pts3_merge = np.unique(pts3_merge, axis=1)
    print(pts3_merge.shape)

    return pts3_merge, tri_merge