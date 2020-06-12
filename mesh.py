import numpy as np
import matplotlib.pyplot as plt

from utils import *
import pickle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import trimesh


def decode(imprefix, start, threshold):

    # we will assume a 10 bit code
    nbits = 10
    I0 = plt.imread(imprefix + '00.png')
    m, n = I0.shape[:2]

    code, mask = np.zeros((m, n)), np.zeros((m, n))
    code_list = []

    for i in range(nbits):
        impair = [plt.imread(f'{imprefix}{2 * i + start:0>2}.png'),
                  plt.imread(f'{imprefix}{2 * i + start + 1:0>2}.png')]
        for i in range(2):
            if len(impair[i].shape) > 2:
                impair[i] = np.dot(impair[i][..., :3], [0.2989, 0.5870, 0.1140])
            impair[i] = impair[i].astype(np.float32)

        code_list.append(impair[0] > impair[1])
        mask += np.abs(impair[0] - impair[1]) < threshold

    for i in range(1, nbits):
        code_list[i] = code_list[i] != code_list[i - 1]

    for i in range(nbits):
        code += code_list[nbits - 1 - i] * (2 ** (i))

    # don't forget to convert images to grayscale / float after loading them in

    return code, mask < 1


def background_mask(imprefix, thresh):
    back = plt.imread(imprefix + '00.png')
    fore = plt.imread(imprefix + '01.png')

    back = np.dot(back[..., :3], [0.2989, 0.5870, 0.1140])
    back = back.astype(np.float32)

    fore = np.dot(fore[..., :3], [0.2989, 0.5870, 0.1140])
    fore = fore.astype(np.float32)
    mask = np.abs(back - fore) > thresh
    return mask


def reconstruct(dir, imprefix_bgL, imprefix_bgR, imprefixL, imprefixR, decode_thresh, camL, camR, bg_thresh=0.05):

    # Decode the H and V coordinates for the two views
    code_LH, mask_LH = decode(dir+imprefixL, 0, decode_thresh)
    code_LV, mask_LV = decode(dir+imprefixL, 20, decode_thresh)
    code_RH, mask_RH = decode(dir+imprefixR, 0, decode_thresh)
    code_RV, mask_RV = decode(dir+imprefixR, 20, decode_thresh)

    # Construct the combined 20 bit code C = H + 1024*V and mask for each view
    code_L = code_LH + 1024 * code_LV
    bg_mask_L = background_mask(dir+imprefix_bgL, bg_thresh)
    mask_L = mask_LH & mask_LV & bg_mask_L
    CL = np.multiply(code_L, mask_L)
    plt.imshow(CL)
    plt.show()

    code_R = code_RH + 1024 * code_RV
    bg_mask_R = background_mask(dir+imprefix_bgR, bg_thresh)
    mask_R = mask_RH & mask_RV & bg_mask_R
    CR = np.multiply(code_R, mask_R)
    plt.imshow(CR)
    plt.show()

    # find intersection and record color
    ints, matchL, matchR = np.intersect1d(CL, CR, return_indices=True)

    xx, yy = np.meshgrid(range(CR.shape[1]), range(CR.shape[0]))
    xx = np.reshape(xx, (-1, 1))
    yy = np.reshape(yy, (-1, 1))
    pts2R = np.concatenate((xx[matchR].T, yy[matchR].T), axis=0)
    pts2L = np.concatenate((xx[matchL].T, yy[matchL].T), axis=0)

    # Now triangulate the points
    pts3 = triangulate(pts2L, camL, pts2R, camR)

    return pts2L, pts2R, pts3


def box_pruning(boxlimits, pts2L, pts2R, pts3):
    x, y, z = pts3
    mask_x = (x < boxlimits[0]) & (x > boxlimits[3])
    mask_y = (y < boxlimits[1]) & (y > boxlimits[4])
    mask_z = (z < boxlimits[2]) & (z > boxlimits[5])
    mask = mask_x & mask_y & mask_z

    pts3_prune = pts3.T[mask].T
    pts2L_prune = pts2L.T[mask].T
    pts2R_prune = pts2R.T[mask].T

    return pts2L_prune, pts2R_prune

def triangle_pruning(tri, pts3, trithresh=20):
    # index_map = np.arange(pts3.shape[1])
    nodes = []

    for i in range(3):
        nodes.append(pts3[:, tri.T[i]])

    tri_new = []
    masks = []
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        distance = np.sum((nodes[i] - nodes[j]) ** 2, 0) ** (1 / 2)
        mask = distance < trithresh
        masks.append(mask)
        tri_new.append(tri.T[i][mask])

    tokeep = np.unique(np.concatenate((tri_new[0], tri_new[1], tri_new[2])))

    mask_index = np.zeros(index_map.shape[0], dtype=int)
    mask_index[tokeep] = 1
    new_mask_index = np.ma.masked_array(index_map, mask_index, fill_value=-1)
    new_mask_index[tokeep] = np.arange(tokeep.shape[0])

    mask_new = masks[0] & masks[1] & masks[2]
    tri_new_get_map = new_mask_index[tri_new[mask_new]]

    return tri_new_get_map, pts3[tokeep]

def triangle_pruning_improved(tri, pts3, trithresh=20):
    # index_map = np.arange(pts3.shape[1])
    # nodes = []
    #
    # for i in range(3):
    #     nodes.append(pts3[:, tri.T[i]])

    tri_new = []
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        distance = np.sum((pts3[:,tri[:,i]] - pts3[:,tri[:,j]]) ** 2, 0) ** (1 / 2)
        mask = distance < trithresh
        tri = tri[mask,: ]

    tokeep = np.unique(tri)

    new_mask_index = np.array([-1]*pts3.shape[1])
    new_mask_index[tokeep] = np.arange(tokeep.shape[0])

    tri_new_get_map = new_mask_index[tri]

    return tri_new_get_map, pts3.T[tokeep].T, tokeep

def simple_smooth(pts3, tri, iteration=1):
    for i in range(iteration):
        m,n = pts3.shape
        result = np.zeros((m, n))
        for i in range(n):
            points = tri[np.where(tri==i)[0],:]
            values = np.unique(points.flatten())
            result[:,i] = np.mean(pts3[:,values], axis=1)
        pts3 = result
    return pts3, tri


def mesh_all(imgdir, thresh, cam):
    dir, imprefixL, imprefixR, imprefix_bgL, imprefix_bgR = imgdir
    decode_thresh, bg_thresh, trithresh, boxlimits = thresh
    camL, camR = cam

    pts2L, pts2R, pts3 = reconstruct(dir, imprefix_bgL, imprefix_bgR, imprefixL, imprefixR, decode_thresh, camL, camR, bg_thresh)


    colorimg_L = plt.imread(dir + imprefix_bgL + '01.png')
    colorimg_R = plt.imread(dir + imprefix_bgR + '01.png')
    colorL = colorimg_L[pts2L[1, :], pts2L[0, :]]
    colorR = colorimg_R[pts2R[1, :], pts2R[0, :]]

        # fid = open('data/render/color.pickle', 'wb')
        # pickle.dump((colorL, colorR), fid)
        # fid.close()

    vis_scene(camL, camR, pts3, looklength=6, boxlimits=None)

    pts2L_prune, pts2R_prune = box_pruning(boxlimits, pts2L, pts2R, pts3)
    pts3_prune_tri = triangulate(pts2L_prune, camL, pts2R_prune, camR)
    tri = Delaunay(pts2L_prune.T).simplices
    new_tri, new_pts3, tokeep = triangle_pruning_improved(tri, pts3_prune_tri, trithresh)
    color = np.average((colorL[tokeep], colorR[tokeep]), axis=0)

    return new_pts3, new_tri, colorL[tokeep], colorR[tokeep], color

