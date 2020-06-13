import numpy as np
import matplotlib.pyplot as plt

from utils import *
import pickle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import trimesh


def decode(imprefix, start, threshold):
    '''

    :param imprefix: the image prefix of the directory
    :param start: the index of the start image
    :param threshold: the threshold to determine the pixel is undecodable or not
    :return: decode code, mask that implies the pixel is undecodable or not
    '''

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
    '''
    :param imprefix: the image directory to the color image of the background and foreground
    :param thresh: the threshold to determine the pixel belongs to background or not
    :return: a binary mask that implies each pixels belongs to foreground or not
    '''
    back = plt.imread(imprefix + '00.png')
    fore = plt.imread(imprefix + '01.png')

    back = np.dot(back[..., :3], [0.2989, 0.5870, 0.1140])
    back = back.astype(np.float32)

    fore = np.dot(fore[..., :3], [0.2989, 0.5870, 0.1140])
    fore = fore.astype(np.float32)
    mask = np.abs(back - fore) > thresh
    return mask


def reconstruct(dir, imprefix_bgL, imprefix_bgR, imprefixL, imprefixR, decode_thresh, camL, camR, bg_thresh=0.05):
    '''
    :param dir: basic image directory
    :param imprefix_bgL: prefix of background color image of left camera
    :param imprefix_bgR: prefix of background color image of right camera
    :param imprefixL: prefix of decoded image of left camera
    :param imprefixR: prefix of decoded image of right camera
    :param decode_thresh: the threshold to determine if the pixel is decodeable or not
    :param camL: Camera Object of the left camera
    :param camR:Camera Object of the right camera
    :param bg_thresh: background threshold
    :return: pts2L, pts2R, pts3
    '''
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
    '''
    :param boxlimits: The boundary of the box that the object is inside the box
    :param pts2L: (n, 2) array of left coordinates
    :param pts2R: (n, 2) array of right coordinates
    :param pts3: (n, 3) array of the triangulate results coordinates
    :return: pts2L_prune, pts2R_prune
    '''
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
    '''
    :param tri: the triangle mesh in the shape (3, # of triangles)
    :param pts3: 3D coordinate of object (# of points, 3)
    :param trithresh: largest length of the triangle edges to keep
    :return: new tirangle mesh, new pts3, and the index of remaing vertices
    '''
    tri_new = []
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        distance = np.sum((pts3[:,tri[:,i]] - pts3[:,tri[:,j]]) ** 2, 0) ** (1 / 2)
        mask = distance < trithresh
        tri = tri[mask,: ]

    tokeep = np.unique(tri)

    new_mask_index = np.full(pts3.shape[1], -1)
    new_mask_index[tokeep] = np.arange(tokeep.shape[0])

    tri_new_get_map = new_mask_index[tri]

    return tri_new_get_map, pts3.T[tokeep].T, tokeep

def smooth(pts3, tri, iteration=1):
    '''
    :param pts3:
    :param tri:
    :param iteration: number of iterations to smooth
    :return: pts3, new triangle mesh
    '''
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
    '''
    :param imgdir: tuple of directory related information
        (dir, imprefixL, imprefixR, imprefix_bgL, imprefix_bgR )

    :param thresh: tuple of all the threshold values
        (decode_thresh, bg_thresh, trithresh, boxlimits)

    :param cam: tuple of left and right camera
        (camL, camR)

    :return: tuple of meshing result after pipeline
        (pts2L_prune, pts2R_prune, new_pts3, new_tri, colorL, colorR, color)

    '''
    dir, imprefixL, imprefixR, imprefix_bgL, imprefix_bgR = imgdir
    decode_thresh, bg_thresh, trithresh, boxlimits = thresh
    camL, camR = cam

    pts2L, pts2R, pts3 = reconstruct(dir, imprefix_bgL, imprefix_bgR, imprefixL, imprefixR, decode_thresh, camL, camR, bg_thresh)


        # fid = open('data/render/color.pickle', 'wb')
        # pickle.dump((colorL, colorR), fid)
        # fid.close()

    vis_scene(camL, camR, pts3, looklength=6, boxlimits=None)

    pts2L_prune, pts2R_prune = box_pruning(boxlimits, pts2L, pts2R, pts3)
    pts3_prune_tri = triangulate(pts2L_prune, camL, pts2R_prune, camR)
    tri = Delaunay(pts2L_prune.T).simplices
    new_tri, new_pts3, tokeep = triangle_pruning(tri, pts3_prune_tri, trithresh)

    colorimg_L = plt.imread(dir + imprefix_bgL + '01.png')
    colorimg_R = plt.imread(dir + imprefix_bgR + '01.png')

    pts2L_prune = pts2L_prune.T[tokeep].T
    pts2R_prune = pts2R_prune.T[tokeep].T
    colorL = colorimg_L[pts2L_prune[1, :], pts2L_prune[0, :]]
    colorR = colorimg_R[pts2R_prune[1, :], pts2R_prune[0, :]]

    color = np.average((colorL, colorR), axis=0)

    return pts2L_prune, pts2R_prune, new_pts3, new_tri, colorL, colorR, color

