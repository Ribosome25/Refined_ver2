# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 20:44:37 2020

@author: Ruibo
"""
import numpy as np
import cv2
from itertools import product
from scipy.linalg import norm
from tsp import travel
from sklearn.manifold import MDS
from sklearn.preprocessing import scale
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import scipy.stats as st
from skimage.metrics import structural_similarity as ssim
from warnings import warn
import tsp_approx
import TSP_Hopfield
from scipy.spatial.distance import directed_hausdorff

#%%
def _threshold_to_binary(img1, img2, thrs=None):
    img1 = img1.astype('uint8')
    img2 = img2.astype('uint8')
    if thrs is None:
        thrs = max(img1.max(), img2.max()) / 2
    _, img1 = cv2.threshold(img1, thrs, 255, cv2.THRESH_BINARY)
    _, img2 = cv2.threshold(img2, thrs, 255, cv2.THRESH_BINARY)
    return img1, img2


def single_conn(img1, img2, thrs=None, k=3, relative=False):
    """
    Connectivity calculation:
    Dilate img1. not img2, return the sum() of intersected.
    """
    img1, img2 = _threshold_to_binary(img1, img2, thrs)
    if isinstance(k, int):
        k = (k, k)
    n_pixels = img2.sum()

    kernel = np.ones(k, np.uint8)
    img1 = cv2.dilate(img1, kernel)
    imgc = np.multiply(img1, img2)

    if relative:
        if n_pixels == 0:
            return 0
        else:
            return imgc.sum()/n_pixels
    return imgc.sum()


def double_conn(img1, img2, thrs=None, k=3, relative=False):
    return max(single_conn(img1, img2, thrs=thrs, k=k, relative=relative),
        single_conn(img2, img1, thrs=thrs, k=k, relative=relative))

#%% Gray scale
from scipy.ndimage import grey_dilation
def grey_dilation_dist(img1, img2, k=3, relative=False):
    if isinstance(k, int):
        k = (k, k)
    norm2 = norm(img2)
    d_img1 = grey_dilation(img1, size=k)
    d_img2 = grey_dilation(img2, size=k)
    dist = norm((d_img1 - d_img2), ord='fro')
    if relative:
        if norm2 == 0:
            return 0
        else:
            return dist/norm2
    return dist

#%%
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size,
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)

def pyramid_conn(img1, img2, max_depth=None, thrs=None, k=3, relative=False):
    """不断做pad = 2,2，  stride=2 的max pooling
    几何距离近的cluster，做出来重叠的次数多。"""
    kwargs = {'thrs':thrs, 'k':k, 'relative':relative}
    if img1.shape[0] <= 2:
        return (img1 * img2).sum()
    if max_depth==0:
        return single_conn(img1, img2, **kwargs)
    else:
        p_img1 = pool2d(img1, kernel_size=4, stride=4, padding=0, pool_mode='max')
        p_img2 = pool2d(img2, kernel_size=4, stride=4, padding=0, pool_mode='max')
        if max_depth is not None:
            max_depth -= 1
        return single_conn(img1, img2, **kwargs) +\
            pyramid_conn(p_img1, p_img2, max_depth=max_depth, **kwargs)

def double_pyramid_conn(img1, img2, max_depth=None, thrs=None, k=3, relative=False):
    kwargs = {'max_depth':max_depth, 'thrs':thrs,
              'k':k, 'relative':relative}
    d1 = pyramid_conn(img1, img2, **kwargs)
    d2 = pyramid_conn(img2, img1, **kwargs)
    return max(d1, d2)

#%% Hausdorff
def hausdorff(img1, img2, directed=True, thrs=None):
    """
    Hausdorff 距离。
    单向是指： img1 中元素， 距 img2 set 全部最小距离们的最大值。
    scipy 库接收坐标，返回是根据欧氏计算的距离， 和对应元素idx。
    """
    img1, img2 = _threshold_to_binary(img1, img2, thrs)
    t1 = np.where(img1)
    t2 = np.where(img2)
    uu = np.array(list(zip(t1[0], t1[1])))
    vv = np.array(list(zip(t2[0], t2[1])))
    if len(uu)==0 or len(vv)==0:
        return np.nan
    hd1 = directed_hausdorff(uu, vv)[0]
    if directed:
        return hd1
    else:
        hd2 = directed_hausdorff(vv, uu)[0]
        return max(hd1, hd2)

def erosion_hausdorff(img1, img2, thrs, k):
    """只写了双向的"""
    img1, img2 = _threshold_to_binary(img1, img2, thrs)
    if isinstance(k, int):
        k = (k, k)

    kernel = np.ones(k, np.uint8)
    eimg1 = cv2.erode(img1, kernel, iterations=1)
    eimg2 = cv2.erode(img2, kernel, iterations=1)
    return hausdorff(eimg1, eimg2, directed=False, thrs=0.5)  # 实际上binaryimage 是0 或255

#%% Gaussian blur
from scipy.ndimage.filters import gaussian_filter
def gaussian_blured(img1, img2, sigma):
    """ The difference of blurred img"""
    b_img1 = gaussian_filter(img1, sigma)
    b_img2 = gaussian_filter(img2, sigma)
    return norm(b_img1 - b_img2, 'fro')

#%% P value thresholding
def binarize_by_pvalue(arrays, p):
    hw = arrays[0].shape
    t = np.asarray(arrays).reshape(len(arrays), -1)
    if len(arrays) < 30:
        warn("Binarize by p-values, less than 30, should be using T-test instead")
    t = scale(t)
    z = - st.norm.ppf(p/2)
    t = (t > z) | (t < -z)
    t = t.reshape(len(arrays), *hw) * 255
    return [x for x in t]

#%%
def list_conn(arrays, connectivity='single', thrs=None, pvalue=None,
              k=3, max_depth=None, relative=False, verbose=False, gauss_sigma=0.5):
    """
    Return the connectivity matrix of a list of arrays.
    connectivity: single 单向连接, double 对称连接, pyramid 池化连接
    """
    if pvalue is not None:
        arrays = binarize_by_pvalue(arrays, pvalue)
    comb = list(product([x for x in range(len(arrays))], repeat=2))
    conn_mat = np.zeros((len(arrays),len(arrays)))
    ii = 1
    for each_comb in comb:
        if verbose:
            print("\r {} / {}".format(ii, len(comb)), end='')
            ii += 1
        xx = each_comb[0]
        yy = each_comb[1]
        if 'euclid' in connectivity.lower() and 'bin' in connectivity.lower():
            conn_mat[xx, yy] = np.logical_and(arrays[xx]>thrs, arrays[yy]>thrs).sum()
        elif 'euclid' in connectivity.lower():
            conn_mat[xx, yy] = 1.41*arrays[xx].shape[0] -\
                norm(arrays[xx] - arrays[yy], 'fro')
        elif 'ssim' in connectivity.lower():
            if arrays[xx].shape[0] < 7:
                w_sz = 5
            else:
                w_sz = None
            conn_mat[xx, yy] = ssim(arrays[xx], arrays[yy], win_size=w_sz)
        elif 'gauss' in connectivity.lower():
            conn_mat[xx, yy] = - gaussian_blured(arrays[xx], arrays[yy], gauss_sigma)
        elif 'haus' in connectivity.lower():
            if 'double' in connectivity.lower():
                d = False
            else:
                d = True
            if "ero" in connectivity.lower():
                conn_mat[xx, yy] = 1.41*arrays[xx].shape[0] - erosion_hausdorff(arrays[xx], arrays[yy],
                        thrs=thrs, k=k)
            else:
                conn_mat[xx, yy] = 1.41*arrays[xx].shape[0] - \
                    hausdorff(arrays[xx], arrays[yy], directed=d, thrs=thrs)
        elif 'pyr' in connectivity.lower():
            if 'double' in connectivity.lower():
                conn_mat[xx, yy] = double_pyramid_conn(arrays[xx], arrays[yy],
                       max_depth=max_depth, thrs=thrs, k=k, relative=relative)
            else:
                conn_mat[xx, yy] = pyramid_conn(arrays[xx], arrays[yy],
                        max_depth=max_depth, thrs=thrs, k=k, relative=relative)
        elif 'gr' in connectivity.lower():
            conn_mat[xx, yy] = - grey_dilation_dist(arrays[xx], arrays[yy],
                                       k=k, relative=relative)

        elif 'sing' in connectivity.lower():
            conn_mat[xx, yy] = single_conn(arrays[xx], arrays[yy],
                                           thrs=thrs, k=k, relative=relative)
        elif 'dou' in connectivity.lower():
            conn_mat[xx, yy] = double_conn(arrays[xx], arrays[yy],
                                           thrs=thrs, k=k, relative=relative)
        # fill nans with the minimum?
        m = np.nanmin(conn_mat)
        conn_mat = np.nan_to_num(conn_mat, nan=m)
    print("\n")
    return conn_mat

import pytsp.christofides_tsp
from sklearn.utils.graph_shortest_path import graph_shortest_path
def find_path(conn_mat, method='mds'):
    # find path from given connectivity matrix.
    d = conn_mat.max() - conn_mat
    d = d / d.max()

    if 'tsp' in method.lower():
        if 'a' in method.lower():
            print("tsp-approximate")
            path_list = tsp_approx.travel(d)
        if 'h' in method.lower():
            path_list = TSP_Hopfield.travel(d, num_iter=200000)
        if 'c' in method.lower():
            d = graph_shortest_path(d)  # 有用吗
            path_list = pytsp.christofides_tsp.christofides_tsp(d)
        else:
            path_list = travel(d)[0]
    elif 'ds' in method.lower():
        mds = MDS(n_components=1, dissimilarity='precomputed')
        xx = mds.fit_transform(d)
        path_list = list(np.argsort(xx[:,0]))
    elif 'clust' in method.lower():
        linkage = spc.linkage(d, method='average')
        rt = spc.dendrogram(linkage)
        path_list = rt['leaves']
    return path_list

def resequence(img_list, metric='double hausdorff', path_method='MDS', precompute=False, **kwargs):
    if precompute:
        try:
            conn = np.load('_precomputed_distance'+metric+'.npy')
            conn = 0.5 * (conn + conn.T)
        except FileNotFoundError:
            conn = list_conn(img_list, connectivity=metric, **kwargs)
            np.save('_precomputed_distance'+metric, conn)
    else:
        conn = list_conn(img_list, connectivity=metric, **kwargs)
    path = find_path(conn, method=path_method)
    return path

import matplotlib.pyplot as plt
import pandas as pd
import ImageIsomap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def two_D_embedding(img_list, text, groups, dim_reduction='MDS', n_dim=2,
                metric='double hausdorff', precompute=False, cisomap=True, **kwargs):
    if precompute:
        try:
            conn = np.load('_precomputed_distance'+metric+'.npy')
            conn = 0.5 * (conn + conn.T)
        except FileNotFoundError:
            conn = list_conn(img_list, connectivity=metric, **kwargs)
            np.save('_precomputed_distance'+metric, conn)
    else:
        conn = list_conn(img_list, connectivity=metric, **kwargs)

    dist = conn.max() - conn
    print("\nResequence.two_D_embedding: dim reduction:")
    if 'mds' in dim_reduction.lower():
        embd = MDS(n_components=n_dim, dissimilarity='precomputed')
    elif 'iso' in dim_reduction.lower():
        embd = ImageIsomap.ImageIsomap(n_neighbors=10, n_components=n_dim, cisomap=cisomap)
    xy = embd.fit_transform(dist)

    f1 = plt.figure(figsize=(8,6))
    if n_dim==2:
        ax1 = f1.add_subplot(111)
        grps = pd.Series(groups)
        handles = []
        # ct = 0
        # cmap = sns.color_palette("Accent", 12)
        for each_group in sorted(grps.unique()):
            each_sub_data = grps == each_group
            each_xy = xy[each_sub_data, :].reshape(-1,2)
            dots = ax1.scatter(each_xy[:, 0], each_xy[:, 1], s=200, alpha=0.6)
            handles.append(dots)
            # ct += 1

        plt.legend(handles, grps.unique())
        for ii in range(len(img_list)):
            plt.text(xy[ii, 0], xy[ii, 1], text[ii].strip(".npy").strip('MDS_'))

    elif n_dim==3:
        ax = f1.add_subplot(111, projection='3d')
        grps = pd.Series(groups)
        handles = []
        for each_group in grps.unique():
            each_sub_data = grps == each_group
            each_xy = xy[each_sub_data, :].reshape(-1,3)
            dots = ax.scatter(each_xy[:, 0], each_xy[:, 1],  each_xy[:, 2],
                               s=200, alpha=0.6)
            handles.append(dots)

        plt.legend(handles, grps.unique())
        # for ii in range(len(img_list)):
        #     plt.text(xy[ii, 0], xy[ii, 1], xy[ii, 2], text[ii].strip(".npy").strip('MDS_'))

        # h1 = ax.scatter(xy[:,0], xy[:,1], xy[:,2], s=2, c=clabel, cmap='jet')
    return f1
#%% dataset generation
def a_list():
    a1 = np.array([[0,0,0], [1,0,0], [0,0,0]])
    a2 = np.array([[0,1,0], [1,1,0], [0,1,0]])
    a3 = np.array([[0,1,0], [0,1,0], [0,1,0]])
    a4 = np.array([[0,1,0], [0,1,1], [0,1,0]])
    a5 = np.array([[0,0,0], [0,0,1], [0,0,0]])
    a6 = np.array([[0,0,0], [0,0,0], [0,0,0]])
    a_list = [a1, a2, a3, a4, a5, a6]
    return a_list

def b_list():
    hw = 5
    bg = np.zeros((hw, hw))
    b_list = []
    for ii in range(2*hw - 1):
        img = bg.copy()
        for xx, yy in product(list(range(hw)), repeat=2):
            if xx + yy == ii:
                img[xx, yy] = 1
        b_list.append(img)
    return b_list

def c_list():
    hw = 7
    bg = np.zeros((hw, hw))
    c_list = []
    for ii in range(1, 7):
        img = bg.copy()
        img[ii, ii-1] = 1
        img[ii-1, ii] = 1
        c_list.append(img)
    return c_list

def d_list():
    hw = 7
    bg = np.zeros((hw, hw))
    k=3
    knl = np.ones((k, k))
    d_list = []
    t = int(hw/2)
    for ii in range(0, 9):
        xx, yy = (int(ii/t)*2, ii%t*k)
        img = bg.copy()
        img[xx:xx+k, yy:yy+k] = 1
        d_list.append(img)
    return d_list
#%%
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns

    the_list = c_list()

    test_list = binarize_by_pvalue(the_list, 0.05)


    ac00 = list_conn(the_list, connectivity='euclidean')
    ac01 = list_conn(the_list, connectivity='single gray', max_depth=1, relative=False)
    ac02 = list_conn(the_list, connectivity='double gray', max_depth=1, relative=False)
    ac11 = list_conn(the_list, connectivity='erosian haus', relative=False)
    ac12 = list_conn(the_list, connectivity='double haus', relative=False)
    ac21 = list_conn(the_list, connectivity='pyra', max_depth=2, relative=False)
    ac22 = list_conn(the_list, connectivity='double pyra', max_depth=2, relative=False)
#%%
    f1 = plt.figure()
    for ii in range(6):
        ax = f1.add_subplot(231+ii)
        ax.set_title('#' + str(ii))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        sns.heatmap(the_list[ii], cbar=False)
#%%
    cn = ac01
    d = cn.max() - cn

    t1 = time.clock()
    from tsp import travel
    print('TSP: ', travel(d)[0])
    p = travel(d)[0]
    t2 = time.clock()
    print('TSP time', t2-t1)


    from sklearn.manifold import MDS
    try:
        mds = MDS(n_components=1, dissimilarity='precomputed')
        xx = mds.fit_transform(d)
        print('MDS / UDS: ', list(np.argsort(xx[:,0])))
    except ValueError:
        pass

    import scipy.cluster.hierarchy as spc
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import fcluster
    linkage = spc.linkage(d, method='average')
    f2 = plt.figure()
    rt = spc.dendrogram(linkage)
    print('Clustering: ', rt['leaves'])

    # d0 = ac00.max() - ac00
    # linkage = spc.linkage(d0, method='single')
    # rt = spc.dendrogram(linkage)
    # print('Euclidean distance -> Clustering: ', rt['leaves'])


    # pp = resequence(the_list, 'hausforff', 'tsp')
    # assert(p == pp)

    # cd1 = list_conn(c_list, 'double')
    # cd2 = list_conn(c_list, 'double pyra', None)
#%%
    f3 = plt.figure(figsize=(3,3))
    sns.heatmap(ac22, annot=True, cbar=False)
    # plt.title('C: Euclidean Connectivity')
    # plt.title("C: Double Dilation Connectivity")
    plt.title("C: Double Pyramid Connectivity")
    # plt.title("C: Double Hausdorff Connectivity")
