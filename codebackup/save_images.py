# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 11:30:47 2020

@author: Ruibo
"""

import os
import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from scipy.linalg import norm
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import cv2
import seaborn as sns
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import Resequence
#%%

def _get_cell_lines_types():
    cell_lines_details = pd.read_excel(
        "../../data/gdsc_v7_Cell_Lines_Details.xlsx", index_col=0)
    cell_lines_details.index = cell_lines_details.index.str.replace("-", "")
    cell_lines_details = cell_lines_details.loc[cell_lines_details['Gene_Expression'] == 'Y']
    cell_lines_types = cell_lines_details[['GDSC_Tissue_descriptor_1',
                                           'GDSC_Tissue_descriptor_2', 'Growth_Properties']]
    return cell_lines_types


def select_cancer(type_list=['glioma', 'melanoma']):
    if isinstance(type_list, str):
        type_list = [type_list]
    types = _get_cell_lines_types()
    tissues = types['GDSC_Tissue_descriptor_2']
    masks = tissues.isin(type_list)
    idx = masks[masks].index.drop_duplicates(keep='first')
    return idx.tolist()


def get_file_list(idx_list):
    rt = []
    for each in idx_list:
        rt.append('MDS_'+each+'.npy')
    return rt


def write_video(video_filename, file_list, path_list, hw=None, fps=5, resz=20):
    rearranged = [file_list[i] for i in path_list]
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    out = cv2.VideoWriter('./output/' + video_filename, fourcc, fps, (hw*resz, hw*resz), True)
    # new frame after each addition of water
    for each_f in tqdm(rearranged):
        each_img = np.load(each_f)
        imgs = each_img.reshape(hw, hw)
        # imgs = cv2.normalize(imgs, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        imgs = cv2.resize(imgs, (hw*resz, hw*resz), interpolation=cv2.INTER_NEAREST).reshape(hw*resz, hw*resz, 1).astype(np.uint8)
        b_img = each_img > 160
        im_color = cv2.applyColorMap(imgs, cv2.COLORMAP_COOL)

        # if each_f in mela:
        #     cv2.putText(im_color, 'Melanoma', (20,20), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.7, (0,0,0), 2, cv2.LINE_AA)
        # elif each_f in glio:
        #     cv2.putText(im_color, 'Glioma', (20,20), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.7, (0,0,0), 2, cv2.LINE_AA)
        out.write(im_color)
    # close out the video writer
    out.release()

#%%
os.chdir('./Images/Melanoma_KEGG')
files = [x for x in os.listdir() if (x.endswith('npy') and not x.startswith('_'))]
mela = get_file_list(select_cancer('melanoma'))
glio = get_file_list(select_cancer('glioma'))
files = mela + glio
n_files = len(files)
hw = 30
#%% use Resequence
img_list = []
for each in files:
    img_list.append(np.load(each).reshape(hw, hw))
path = Resequence.resequence(img_list, metric='double dilate', path_method='mds',
                             precompute=False, verbose=True, thrs=None)
write_video('Shuffle.avi', files, path, hw, 4, 10)
raise
#%%
comb = list(combinations([x for x in range(n_files)], 2))
dist_mat = np.zeros((n_files, n_files))
for (ii, jj) in tqdm(comb):
    img1 = np.load(files[ii]).reshape(hw, hw)
    img2 = np.load(files[jj]).reshape(hw, hw)

    # Do dialation:
    thrs = 128
    _, img1 = cv2.threshold(img1, thrs, 255, cv2.THRESH_BINARY)
    _, img2 = cv2.threshold(img2, thrs, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2), np.uint8)
    img1 = cv2.dilate(img1, kernel)
    img2 = cv2.dilate(img2, kernel)
    # Do Gaussian bluring
    # img1 = cv2.GaussianBlur(img1,(7,7), 0)
    # img2 = cv2.GaussianBlur(img2,(7,7), 0)


    d = norm(img1 - img2)
    dist_mat[ii, jj] = d
    dist_mat[jj, ii] = d
np.save('./output/distance_matrix_dialation2x', dist_mat)
# And:
dist_mat = np.load('./output/distance_matrix_dialation2x.npy')
linkage = spc.linkage(dist_mat, method='single')
rt = spc.dendrogram(linkage)
rearranged = [files[i] for i in rt['leaves']]
#%%
# initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'MP42')
fps = 2
video_filename = './output/output_dialation.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (hw*10, hw*10), True)

# new frame after each addition of water
for each_f in tqdm(rearranged):
    each_img = np.load(each_f)
    imgs = each_img.reshape(hw, hw)
    imgs = cv2.normalize(imgs, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imgs = cv2.resize(imgs, (hw*10, hw*10), interpolation=cv2.INTER_NEAREST).reshape(hw*10, hw*10, 1).astype(np.uint8)
    im_color = cv2.applyColorMap(imgs, cv2.COLORMAP_COOL)

    out.write(im_color)

# close out the video writer
out.release()
#%%
raise
#%% load images together
load_arrays = []
for each in files:
    load_arrays.append(np.load(each).reshape(1, -1))
arrays = np.vstack(load_arrays)

dist = pdist(arrays, metric='euclidean')
linkage = spc.linkage(dist, method='average')
rt = spc.dendrogram(linkage)
rearranged = arrays[rt['leaves'], :]
rearranged = minmax_scale(rearranged, feature_range=(0, 255))
#%%

# initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'MP42')
fps = 2
video_filename = './output/output_euclid.avi'
out = cv2.VideoWriter(video_filename, fourcc, fps, (hw*10, hw*10), True)

# new frame after each addition of water
for each_img in tqdm(rearranged):
    imgs = each_img.reshape(hw, hw)
    imgs = cv2.normalize(imgs, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imgs = cv2.resize(imgs, (hw*10, hw*10),interpolation=cv2.INTER_NEAREST).reshape(hw*10, hw*10, 1).astype(np.uint8)
    im_color = cv2.applyColorMap(imgs, cv2.COLORMAP_COOL)

    out.write(im_color)

# close out the video writer
out.release()

