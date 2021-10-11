# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:56:33 2019

@author: Ruibzhan

Generate the images from
0) pickle file after hill_climbing
1) pickle file after mapping initialization

save to img files

"""

import pickle
import numpy as np
#from joblib import Parallel, delayed
#import multiprocessing
import os
import pandas as pd
from myToolbox import ToolBox
from myToolbox.Stat import normalize_df
from imageio import imwrite as imsave
#%% Load file
map_dir = "./Gauss_clusters/"

if 0:
    with open(map_dir + 'toImgGeneration_MDS_hc.pickle','rb') as file:
        feature_name_list,pos_mat,init_map = pickle.load(file)
else: # skip the hill climbing
    with open(map_dir + 'mapping_MDS.pickle','rb') as file:
        feature_name_list,dist_matr,init_map = pickle.load(file)
    int_map = np.char.strip(init_map.astype(str),'F').astype(int)
    coords = []
    for xx in range(len(dist_matr)):
        coord_in_arrays = np.where(int_map==xx)
        coord_in_list = [coord_in_arrays[0][0],coord_in_arrays[1][0]]
        coords.append(coord_in_list)
    pos_mat = np.array(coords)

mapping_dict = pd.DataFrame(pos_mat, index = feature_name_list, columns = ['x','y']).to_dict('index')

# {index -> dict {'x':45,'y':67}}
#%% otw
#coord = np.array([[item[0] for item in np.where(int_mapping == ii)] for ii in range(2147)])
#%% Load GeneExpression
with open('./data/GDSC_10fold.pickle', 'rb') as file:
    data = pickle.load(file)
    rna = data['rna_seq']

gene = []
# gene.append(pd.read_csv('./data/HH.csv', index_col=0, header=None))
# gene.append(pd.read_csv('./data/NOTCH.csv', index_col=0, header=None))
# gene.append(pd.read_csv('./data/TGFB.csv', index_col=0, header=None))
gene.append(pd.read_csv('./data/Melanoma.csv', index_col=0, header=None))
kegg = pd.concat(gene, axis=0).iloc[:, 0].unique()
# Filter for the KEGG pathway
gene_expression = rna.reindex(columns=kegg).dropna(axis=1)
#%%
gene_expression = pd.read_csv('./Gauss_clusters/original_image/data.csv', index_col=0)
#%% Generate random mapping
#import random
#random.seed(7)
#random.shuffle(feature_name_list)
#mapping_dict = pd.DataFrame(pos_mat,index = feature_name_list,columns = ['x','y']).to_dict('index')
#% White noise
# white_noise = np.random.gamma(4,2,size = gene_expression.shape)
# gene_expression = pd.DataFrame(data = white_noise,index = gene_expression.index,columns=gene_expression.columns,dtype = int)
#%% Normlize 255
gene_expression = normalize_df(gene_expression, (0, 255))
gene_expression = ToolBox.normalize_int_between(gene_expression, 0, 255)
# or save the
# from sklearn.preprocessing import scale
# gene_expression = pd.DataFrame(scale(gene_expression)*255 + 128,
#                            columns=gene_expression.columns, index=gene_expression.index)
#%% Img. cell_line_names can also be drug names, does not distinguish here.
# import progressbar
# p = progressbar.ProgressBar(max_value=len(gene_expression))
folder_name = 'Melanoma_KEGG'
try:
    os.mkdir('./Images/'+folder_name)
except FileExistsError:
    print("Folder already exists.")
finally:
    os.chdir('./Images/'+folder_name)
#%%
debug_ii = 0

from tqdm import tqdm
for idx, each_cell_line in tqdm(gene_expression.iterrows()):
    # imwrite requires unit8
    Img = np.zeros(init_map.shape).astype(np.uint8)
    # Save np instead
    # Img = np.zeros(init_map.shape)

    cell_line_name = each_cell_line.name
    # Exceptions:
    if cell_line_name == 'PE/CAPJ15':
        cell_line_name = 'PE'
    # save
    for each_feature in each_cell_line.index:
        xx = mapping_dict[each_feature]['x']
        yy = mapping_dict[each_feature]['y']
        val = each_cell_line[each_feature]

        Img[xx,yy] = val # note: here is defined as this, x is the rows, y is the cols.
    # Drug imgs does not accept underscore
    cell_line_name = cell_line_name.replace('_','')
    if 'VNLG124' == cell_line_name:
        cell_line_name = 'VNLG_124'
    # imsave('MDS_'+cell_line_name+'.png',Img)
    np.save('MDS_'+ str(cell_line_name), Img)

    debug_ii += 1
    # p.update(debug_ii)
#%% Map generation
import matplotlib.pyplot as plt
genes = gene_expression.columns.tolist()
# cs = []
# cs.extend(['b']*29)
# cs.extend(['r']*16)
# cs.extend(['g']*10)
hw = 30
for ii in range(len(pos_mat)):
    xx = pos_mat[ii][1] + 0.5
    yy = hw - pos_mat[ii][0] - 0.5
    txt = genes[ii]
    plt.text(xx, yy, txt,
             horizontalalignment='center', verticalalignment='center')
plt.grid()
plt.xlim(0, hw)
plt.ylim(0, hw)


