# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:15:52 2020

@author: Ruibo
"""
import numpy as np
import pandas as pd
import cv2
import os
#%%
def _get_cell_lines_types():
    cell_lines_details = pd.read_excel(
        "../data/gdsc_v7_Cell_Lines_Details.xlsx", index_col=0)
    cell_lines_details.index = cell_lines_details.index.str.replace("-", "")
    cell_lines_details = cell_lines_details.loc[cell_lines_details['Gene_Expression'] == 'Y']
    cell_lines_types = cell_lines_details[['GDSC_Tissue_descriptor_1',
                                           'GDSC_Tissue_descriptor_2', 'Growth_Properties']]
    return cell_lines_types

def select_cancer(type_list=['glioma', 'melanoma']):
    if isinstance(type_list, str):
        type_list = [type_list]
    types = _get_cell_lines_types()
    tissues = types['GDSC_Tissue_descriptor_1']
    masks = tissues.isin(type_list)
    idx = masks[masks].index.drop_duplicates(keep='first')
    return idx.tolist()

def get_file_list(idx_list):
    rt = []
    for each in idx_list:
        rt.append('MDS_' + str(each) + '.npy')
    return rt


def write_video(video_filename, file_list, sequence,
                hw=None, fps=5, resz=20,
                binarize_thrs=None,
                text_groups=None,
                gamma=1, cm=cv2.COLORMAP_COOL):
    """
    Text list: pass a list to annotate images.
    sequence: a list of the index of the item sequence in a path. e.g. [3, 2, 4, 1, 0]
    binarize threshold: if not None, output video will be binarized. (For better debug)
    text_groups: a list of annotations, will be put on each frame of generated video.
    gamma: gamma correction.
    cm: cv2 colormap.
    """
    sample_img = np.load(file_list[0])
    hw = sample_img.shape[0]
    rearranged = [file_list[i] for i in sequence]
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    if not os.path.exists("./output/"):
        os.mkdir("./output")
    out = cv2.VideoWriter('./output/' + video_filename, fourcc, fps, (hw*resz, hw*resz), True)
    if text_groups is not None:
        rev_txt = text_groups[::-1]
    # Gamma correction
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    #临时的
    # for i in range(256):
    #     lookUpTable[0, i] = np.clip((255 * ((i-127)/127)**3 + 127), 0, 255)
    # new frame after each addition of water
    for each_f in rearranged:
        each_img = np.load(each_f)
        imgs = each_img.reshape(hw, hw)
        if binarize_thrs is not None:
            imgs = (imgs > binarize_thrs)*255
        # imgs = cv2.normalize(imgs, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        imgs = cv2.resize(imgs, (hw*resz, hw*resz), interpolation=cv2.INTER_NEAREST).reshape(hw*resz, hw*resz, 1).astype(np.uint8)
        imgs = cv2.LUT(imgs, lookUpTable)
        im_color = cv2.applyColorMap(imgs, cm)

        t_scale = int(hw*resz/550) + 1
        if text_groups is not None:
            cv2.putText(im_color, rev_txt.pop(), (t_scale*20, t_scale*30), cv2.FONT_HERSHEY_SIMPLEX,
                        t_scale/2, (0,0,0), t_scale, cv2.LINE_AA)

        out.write(im_color)
    # close out the video writer
    out.release()

def write_img_list(video_filename, image_list, hw=None, fps=5, resz=20,
                binarize_thrs=None,
                text_groups=None,
                gamma=1, cm=cv2.COLORMAP_COOL):
    """
    Text list: pass a list to annotate images.
    sequence: a list of the index of the item sequence in a path. e.g. [3, 2, 4, 1, 0]
    binarize threshold: if not None, output video will be binarized. (For better debug)
    text_groups: a list of annotations, will be put on each frame of generated video.
    gamma: gamma correction.
    cm: cv2 colormap.
    """
    w = image_list[0].shape[1]
    h = image_list[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    out = cv2.VideoWriter('./output/' + video_filename, fourcc, fps, (w*resz, h*resz), True)
    if text_groups is not None:
        rev_txt = text_groups[::-1]
    # Gamma correction
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    #临时的
    # for i in range(256):
    #     lookUpTable[0, i] = np.clip((255 * ((i-127)/127)**3 + 127), 0, 255)
    # new frame after each addition of water
    blk_f = np.zeros((w*resz, h*resz, 3)).astype(np.uint8)
    out.write(blk_f)
    for imgs in image_list:
        if binarize_thrs is not None:
            imgs = (imgs > binarize_thrs)*255
        # imgs = cv2.normalize(imgs, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        imgs = cv2.resize(imgs, (w*resz, h*resz), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        imgs = cv2.merge([imgs])
        imgs = cv2.LUT(imgs, lookUpTable)
        im_color = cv2.applyColorMap(imgs, cm)

        t_scale = int(h*resz/550) + 1
        if text_groups is not None:
            cv2.putText(im_color, rev_txt.pop(), (t_scale*20, t_scale*30), cv2.FONT_HERSHEY_SIMPLEX,
                        t_scale/2, (0,0,0), t_scale, cv2.LINE_AA)

        out.write(im_color)
    # close out the video writer
    out.release()