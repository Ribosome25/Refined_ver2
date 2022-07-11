# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:15:52 2020

@author: Ruibo

write video or save images.

when writing video, the last frame is often skipped. Can be solved by adding a dummy frame in the end.

"""
import numpy as np
import pandas as pd
import cv2
import os
import PIL.Image as Image
from PIL import ImageDraw, ImageFont
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

    imgs = np.zeros((hw*resz, hw*resz), np.uint8)
    im_end = cv2.applyColorMap(imgs, cm)
    out.write(im_end)  # 避免最后一闪而过。还没有测试过。
    # close out the video writer
    out.release()


def _load_img_to_array(path):
    "load np .npy or PIL load images."
    try:
        array = np.load(path)
    except ValueError:
        with Image.open(path) as img:
            array = np.asarray(img)
    return array


def enlarge_images(file_list, output_dir=None, resz=10, text_groups=None, cm=cv2.COLORMAP_COOL, gamma=1):
    """
    Text list: pass a list to annotate images.
    sequence: a list of the index of the item sequence in a path. e.g. [3, 2, 4, 1, 0]
    binarize threshold: if not None, output video will be binarized. (For better debug)
    text_groups: a list of annotations, will be put on each frame of generated video.
    gamma: gamma correction.
    cm: cv2 colormap.
    resz: 放大几倍
    """
    sample_img = _load_img_to_array(file_list[0])
    hw = sample_img.shape[0]
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
    for each_f in file_list:
        each_img = _load_img_to_array(each_f)
        # Normalize between 0-255
        each_img = 255 * (each_img / each_img.max(axis=None))  

        imgs = each_img.reshape(hw, hw)
        imgs = cv2.resize(imgs, (hw*resz, hw*resz), interpolation=cv2.INTER_NEAREST).reshape(hw*resz, hw*resz, 1).astype(np.uint8)
        imgs = cv2.LUT(imgs, lookUpTable)
        im_color = cv2.applyColorMap(imgs, cm)

        t_scale = int(hw*resz/550) + 1
        if text_groups is not None:
            cv2.putText(im_color, rev_txt.pop(), (t_scale*20, t_scale*30), cv2.FONT_HERSHEY_SIMPLEX,
                        t_scale/2, (0,0,0), t_scale, cv2.LINE_AA)
        if output_dir is None:
            out_fname = each_f + '.png'
        else:
            out_fname = os.path.join(output_dir, os.path.split(each_f)[-1]+".png")
        cv2.imwrite(out_fname, im_color)



def concat_images_2d(save_path, image_list, axis=1, n_rows=5, n_cols=5, gap=5):
    """
    图片拼接。
    Assume every pic has the same size.
    https://blog.csdn.net/weixin_44441009/article/details/113925581
    axis =1: col first, then rows. axis=0: row first, then cols.
    """

    if len(image_list) != n_rows * n_cols:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")
    img_size = Image.open(image_list[0]).width  # assume square. 方形图。
    to_image = Image.new('RGB', (n_cols * img_size+gap*(n_cols-1), n_rows * img_size+gap*(n_rows-1)),'white' )  # 创建一个新图

    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    posi = []
    if axis==1:
        for y in range(1, n_rows + 1):
            for x in range(1, n_cols + 1):
                posi.append((x, y))
    else:
        for y in range(1, n_rows + 1):
            for x in range(1, n_cols + 1):
                posi.append((y, x))

    for ii in range(len(image_list)):
        x, y = posi[ii]
        from_image = Image.open(image_list[ii]).resize(
            (img_size, img_size), Image.ANTIALIAS)
        to_image.paste(from_image, ((x - 1) * img_size+gap* (x - 1), (y - 1) * img_size+gap* (y - 1)))

    draw = ImageDraw.Draw(to_image)

    return to_image.save(save_path)  # 保存新图


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