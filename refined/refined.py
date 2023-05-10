"""


"""
import os
from os.path import join as pj
import math
import numpy as np
import pandas as pd
import pickle
import json
import re

import sklearn.manifold as mnf
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale
from imageio import imwrite as imsave
import textwrap

from myToolbox import ToolBox
from myToolbox.Stat import normalize_df
from myToolbox.Io import read_df, check_path_exists, load_int_dict_from_json
from refined.ImageIsomap import ImageIsomap
from refined.assignment import two_d_norm, two_d_eq
from refined.assignment import assign_features_to_pixels, lap_scipy

#%%
class Refined(object):
    def __init__(self, dim_reduction='MDS', distance_metric='correlation',
                 assignment='lap', hw=None, working_dir='.', verbose=True, seed=None, **kwarg):
        """
        Params:
        dim_reduction: alg to perform the dim-reduction, 'mds', 'c-iso', 'tsne', ... etc.
        distance_metric: correlation or euclidean;
        assignment: "refined" for the old one (nearest first), 'lap' for scipy linear assignment problem;
        hw: the height and width (assume square for now). If hw=None, minimum hw (the most compact img) will be used.
        seed: MDS, LLE, tSNE have random states.

        """
        self.dist_m = distance_metric.lower()
        self.dim_r = dim_reduction.lower()
        self.hw = hw
        self.assign = assignment.lower()
        self.wd = working_dir
        check_path_exists(working_dir)

        self.verbose = verbose
        self._fitted = False
        self.args = kwarg
        self.seed = seed

    @staticmethod
    def _transform_mapping_dict(init_map, feature_names_list):
        # init_map = self.mapping_obj_array.copy()
        int_map = np.char.strip(init_map.astype(str),'F').astype(int)
        coords = []
        for xx in range(int(np.max(int_map)+1)):  # TODO: changed here. Used to be stable.
            coord_in_arrays = np.where(int_map==xx)
            coord_in_list = [coord_in_arrays[0][0],coord_in_arrays[1][0]]
            coords.append(coord_in_list)
        pos_mat = np.array(coords)
        # self.mapping_dict = pd.DataFrame(pos_mat, index=self.feature_names_list, columns=['x','y']).to_dict('index')
        return pd.DataFrame(pos_mat, index=feature_names_list, columns=['x','y']).to_dict('index')


    def fit(self, original_input: pd.DataFrame, key_param=None):
        """
        Calculate the initial correlations (distances) of features,
        Assign the original positions mapping.
        key_param is n_neighbors for isomap, etc.. or perplexity for tSNE.

        """
        assert isinstance(original_input, pd.DataFrame)
        feature_names_list = original_input.columns.tolist()
        if self.hw is None:
            self.hw = math.ceil(np.sqrt(len(feature_names_list))) # image dimension
        nn = self.hw
        Nn = original_input.shape[1] # Feature amount total

        #%% calculate distances
        if 'corr' in self.dist_m:
            c = np.corrcoef(original_input.T)
            c = np.nan_to_num(c, 0)
            dist_mat = pd.DataFrame(1 - c, index=original_input.columns, columns=original_input.columns)
        elif 'euclid' in self.dist_m:
            dist_mat = pd.DataFrame(
                euclidean_distances(original_input.T), index=original_input.columns, columns=original_input.columns
            )
        else:
            raise ValueError("Tobedone. Initial distance metric. Now Corrs only.")
        original_input = normalize_df(original_input)
        transposed_input = original_input.T # dim-reduc, the fts as instances, data T as DF

        #%% Dim reduction:
        if self.verbose:
            print("Dimensional reduction...")
        if 'mds' in self.dim_r:
            mds = mnf.MDS(dissimilarity='precomputed', random_state=self.seed)
            xy = mds.fit_transform(dist_mat)
        elif 'c-iso' in self.dim_r:
            ciso = ImageIsomap(metric='precomputed', n_neighbors=key_param, n_jobs=-1, cisomap=True)  # default 25
            xy = ciso.fit_transform(dist_mat)
        elif 'isomap' in self.dim_r:
            isomap = mnf.Isomap(metric='precomputed', n_neighbors=key_param, n_components=2,
                    eigen_solver='dense', path_method= 'D', n_jobs=3)  # default 25
            xy = isomap.fit_transform(dist_mat)
        elif 'lle' in self.dim_r:
            lle = mnf.LocallyLinearEmbedding(n_neighbors=key_param, random_state=self.seed)  # default 15
            print("LLE doesn't support precomputed dist so far. ")
            xy = lle.fit_transform(transposed_input)
        elif 'tsne' in self.dim_r:
            if key_param is None:
                key_param = 30
            tsne = mnf.TSNE(metric='precomputed', perplexity=key_param, random_state=self.seed)  # default by default
            xy = tsne.fit_transform(dist_mat)
        elif 'pca' in self.dim_r:
            pca = PCA(n_components=2)
            xy = pca.fit_transform(transposed_input)
        elif 'dm' or 'diffusion' in self.dim_r:
            import pydiffmap as dm
            print("By default, using correlation")
            dfmap = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = 'correlation',
                                                            alpha = 0.5, epsilon = 1, k = 32)
            xy = dfmap.fit_transform(transposed_input)
        if self.verbose:
            print(">>>> Dim-reduction done")

        #%% Assign pixels
        if self.assign == 'refined':
            eq_xy = two_d_eq(xy)
            mapping = assign_features_to_pixels(eq_xy, nn, verbose=self.verbose, output_dir=self.wd)
        elif self.assign == 'lap':
            mapping = lap_scipy(xy, nn, verbose=self.verbose, output_dir=self.wd)

        try:
            InitCorr(dist_mat, mapping, nn)
        except ValueError:
            print("Refined fitting: Initial Corrs unavailable. ")
        #%%
        self.feature_names_list = feature_names_list
        self.dist_mat = dist_mat
        self.mapping_obj_array = mapping
        self.mapping_dict = self._transform_mapping_dict(self.mapping_obj_array.copy(), self.feature_names_list)
        self._fitted = True
        return self


    def save_mapping_for_hill_climbing(self, f_name='to_hill_climbing.pickle'):
        try:
            with open(pj(self.wd, f_name),'wb') as file:
                pickle.dump([self.feature_names_list,
                    self.dist_mat, self.mapping_obj_array], file)
            print("File saved for hill climbing as {}.".format(pj(self.wd, f_name)))
        except IOError:
            print("IO Error. File not saved for hill climbing.")

    def load_from_hill_climbing(self, f_name='to_hill_climbing.pickle'):
        try:
            with open(f_name, 'rb') as file:
                feature_name_list, pos_mat, init_map = pickle.load(file)
            self._fitted = True
        except:
            print("Cannot read files from Hill climbing. ")
            return None
        self.feature_names_list = feature_name_list
        self.mapping_dict = pd.DataFrame(pos_mat, index=feature_name_list, columns=['x','y']).to_dict('index')
        self.mapping_obj_array = init_map
        return self


    def load_from_string_coords(self, string_f):
        """
        Written for STRING protein interaction network format: feature_name, x (float), y(float)
        """
        tb = read_df(string_f)
        Nn = len(tb)
        if self.hw is None:
            nn = math.ceil(math.sqrt(Nn))
        else:
            nn = self.hw
        self.hw = nn

        self.feature_names_list = tb.index.tolist()
        xy = tb.iloc[:, :2].values

        #%% Assign pixels
        if self.assign == 'refined':
            eq_xy = two_d_eq(xy)
            mapping = assign_features_to_pixels(eq_xy, nn, verbose=self.verbose, output_dir=self.wd)
        elif self.assign == 'lap':
            mapping = lap_scipy(xy, nn, verbose=self.verbose, output_dir=self.wd)

        self.mapping_obj_array = mapping
        self.mapping_dict = self._transform_mapping_dict(self.mapping_obj_array.copy(),
                                                         self.feature_names_list)
        self._fitted = True
        return self


    def generate_image(self, data_df, output_folder='RFD_Images', img_format='npy',
                       normalize_feature=False, zscore=False, zscore_cutoff=5, fill_blank='zeros',
                       random_map=False, white_noise=False):
        """
        data_df: index is sample names, col is feature names.
        output_folder: this is a subfolder's name under the RFD obj working dir.
        img_format: npy will save values in float. if normalize_feature is on, the values will be normed to [0, 255].
                Other formats will save values as int8.
        fill_blank: mean or zeros. for the extra pixels, fill with mean or with zeros.
        random_map: if Ture, will generate the images will a mapping where features are randomly placed.
        white_noise: generate the images with the correct mapping, but gamma noise are filled.
        """
        # check prerequisits
        assert self._fitted, "The REFINED object is not fitted."
        assert isinstance(data_df, pd.DataFrame)
        # make data df
        gene_expression = data_df.copy()
        gene_expression.index = gene_expression.index.map(str) # In cases that index are ints.
        feature_name_list = self.feature_names_list.copy()
        # Check if features_names in REFINED is same to the given DF
        print("Features not found in DF: ", [x for x in feature_name_list if x not in gene_expression.columns])
        gene_expression = gene_expression.reindex(feature_name_list, axis="columns")
        mapping_dict = self.mapping_dict.copy()

        # Debug cases
        if random_map:
            import random
            random.seed(7)
            random.shuffle(feature_name_list)
            # mapping_dict = pd.DataFrame(pos_mat, index=feature_name_list, columns = ['x','y']).to_dict('index')
            df = pd.DataFrame.from_dict(self.mapping_dict, orient='index', columns=['x', 'y'])
            df.index = feature_name_list
            mapping_dict = df.to_dict('index')
        if white_noise:
            white_noise = np.random.gamma(4, 2, size=gene_expression.shape)
            gene_expression = pd.DataFrame(data=white_noise, index=gene_expression.index,
                                           columns=gene_expression.columns, dtype=int)
        # Normalization
        if zscore:
            temp = scale(gene_expression)
            temp = np.clip(temp, -zscore_cutoff, zscore_cutoff)
            gene_expression = pd.DataFrame(temp,
                       index=gene_expression.index, columns=gene_expression.columns).fillna(0)
        elif normalize_feature:
            gene_expression = normalize_df(gene_expression, (0, 255)).fillna(0)

        # Normalize between [0, 255]
        if 'np' not in img_format:
            gene_expression = ToolBox.normalize_int_between(gene_expression, 0, 255)

        # Save images
        img_dir = pj(self.wd, output_folder)
        check_path_exists(img_dir)

        n_img = len(gene_expression)
        debug_i = 1
        for idx, each_cell_line in gene_expression.iterrows():
            # each_cell_line is a Series.
            if self.verbose:
                print("\rGenerating images: {} / {}".format(debug_i, str(n_img)), end='')

            if 'np' in img_format:
                # Img = np.zeros((self.hw, self.hw))
                # or start with 125?
                Img = np.full((self.hw, self.hw), np.nan)
            else:
                # imwrite requires unit8
                # Img = np.zeros((self.hw, self.hw)).astype(np.uint8)
                # or 125?
                Img = np.full((self.hw, self.hw), np.nan).astype(np.uint8)

            # make img
            # for each_feature in each_cell_line.index:
            for each_feature in self.feature_names_list:
                xx = mapping_dict[each_feature]['x']
                yy = mapping_dict[each_feature]['y']
                val = each_cell_line[each_feature]
                Img[xx, yy] = val # note: here is defined as this, x is the rows, y is the cols.

            if "zero" in fill_blank:
                np.nan_to_num(Img, copy=False, nan=0)
            elif fill_blank == "mean":
                np.nan_to_num(Img, copy=False, nan=np.nanmean(Img))
            else:
                raise ValueError("Unknown blank pixel filling method.")

            cell_line_name = each_cell_line.name
            cell_line_name = re.sub(r'[\\/*?:"<>|]', "", cell_line_name)  # invalid chars for file names
            if 'np' in img_format:# and dtype is float: （why have this dtype check?
                np.save(pj(img_dir, str(cell_line_name)), Img)
            else:
                imsave(pj(img_dir, str(cell_line_name) + '.' + img_format), Img)

            debug_i += 1
        print("\n>>> Image generated.")
        return None

    def generate_array(self, item, fill_blank='zeros') -> np.ndarray:
        """
        This method is for coverting only one instance. Right now only get_item is using it.
        If multiple instances is needed, do it later.
        item is a df or a series, with feature names given.
        """
        # check prerequisits
        assert self._fitted, "The REFINED object is not fitted."
        if isinstance(item, pd.DataFrame):
            assert len(item) == 1, "This method is for coverting only one instance."
            item = item.iloc[0]
        item = item.copy()
        item.index = item.index.map(str)
        feature_name_list = self.feature_names_list.copy()
        _not_found_list = [x for x in feature_name_list if x not in item.index]
        if len(_not_found_list) > 0:
            print("Features not found in item: ", _not_found_list)
        mapping_dict = self.mapping_dict.copy()

        Img = np.full((self.hw, self.hw), np.nan)
        for each_feature in self.feature_names_list:
            xx = mapping_dict[each_feature]['x']
            yy = mapping_dict[each_feature]['y']
            val = item.loc[each_feature]
            Img[xx, yy] = val # note: here is defined as this, x is the rows, y is the cols.

        if "zero" in fill_blank:
            np.nan_to_num(Img, copy=False, nan=0)
        elif fill_blank == "mean":
            np.nan_to_num(Img, copy=False, nan=np.nanmean(Img))
        else:
            raise ValueError("Unknown blank pixel filling method.")

        return Img

    def plot_mapping(self, output_dir=None):
        import matplotlib.pyplot as plt
        assert self._fitted
        plt.figure(figsize=(16, 10))
        # hw = self.mapping_obj_array.shape[0]
        hw = self.hw
        mapp = self.mapping_dict
        for txt in self.feature_names_list:
            yy = mapp[txt]['y'] + 0.5
            xx = hw - mapp[txt]['x'] - 0.5
            w_txt = textwrap.fill(txt, 5)  # added.
            plt.text(yy, xx, w_txt,
                    horizontalalignment='center', verticalalignment='center',
                    wrap=True)
        plt.grid()
        plt.xlim(0, hw)
        plt.ylim(0, hw)
        if output_dir is None:
            output_dir = self.wd
        plt.savefig(os.path.join(output_dir, "REFINED_mapping.png"))
        return None

    def save_mapping_to_csv(self):
        output_path = pj(self.wd, "REFINED mapping.csv")
        hw = self.hw
        result = pd.DataFrame(np.zeros((hw, hw)))
        mapp = self.mapping_dict
        for txt in self.feature_names_list:
            yy = int(mapp[txt]['y'])
            xx = int(mapp[txt]['x'])
            result.iloc[xx, yy] = txt

        result.to_csv(output_path)

    def reverse_mapping(self, array):
        pass

    def save_mapping_to_json(self):
        assert self._fitted
        with open(pj(self.wd, "REFINED_mapping.json"), 'w') as f:
            json.dump(self.mapping_dict, f)

    @staticmethod
    def _get_max_hw(mapp: dict):
        max_x = 0
        max_y = 0
        for each_key in mapp:
            if mapp[each_key]['x'] > max_x:
                max_x = mapp[each_key]['x']
            if mapp[each_key]['y'] > max_y:
                max_y = mapp[each_key]['y']
        return max(max_x, max_y)

    def load_from_json(self, path):
        mapp = load_int_dict_from_json(path)
        self.mapping_dict = mapp
        self._fitted = True
        max_hw = self._get_max_hw(mapp) + 1
        if self.hw is None or self.hw < max_hw:
            print("Warning: Image H&W resized to", max_hw)
            self.hw = max_hw
        self.feature_names_list = list(mapp.keys())
        # self.hw = math.ceil(np.sqrt(len(self.feature_names_list)))
        return self


#%%
#%%  InitCorr legacy code
def InitCorr(Dist_mat, parent_pop, NN, BB=0, KK=0):
    # parent_pop 47x47, elements str F***
    # Dist_mat: original distance between feats, np.array?
    FFN = Dist_mat.shape[0] # number of features
    NN2 = NN*NN
    FF = [f'F{i}' for i in range(NN2)]
    FF_FFN = [f'F{i}' for i in range(FFN)]
    PP_MM = np.reshape([[0.0 for y in range(NN2)] for x in range(NN2)],(NN2,NN2))
    PP_mm = []
    for i in range(NN):
        for j in range(NN):
            PP_mm.append([i,j])
    pix_dist = distance.pdist(PP_mm)
    PP_MM = distance.squareform(pix_dist, checks=False)
    Dist_mat_Sq = distance.squareform(Dist_mat, checks=False)
    parent_pop2 = parent_pop          ## Added by Omid
    parent_pop2_vec = np.reshape(parent_pop2,(1,NN2)).tolist()[0]
    Indd2_vec = [parent_pop2_vec.index(i) for i in FF_FFN]
    PP_MM2_vec = PP_MM[np.ix_(Indd2_vec,Indd2_vec)]
    CORRr1, P_VAL = pearsonr(Dist_mat_Sq,
                             distance.squareform(PP_MM2_vec, checks=False))
    #CORRr1 = Cor2_arr[1,0]
    print("\nInitial: >>>> ",CORRr1)
    return CORRr1


if __name__ == "__main__":
    pass