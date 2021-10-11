"""


"""
import os
import numpy as np
import pandas as pd
import math
import pickle
import sklearn.manifold as mnf
from sklearn.decomposition import PCA
from imageio import imwrite as imsave
from scipy.stats import pearsonr
from scipy.spatial import distance
from sklearn.preprocessing import scale
from myToolbox import ToolBox
from myToolbox.Stat import normalize_df
from refined.ImageIsomap import ImageIsomap
from refined.assignment import two_d_norm, two_d_eq
from refined.assignment import InitCorr
from refined.assignment import assign_features_to_pixels, lap_scipy
from refined.args import RFDArgs, GenImgArgs, PipelineArgs

#%%
class Refined(object):
    def __init__(self, dim_reduction='MDS', distance_metric='correlation',
                 assignment='lap', verbose=True, **kwarg):
        self.dist_m = distance_metric.lower()
        self.dim_r = dim_reduction.lower()
        self.verbose = verbose
        self.hw = None
        self.assign = assignment
        self._fitted = False
        self.args = kwarg

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
        key_param is n_neighbors or perplexity for tSNE. 

        """
        assert isinstance(original_input, pd.DataFrame)
        feature_names_list = original_input.columns.tolist()
        nn = math.ceil(np.sqrt(len(feature_names_list))) # image dimension
        Nn = original_input.shape[1] # Feature amount total
        if 'corr' in self.dist_m:
            dist_mat = 1 - (original_input.corr()) # dist = 1-correlation
        else:
            raise ValueError("Initial distance metric. Now Corrs only.")
        original_input = normalize_df(original_input)
        transposed_input = original_input.T # dim-reduc, the fts as instances, data T as DF

        #%% Dim reduction:
        if 'mds' in self.dim_r:
            mds = mnf.MDS(dissimilarity='precomputed')
            xy = mds.fit_transform(dist_mat)
        elif 'c-iso' in self.dim_r:
            mds = ImageIsomap(n_neighbors=key_param, n_jobs=-1, cisomap=True)  # default 25
        elif 'isomap' in self.dim_r:
            isomap = mnf.Isomap(n_neighbors=key_param, n_components=2, 
                    eigen_solver='dense', path_method= 'D', n_jobs=3)  # default 25
            xy = isomap.fit_transform(transposed_input)
        elif 'lle' in self.dim_r:
            lle = mnf.LocallyLinearEmbedding(n_neighbors=key_param)  # default 15
            xy = lle.fit_transform(transposed_input)
        elif 'tsne' in self.dim_r:
            tsne = mnf.TSNE(perplexity=key_param)  # default by default
            xy = tsne.fit_transform(transposed_input)
        elif 'pca' in self.dim_r:
            pca = PCA(n_components=2)
            xy = pca.fit_transform(transposed_input)
        elif 'dm' or 'diffusion' in self.dim_r:
            import pydiffmap as dm
            dfmap = dm.diffusion_map.DiffusionMap.from_sklearn(n_evecs=2,metric = 'correlation',
                                                            alpha = 0.5, epsilon = 1, k = 32)
            xy = dfmap.fit_transform(transposed_input)
        if self.verbose:
            print(">>>> Dim-reduction done")

        #%% Assign pixels
        if self.assign == 'refined':
            eq_xy = two_d_eq(xy)
            mapping = assign_features_to_pixels(eq_xy, nn, verbose=self.verbose)
        elif self.assign == 'lap':
            mapping = lap_scipy(xy, nn, verbose=self.verbose)

        try:
            InitCorr(dist_mat, mapping, nn)
        except ValueError:
            print("Refined fitting: Initial Corrs unavailable. ")
        #%%
        self.hw = nn
        self.feature_names_list = feature_names_list
        self.dist_mat = dist_mat
        self.mapping_obj_array = mapping
        # self._transform_mapping_dict()
        self.mapping_dict = self._transform_mapping_dict(self.mapping_obj_array.copy(), self.feature_names_list)
        self._fitted = True
        return self

    def save_mapping_for_hill_climbing(self, f_name='to_hill_climbing'):
        try:
            with open('./{}.pickle'.format(f_name),'wb') as file:
                pickle.dump([self.feature_names_list,
                    self.dist_mat, self.mapping_obj_array], file)
            print("File saved.")
        except IOError:
            print("IO Error. Not saved.")

    def load_from_hill_climbing(self, f_name='to_hill_climbing'):
        f_name = f_name.strip('.pickle')
        try:
            with open('{}.pickle'.format(f_name), 'rb') as file:
                feature_name_list, pos_mat, init_map = pickle.load(file)
            self._fitted = True
        except :
            print("Cannot read files from Hill climbing. ")
            return None
        self.feature_names_list = feature_name_list
        self.mapping_dict = pd.DataFrame(pos_mat, index=feature_name_list, columns=['x','y']).to_dict('index')
        self.mapping_obj_array = init_map
        return self

    def load_from_string_coords(self, string_f):
        if string_f.endswith('.csv'):
            tb = pd.read_csv(string_f, index_col=0)
        else:
            tb = pd.read_table(string_f, index_col=0)
        Nn = len(tb)
        nn = math.ceil(math.sqrt(Nn))
        self.feature_names_list = tb.index.tolist()
        xy = tb.iloc[:, :2].values
        eq_xy = two_d_eq(xy)
        mapping = assign_features_to_pixels(eq_xy, nn, verbose=self.verbose)
        self.mapping_obj_array = mapping
        self.mapping_dict = self._transform_mapping_dict(self.mapping_obj_array.copy(),
                                                         self.feature_names_list)
        self._fitted = True
        self.hw = nn
        return self

    def generate_image(self, data_df, output_folder='Images', img_format='npy',
                       dtype=int, normalize_feature=True, zscore=False, zscore_cutoff=5,
                       random_map=False, white_noise=False):
        assert self._fitted
        assert isinstance(data_df, pd.DataFrame)
        gene_expression = data_df.copy()
        gene_expression.index = gene_expression.index.map(str) # In cases that index are ints.
        feature_name_list = self.feature_names_list.copy()
        # Check if features_names in REFINED is same to the given DF
        print("Features not found in DF: ", [x for x in feature_name_list if x not in gene_expression.columns])
        gene_expression = gene_expression.reindex(feature_name_list, axis="columns")
        mapping_dict = self.mapping_dict.copy()

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

        if zscore:
            # temp = minmax_scale(scale(gene_expression), feature_range=(0,255), axis=0)
            temp = scale(gene_expression)
            temp = np.clip(temp, -zscore_cutoff, zscore_cutoff)
            # temp = minmax_scale(temp, feature_range=(0,255), axis=0)
            gene_expression = pd.DataFrame(temp,
                       index=gene_expression.index, columns=gene_expression.columns).fillna(0)
        elif normalize_feature:
            gene_expression = normalize_df(gene_expression, (0, 255)).fillna(0)\
        # Normalize [0, 255]
        gene_expression = ToolBox.normalize_int_between(gene_expression, 0, 255)

        # Make folder
        try:
            os.mkdir('./'+output_folder)
        except FileExistsError:
            print("Folder already exists.")
        finally:
            os.chdir('./'+output_folder)

        # Save images
        n_img = len(gene_expression)
        debug_i = 1
        for idx, each_cell_line in gene_expression.iterrows():
            # each_cell_line is a Series.
            if self.verbose:
                print("\rGenerating images: {} / {}".format(debug_i, str(n_img)), end='')
            if ('np' in img_format) and (dtype == float):
                Img = np.zeros(self.mapping_obj_array.shape)
                # or start with 125?
                Img = np.full(self.mapping_obj_array.shape, np.nan)
            else:
                # imwrite requires unit8
                Img = np.zeros(self.mapping_obj_array.shape).astype(np.uint8)
                # or 125?
                Img = np.full(self.mapping_obj_array.shape, np.nan).astype(np.uint8)

            cell_line_name = each_cell_line.name
            # Exceptions:
            if cell_line_name == 'PE/CAPJ15':
                cell_line_name = 'PE'
            # save
            # for each_feature in each_cell_line.index:
            for each_feature in self.feature_names_list:
                xx = mapping_dict[each_feature]['x']
                yy = mapping_dict[each_feature]['y']
                val = each_cell_line[each_feature]
                Img[xx,yy] = val # note: here is defined as this, x is the rows, y is the cols.

            # OR use mean instead of 0?
            np.nan_to_num(Img, copy=False, nan=np.nanmean(Img))

            # Drug imgs does not accept underscore
            cell_line_name = cell_line_name.replace('_','')
            if 'VNLG124' == cell_line_name:
                cell_line_name = 'VNLG_124'

            if 'np' in img_format:# and dtype is float: （why have this dtype check?
                np.save('MDS_'+ str(cell_line_name), Img)
            else:
                imsave('MDS_'+cell_line_name+'.'+img_format, Img)

            debug_i += 1
        print("\n>>> Image generated.")
        return None

    def plot_mapping(self):
        import matplotlib.pyplot as plt
        assert self._fitted
        hw = self.mapping_obj_array.shape[0]
        mapp = self.mapping_dict
        for txt in self.feature_names_list:
            yy = mapp[txt]['y'] + 0.5
            xx = hw - mapp[txt]['x'] - 0.5
            plt.text(yy, xx, txt,
                    horizontalalignment='center', verticalalignment='center',
                    wrap=True)
        plt.grid()
        plt.xlim(0, hw)
        plt.ylim(0, hw)
        return None

    def reverse_mapping(self, array):
        pass

def gen_mapping(args: RFDArgs):

    pass

def gen_images(args: GenImgArgs):
    pass

def pipeline(args: PipelineArgs):
    pass

if __name__ == "__main__":
    pass