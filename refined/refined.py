"""


"""
import os
from platform import dist
import numpy as np
import pandas as pd
import math
import pickle
import json
import sklearn.manifold as mnf
from sklearn.decomposition import PCA
from imageio import imwrite as imsave
from scipy.stats import pearsonr
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale
from myToolbox import ToolBox
from myToolbox.Stat import normalize_df
from refined.ImageIsomap import ImageIsomap
from refined.assignment import two_d_norm, two_d_eq
from refined.assignment import assign_features_to_pixels, lap_scipy
from refined.args import RFDArgs, GenImgArgs, PipelineArgs
from refined.io import check_path_exists, float_to_int, read_df_list
import textwrap

#%%
class Refined(object):
    def __init__(self, dim_reduction='MDS', distance_metric='correlation',
                 assignment='lap', verbose=True, seed=None, **kwarg):
        """
        dim_reduction: 'mds', 'c-iso', ... etc.
        distance_metric: correlation or euclidean;
        assignment: "refined" for the old one (nearest first), 'lap' for scipy linear assignment problem;
        seed: MDS, LLE, tSNE have random states.

        """
        self.dist_m = distance_metric.lower()
        self.dim_r = dim_reduction.lower()
        self.verbose = verbose
        self.hw = None
        self.assign = assignment.lower()
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


    def fit(self, original_input: pd.DataFrame, key_param=None, output_dir='.'):
        """
        Calculate the initial correlations (distances) of features,
        Assign the original positions mapping.
        key_param is n_neighbors for isomap, etc.. or perplexity for tSNE.

        """
        assert isinstance(original_input, pd.DataFrame)
        feature_names_list = original_input.columns.tolist()
        nn = math.ceil(np.sqrt(len(feature_names_list))) # image dimension
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
            mapping = assign_features_to_pixels(eq_xy, nn, verbose=self.verbose, output_dir=output_dir)
        elif self.assign == 'lap':
            mapping = lap_scipy(xy, nn, verbose=self.verbose, output_dir=output_dir)

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

    def generate_image(self, data_df, output_folder='.', img_format='npy',
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
        img_dir = os.path.join(output_folder, "RFD_Images/")
        check_path_exists(output_folder)
        check_path_exists(img_dir)

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

            # An Exceptions:
            if cell_line_name == 'PE/CAPJ15':
                cell_line_name = 'PE'

            if "/" in cell_line_name:
                cell_line_name = cell_line_name.replace("/", "-")  # correct way but
            # save
            # for each_feature in each_cell_line.index:
            for each_feature in self.feature_names_list:
                xx = mapping_dict[each_feature]['x']
                yy = mapping_dict[each_feature]['y']
                val = each_cell_line[each_feature]
                Img[xx, yy] = val # note: here is defined as this, x is the rows, y is the cols.

            # OR use mean instead of 0?
            np.nan_to_num(Img, copy=False, nan=np.nanmean(Img))

            # Drug imgs does not accept underscore
            cell_line_name = cell_line_name.replace('_','')
            if 'VNLG124' == cell_line_name:
                cell_line_name = 'VNLG_124'

            if 'np' in img_format:# and dtype is float: （why have this dtype check?
                np.save(os.path.join(img_dir, 'RFD_'+ str(cell_line_name)), Img)
            else:
                imsave(os.path.join(img_dir,'RFD_'+cell_line_name+'.'+img_format), Img)

            debug_i += 1
        print("\n>>> Image generated.")
        return None

    def plot_mapping(self, output_dir=None):
        import matplotlib.pyplot as plt
        assert self._fitted
        hw = self.mapping_obj_array.shape[0]
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
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, "REFINED_mapping.png"))
        return None

    def reverse_mapping(self, array):
        pass

    def load_from_map_dict(self, fname):
        pass

#%%
def gen_mapping(args: RFDArgs):
    odir = args.output_dir
    check_path_exists(odir)

    rfd = Refined(
        dim_reduction=args.dim_reduction,
        distance_metric=args.distance_metric,
        assignment=args.assignment,
        verbose=args.verbose
        )

    data = read_df_list(args.df_path)
    key_param = float_to_int(args.key_param)
    rfd.fit(data, key_param=key_param, output_dir=odir)

    with open(os.path.join(odir, "REFINED_obj.pickle"), 'wb') as f:
        pickle.dump(rfd, f)
    with open(os.path.join(odir, "REFINED_mapping.json"), 'w') as f:
        json.dump(rfd.mapping_dict, f)
    return rfd

from pydoc import locate
def gen_images(args: GenImgArgs):
    with open(args.path, 'rb') as f:
        rfd = pickle.load(f)
    data = read_df_list(args.df_path)
    dtype = locate(args.dtype)
    rfd.generate_image(
        data,
        output_folder=args.output_dir,
        img_format=args.img_format,
        dtype=dtype,
        normalize_feature=args.normalize,
        zscore=args.zscore,
        zscore_cutoff=args.zscore_cutoff,
        random_map=False, white_noise=False)

def pipeline(args: PipelineArgs):
    pass

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