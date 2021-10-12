"""
Assign 2d xy to grid pixels


"""
import os
import numpy as np
from sklearn import manifold
from sklearn.metrics import pairwise_distances
from itertools import product

#%% to 2-d representation (xy coordinates)
def two_d_norm(xy):
    # xy is N x 2 xy cordinates, returns normed-xy on [0,1]
    norm_xy = (xy - xy.min(axis = 0)) / (xy - xy.min(axis = 0)).max(axis = 0)
    return norm_xy

def two_d_eq(xy):
    # xy is N x 2 xy cordinates, returns eq-xy on [0,1]
    xx_rank = np.argsort(xy[:,0])
    yy_rank = np.argsort(xy[:,1])
    eq_xy = np.full(xy.shape,np.nan)
    for ii in range(xy.shape[0]):
        xx_idx = xx_rank[ii]
        yy_idx = yy_rank[ii]
        eq_xy[xx_idx,0] = ii * 1/len(xy)
        eq_xy[yy_idx,1] = ii * 1/len(xy)
    return eq_xy


#%% to pixels
def assign_features_to_pixels(xy, nn, verbose = False, output_dir='.'):
    # For each unassigned feature, find its nearest pixel, repeat until every ft is assigned
    # xy is the 2-d coordinates (normalized to [0,1]); nn is the image width. Img size = n x n
    # generate the result summary table, xy pixels; 3rd is nan for filling the idx
    Nn = xy.shape[0]
    xy = two_d_norm(xy)

    pixel_iter = product([x for x in range(nn)],repeat = 2)
    result_table = np.full((nn*nn,3),np.nan)
    ii = 0
    for each_pixel in pixel_iter:
        result_table[ii,:2] = np.array(each_pixel)
        ii+=1
    # Use numpy array for speed


    # xy = eq_xy
    centroids = result_table[:,:2] / nn + 0.5/nn
    pixel_avail = np.ones(nn*nn).astype(bool)
    feature_assigned = np.zeros(Nn).astype(bool)

    dist_xy_centroids = pairwise_distances(centroids,xy,metric='euclidean')

    while feature_assigned.sum()<Nn:
        # Init the pick-relationship table
        pick_xy_centroids = np.zeros(dist_xy_centroids.shape).astype(bool)

        for each_feature in range(Nn):
            # for each feature, find the nearest available pixel
            if feature_assigned[each_feature] == True:
                # if this feature is already assigned, skip to the next ft
                continue
            else:
                # if not assigned:
                for ii in range(nn*nn):
                    # find the nearest avail pixel
                    nearest_pixel_idx = np.argsort(dist_xy_centroids[:,each_feature])[ii]
                    if pixel_avail[nearest_pixel_idx] == True:
                        break
                    else:
                        continue
                pick_xy_centroids[nearest_pixel_idx,each_feature] = True

        for each_pixel in range(nn*nn):
            # Assign the feature No to pixels
            if pixel_avail[each_pixel] == False:
                continue
            else:
                # find all the "True" features. np.where returns a tuple size 1
                related_features = np.where(pick_xy_centroids[each_pixel,:] == 1)[0]
                if len(related_features) == 1:
                    # Assign it
                    result_table[each_pixel,2] = related_features[0]
                    pixel_avail[each_pixel] = False
                    feature_assigned[related_features[0]] = True
                elif len(related_features) > 1:
                    related_dists = dist_xy_centroids[each_pixel,related_features]
                    best_feature = related_features[np.argsort(related_dists)[0]] # Sort, and pick the nearest one among them
                    result_table[each_pixel,2] = best_feature
                    pixel_avail[each_pixel] = False
                    feature_assigned[best_feature] = True
        if verbose:
            print("\r>> Assign features to pixels: {} / {}".format(feature_assigned.sum(), Nn), end='')
    result_table = result_table.astype(int)

    if verbose:
        plt.figure(figsize=(8, 8))
        plt.scatter(xy[:, 0], xy[:, 1])
        for item in result_table:  # restult_table is a np array.
            xx, yy, ft = item
            if ft < -1:
                continue
            # xx, yy need to transfer to [0, 1]
            start = xy[ft]  # 第3列是xy的第几个点
            end = (xx/nn, yy/nn)  # xx, yy 是grid 坐标
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                    head_length=0.01, head_width=0.01)
        plt.title("REFINED assignment displacement")
        plt.savefig(os.path.join(output_dir, "assignment_displacement.png"))
        # plt.show()


    img = np.full((nn,nn),'NaN').astype(object)
    for each_pixel in range(nn*nn):
        xx = result_table[each_pixel,0]
        yy = result_table[each_pixel,1]
        ft = 'F' + str(result_table[each_pixel,2])
        img[xx,yy] = ft
    return img.astype(object)

#%%  Scipy method
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def lap_scipy(xy, nn, verbose=False, output_dir='.'):
    """
    xy is the 2D data.
    Grid size nn x nn.
    """
    assert(len(xy) <= nn * nn), "Num of data points must less than (or equal to) grid points."
    Nn = xy.shape[0]
    # normalize to [0, 1]
    xy -= xy.min(axis=0)
    xy /= xy.max(axis=0)
    xv, yv = np.meshgrid(np.linspace(0, 1, nn), np.linspace(0, 1, nn))
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    grid_ij = np.dstack(np.meshgrid(range(nn), range(nn))).reshape(-1, 2)  # same as the previous line

    cost = cdist(grid, xy, 'sqeuclidean')
    cost = cost * (10000000. / cost.max())  # is it nessesary? 
    row_assigns, col_assigns = linear_sum_assignment(np.copy(cost))
    grid_scipy = grid[row_assigns]
    scipy_data2d = xy[col_assigns]

    # row_assigns 那个点(i, j) 对应的是col_assign 同位置int指示的那个feature点 (2dxy 每一行是一个feature)

    if verbose:
        plt.figure(figsize=(8, 8))
        plt.scatter(xy[:, 0], xy[:, 1])
        for start, end in zip(scipy_data2d, grid_scipy):
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                    head_length=0.01, head_width=0.01)
        plt.title("LAP assignment displacement")
        plt.savefig(os.path.join(output_dir, "assignment_displacement.png"))
        # plt.show()


    img = np.full((nn,nn),'NaN').astype(object)
    img = np.full((nn,nn), -1).astype(object)
    for col, row in zip(col_assigns, row_assigns):
        xx = grid_ij[row][0]
        yy = grid_ij[row][1]
        ft = 'F' + str(col)
        img[xx, yy] = ft

    return img.astype(object)

#%%
def test_show_np_array_reshaping():
    xv, yv = np.meshgrid(np.linspace(1, 3, 3), np.linspace(1, 3, 3))
    grid = np.dstack((xv, yv))
    flat_grid = grid.reshape(-1, 2)

    print(grid)
    print("->")
    print(flat_grid)

def test_scipy_assignment():
    side = 20
    totalDataPoints = side * side - 10
    data3d = np.random.uniform(low=0.0, high=1.0, size=(totalDataPoints, 3))
    tsne = manifold.MDS(n_components=2)
    data2d = tsne.fit_transform(data3d)
    lap_scipy(data2d, side, True)

def test_rfd_assignment():
    side = 20
    totalDataPoints = side * side - 10
    data3d = np.random.uniform(low=0.0, high=1.0, size=(totalDataPoints, 3))
    tsne = manifold.MDS(n_components=2)
    data2d = tsne.fit_transform(data3d)
    assign_features_to_pixels(data2d, side, True)


if __name__ == "__main__":
    # test_rfd_assignment()
    test_scipy_assignment()
