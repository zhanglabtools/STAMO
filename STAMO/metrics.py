r"""
Performance evaluation metrics
"""

from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
import sklearn.metrics
import sklearn.neighbors
from anndata import AnnData
from scipy.sparse.csgraph import connected_components

from .typehint import RandomState
from .utils import get_rs


def mean_average_precision(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Mean average precision

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    neighbor_frac
        Nearest neighbor fraction
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    map
        Mean average precision
    """
    k = max(round(y.shape[0] * neighbor_frac), 1)
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), **kwargs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    return np.apply_along_axis(_average_precision, 1, match).mean().item()


def _average_precision(match: np.ndarray) -> float:
    if np.any(match):
        cummean = np.cumsum(match) / (np.arange(match.size) + 1)
        return cummean[match].mean().item()
    return 0.0


def normalized_mutual_info(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Normalized mutual information with true clustering

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.normalized_mutual_info_score`

    Returns
    -------
    nmi
        Normalized mutual information

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    x = AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    nmi_list = []
    for res in (np.arange(20) + 1) / 10:
        sc.tl.leiden(x, resolution=res)
        leiden = x.obs["leiden"]
        nmi_list.append(sklearn.metrics.normalized_mutual_info_score(
            y, leiden, **kwargs
        ).item())
    return max(nmi_list)

def ARI(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Normalized mutual information with true clustering

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.normalized_mutual_info_score`

    Returns
    -------
    nmi
        Normalized mutual information

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    x = AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    ari_list = []
    for res in (np.arange(20) + 1) / 10:
        sc.tl.leiden(x, resolution=res)
        leiden = x.obs["leiden"]
        ari_list.append(sklearn.metrics.adjusted_rand_score(
            y, leiden, **kwargs
        ).item())
    return max(ari_list)

def avg_silhouette_width(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Cell type average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_score`

    Returns
    -------
    asw
        Cell type average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    return (sklearn.metrics.silhouette_score(x, y, **kwargs).item() + 1) / 2


def graph_connectivity(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> float:
    r"""
    Graph connectivity

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`scanpy.pp.neighbors`

    Returns
    -------
    conn
        Graph connectivity
    """
    x = AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X", **kwargs)
    conns = []
    for y_ in np.unique(y):
        x_ = x[y == y_]
        _, c = connected_components(
            x_.obsp['connectivities'],
            connection='strong'
        )
        counts = pd.value_counts(c)
        conns.append(counts.max() / counts.sum())
    return np.mean(conns).item()


def seurat_alignment_score(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01,
        n_repeats: int = 4, random_state: RandomState = None, **kwargs
) -> float:
    r"""
    Seurat alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = get_rs(random_state)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k + 1, **kwargs
        ).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
            subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.mean(repeat_scores).item()


def avg_silhouette_width_batch(
        x: np.ndarray, y: np.ndarray, ct: np.ndarray, **kwargs
) -> float:
    r"""
    Batch average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    ct
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_samples`

    Returns
    -------
    asw_batch
        Batch average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    s_per_ct = []
    for t in np.unique(ct):
        mask = ct == t
        try:
            s = sklearn.metrics.silhouette_samples(x[mask], y[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
    return np.mean(s_per_ct).item()


def neighbor_conservation(
        x: np.ndarray, y: np.ndarray, batch: np.ndarray,
        neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Neighbor conservation score

    Parameters
    ----------
    x
        Cooordinates after integration
    y
        Coordinates before integration
    b
        Batch
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    nn_cons
        Neighbor conservation score
    """
    nn_cons_per_batch = []
    for b in np.unique(batch):
        mask = batch == b
        x_, y_ = x[mask], y[mask]
        k = max(round(x.shape[0] * neighbor_frac), 1)
        nnx = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(x_.shape[0], k + 1), **kwargs
        ).fit(x_).kneighbors_graph(x_)
        nny = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(y_.shape[0], k + 1), **kwargs
        ).fit(y_).kneighbors_graph(y_)
        nnx.setdiag(0)  # Remove self
        nny.setdiag(0)  # Remove self
        n_intersection = nnx.multiply(nny).sum(axis=1).A1
        n_union = (nnx + nny).astype(bool).sum(axis=1).A1
        nn_cons_per_batch.append((n_intersection / n_union).mean())
    return np.mean(nn_cons_per_batch).item()


def foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return foscttm_x, foscttm_y


import esda
import libpysal
def moran_i(adata, coord_keys, obs_key, k=1):
    """
    Calculate Moran's I for a given spatially distributed variable.

    Args:
        adata (anndata.AnnData): Annotated data matrix.
        coord_keys (list): List of keys in adata.obsm containing spatial coordinates.
        obs_key (str): Key in adata.obs containing the variable of interest.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Moran's I value.
    """
    coordinates = adata.obsm[coord_keys]
    w_knn = libpysal.weights.KNN.from_array(
        coordinates,
        k=k,
    )
    values = adata.obs[obs_key].cat.codes
    moran = esda.Moran(values, w_knn)
    moran_i = moran.I
    return moran_i

## https://github.com/JinmiaoChenLab/SpaMosaic
# adata.obs['cluster'] = adata.obs['cluster'].astype('int')
# res = Morans(adata, cols=['cluster'])
# import squidpy as sq
# def Morans(ad, cols, coord_type='generic', knns=4, **kwargs):
#     col_data = []
#     for col in cols:
#         if pd.api.types.is_numeric_dtype(ad.obs[col]):
#             col_data.append(ad.obs[col].to_list())
#         else:
#             col_data.append(ad.obs[col].astype('category').cat.codes)
#
#     col_data = np.hstack(col_data).reshape(len(cols), -1).T
#     ad_holder = sc.AnnData(col_data, obsm={'spatial': ad.obsm['spatial']})
#     ad_holder.var_names = cols
#
#     sq.gr.spatial_neighbors(ad_holder, coord_type=coord_type, n_neighs=knns, **kwargs)
#     sq.gr.spatial_autocorr(
#         ad_holder,
#         mode="moran",
#         genes=cols,
#         n_perms=100,
#         n_jobs=1,
#     )
#     return ad_holder.uns["moranI"]


import gc
import copy
import math
import torch
import scipy
import pickle
import warnings
# import episcanpy as epi
import matplotlib as mpl
import matplotlib.pyplot as plt

import anndata as ad
import numpy as np
# import squidpy as sq
import pandas as pd
import logging
import scanpy as sc
from os.path import join
import scipy.io as sio
import scipy.sparse as sps
from sklearn.cluster import KMeans
# import gzip
from scipy.io import mmread
from pathlib import Path, PurePath
from sklearn.metrics import adjusted_rand_score, roc_auc_score, f1_score
from annoy import AnnoyIndex
import itertools
# from scib.metrics import lisi
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


def binarize(Xs, bin_thr=0):
    rs = []
    for X in Xs:
        X = copy.deepcopy(X.A) if sps.issparse(X) else copy.deepcopy(X)
        # X[X>bin_thr] = 1
        X = np.where(X > bin_thr, 1, 0)
        rs.append(X)
    return rs


def eval_AUC_all(gt_X, pr_X, bin_thr=1):
    gt_X = binarize([gt_X], bin_thr)[0].flatten()
    pr_X = pr_X.flatten()
    auroc = roc_auc_score(gt_X, pr_X)
    return auroc


def PCCs(gt_X, pr_X):
    pcc_cell = [np.corrcoef(gt_X[i, :], pr_X[i, :])[0, 1] for i in range(gt_X.shape[0])]
    pcc_feat = [np.corrcoef(gt_X[:, i], pr_X[:, i])[0, 1] for i in range(gt_X.shape[1])]
    return pcc_cell, pcc_feat


def CMD(pr_X, gt_X):
    zero_rows_indices1 = list(np.where(~pr_X.any(axis=1))[0])  # all-zero rows
    zero_rows_indices2 = list(np.where(~gt_X.any(axis=1))[0])
    zero_rows_indices = zero_rows_indices1 + zero_rows_indices2
    rm_p = len(zero_rows_indices) / pr_X.shape[0]
    if rm_p >= .05:
        print(f'Warning: two many rows {rm_p}% with all zeros')
    pr_array = pr_X[~np.isin(np.arange(pr_X.shape[0]), zero_rows_indices)].copy()
    gt_array = gt_X[~np.isin(np.arange(gt_X.shape[0]), zero_rows_indices)].copy()
    corr_pr = np.corrcoef(pr_array, dtype=np.float32)  # correlation matrix
    corr_gt = np.corrcoef(gt_array, dtype=np.float32)

    x = np.trace(corr_pr.dot(corr_gt))
    y = np.linalg.norm(corr_pr, 'fro') * np.linalg.norm(corr_gt, 'fro')
    cmd = 1 - x / (y + 1e-8)
    return cmd


def nn_annoy(ds1, ds2, norm=True, knn=20, metric='euclidean', n_trees=10):
    if norm:
        ds1 = normalize(ds1)
        ds2 = normalize(ds2)

    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    return ind


def knn_smoothing(ad, hvf_name=None, dim_red_key='X_lsi', knn=50):
    knn_ind = nn_annoy(ad.obsm[dim_red_key], ad.obsm[dim_red_key],
                       norm=True, knn=knn + 1, metric='manhattan', n_trees=10)[:, 1:]
    X = ad[:, hvf_name].X.A if sps.issparse(ad.X) else ad[:, hvf_name].X
    smthed_X = np.mean(X[knn_ind.ravel()].reshape(X.shape[0], knn, X.shape[1]), axis=1)
    return smthed_X


def Morans(ad, cols, coord_type='generic', **kwargs):
    col_data = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(ad.obs[col]):
            col_data.append(ad.obs[col].to_list())
        else:
            col_data.append(ad.obs[col].astype('category').cat.codes)

    col_data = np.hstack(col_data).reshape(len(cols), -1).T
    ad_holder = sc.AnnData(col_data, obsm={'spatial': ad.obsm['spatial']})
    ad_holder.var_names = cols

    sq.gr.spatial_neighbors(ad_holder, coord_type=coord_type, **kwargs)
    sq.gr.spatial_autocorr(
        ad_holder,
        mode="moran",
        genes=cols,
        n_perms=100,
        n_jobs=1,
    )
    return ad_holder.uns["moranI"]


def iLISI(adata, batch_key, use_rep):
    _lisi = lisi.ilisi_graph(
        adata,
        batch_key,
        'embed',
        use_rep=use_rep,
        k0=90,
        subsample=None,
        scale=True,
        n_cores=1,
        verbose=False,
    )
    return _lisi


def snn_scores(
        x, y, k=1
):
    '''
        return: matching score matrix
    '''

    # print(f'{k} neighbors to consider during matching')

    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    ky = k or min(round(0.01 * y.shape[0]), 1000)
    nny = NearestNeighbors(n_neighbors=ky).fit(y)
    x2y = nny.kneighbors_graph(x)
    y2y = nny.kneighbors_graph(y)

    kx = k or min(round(0.01 * x.shape[0]), 1000)
    nnx = NearestNeighbors(n_neighbors=kx).fit(x)
    y2x = nnx.kneighbors_graph(y)
    x2x = nnx.kneighbors_graph(x)

    x2y_intersection = x2y @ y2y.T
    y2x_intersection = y2x @ x2x.T
    jaccard = x2y_intersection + y2x_intersection.T
    jaccard.data = jaccard.data / (2 * kx + 2 * ky - jaccard.data)
    matching_matrix = jaccard.multiply(1 / jaccard.sum(axis=1)).tocsr()
    return matching_matrix


def MS(
        mod1, mod2, split_by='batch', k=1, use_rep='X'
):
    '''
        return: scipy.sparse.csr_matrix
    '''

    mod1_splits = set(mod1.obs[split_by])
    mod2_splits = set(mod2.obs[split_by])
    splits = mod1_splits | mod2_splits

    matching_matrices, mod1_obs_names, mod2_obs_names = [], [], []
    for split in splits:
        mod1_split = mod1[mod1.obs[split_by] == split]
        mod2_split = mod2[mod2.obs[split_by] == split]
        mod1_obs_names.append(mod1_split.obs_names)
        mod2_obs_names.append(mod2_split.obs_names)

        matching_matrices.append(
            snn_scores(mod1_split.X, mod2_split.X, k)
            if use_rep == 'X' else
            snn_scores(mod1_split.obsm[use_rep], mod2_split.obsm[use_rep], k)
        )

    mod1_obs_names = pd.Index(np.concatenate(mod1_obs_names))
    mod2_obs_names = pd.Index(np.concatenate(mod2_obs_names))
    combined_matrix = scipy.sparse.block_diag(matching_matrices, format="csr")
    score_matrix = combined_matrix[
                   mod1_obs_names.get_indexer(mod1.obs_names), :
                   ][
                   :, mod2_obs_names.get_indexer(mod2.obs_names)
                   ]

    score = (score_matrix.diagonal() / score_matrix.sum(axis=1).A1).mean()
    return score


def batch_gpu_pairdist(emb1, emb2, batch_size=1024):
    n_batch = math.ceil(emb2.shape[0] / batch_size)
    emb2_gpu = torch.FloatTensor(emb2).cuda()
    emb2_gpu = emb2_gpu / torch.linalg.norm(emb2_gpu, ord=2, dim=1, keepdim=True)

    st = 0
    dist = []
    for i in range(n_batch):
        bsz = min(batch_size, emb1.shape[0] - i * batch_size)
        emb1_batch_gpu = torch.FloatTensor(emb1[st:st + bsz]).cuda()
        emb1_batch_gpu /= torch.linalg.norm(emb1_batch_gpu, ord=2, dim=1, keepdim=True)

        _ = -emb1_batch_gpu @ emb2_gpu.T  # 0-similarity => dist
        dist.append(_.cpu().numpy())
        st = st + bsz

        del emb1_batch_gpu
        torch.cuda.empty_cache()
        gc.collect()

    del emb2_gpu
    torch.cuda.empty_cache()
    gc.collect()

    dist = np.vstack(dist)
    return dist


# def FOSCTTM(adata1, adata2, use_rep='X_emb'):
#     dist = batch_gpu_pairdist(adata1.obsm[use_rep], adata2.obsm[use_rep], batch_size=2048)
#     foscttm_x = (dist < dist.diagonal().reshape(-1, 1)).mean(axis=1)
#     foscttm_y = (dist < dist.diagonal()).mean(axis=0)
#     foscttm = (foscttm_x + foscttm_y).mean() / 2
#
#     return foscttm


def LabTransfer(ad1, ad2, use_rep, lab_key, knn=10):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        neigh1 = KNeighborsClassifier(n_neighbors=knn)
        neigh1.fit(ad1.obsm[use_rep], ad1.obs[lab_key].to_list())
        pr_lab2 = neigh1.predict(ad2.obsm[use_rep])
        f1_1 = f1_score(ad2.obs[lab_key].values, pr_lab2,
                        average='macro')

        neigh2 = KNeighborsClassifier(n_neighbors=knn)
        neigh2.fit(ad2.obsm[use_rep], ad2.obs[lab_key].to_list())
        pr_lab1 = neigh2.predict(ad1.obsm[use_rep])
        f1_2 = f1_score(ad1.obs[lab_key].values, pr_lab1,
                        average='macro')
        return (f1_1 + f1_2) / 2
