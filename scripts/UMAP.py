import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn import manifold
from sklearn.cluster import KMeans
import os
import pickle
from typing import Set
import numpy as np 
from numpy.lib.function_base import append 
from scipy.io import loadmat, savemat
import importlib.util
from scipy.sparse import data 
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from plotnine.data import mtcars
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA,IncrementalPCA,SparsePCA
from sklearn.preprocessing import StandardScaler, normalize
from PCA import shuffle_prams
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from analyze_single_cell import collect_drug_and_acsf
from ephys_set import return_all_ephys_dict
# import umap
import pickle
from PCA import * 
# umap.UMAP()
import umap.umap_ as umap

def plot_UMAP(data_inh,data_exc,c_exc,c_inh,neighbours,distance,legend):
    """plots UMAP for excitatory and inhibitory cells 

    Args:
        data_inh (ndarray): cell X feature matrix for inhibitory cells
        data_exc (ndarray): cell X feature matrix for excitatory cells
        c_exc (array): _description_
        c_inh (array): _description_
        neighbours (int): number of neares neighbours
        distance (float): minimum distance between points
    """
    data_exc_umap_scaler = StandardScaler()
    data_exc_umap = data_exc_umap_scaler.fit_transform(data_exc)
    data_exc_umap = normalize(data_exc_umap)
    data_inh_umap = data_exc_umap_scaler.fit_transform(data_inh)
    data_inh_umap = normalize(data_inh_umap)
    # min_size = min(data_exc_umap.shape[0],data_inh_umap.shape[0])
    # kmeans_labels_inh = cluster.KMeans(n_clusters=5).fit_predict(data_inh_umap[:min_size,])
    # kmeans_labels_exc = cluster.KMeans(n_clusters=5).fit_predict(data_exc_umap[:min_size,])
    fig = plt.figure(figsize=[18,8])
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax12d = fig.add_subplot(1,2,2)
    neighbours = neighbours
    dist = distance
    clusterable_embedding3d = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
        n_components=3,random_state=42,).fit_transform(data_exc_umap)
    clusterable_embedding2d = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
        n_components=2,random_state=42,).fit_transform(data_exc_umap)

    ax1.scatter(clusterable_embedding3d[:, 0], clusterable_embedding3d[:, 1], clusterable_embedding3d[:, 2], c= c_exc, cmap='gist_rainbow')
    ax1.set_title('UMAP clusters for excitatory neurons')
    ax12d.scatter(clusterable_embedding2d[:, 0], clusterable_embedding2d[:, 1], c= c_exc, cmap='gist_rainbow')
    ax12d.set_title('UMAP clusters for excitatory neurons 2D')
    plt.legend(legend)
    plt.show()

    fig = plt.figure(figsize=[18,8])
    ax2 = fig.add_subplot(1,2,1,projection='3d')
    ax22d = fig.add_subplot(1,2,2)
    
    clusterable_embedding3d = umap.UMAP(n_neighbors=neighbours, min_dist=dist,
                                        n_components=3,random_state=42).fit_transform(data_inh_umap)

    clusterable_embedding2d = umap.UMAP(n_neighbors=neighbours, min_dist=dist,
                                        n_components=2,random_state=42).fit_transform(data_inh_umap)

    ax2.scatter(clusterable_embedding3d[:, 0], clusterable_embedding3d[:, 1], clusterable_embedding3d[:, 2],c=c_inh,  cmap='gist_rainbow')
    ax2.set_title('UMAP clusters for inhibitory neurons 3D')
    ax22d.scatter(clusterable_embedding2d[:, 0], clusterable_embedding2d[:, 1],c= c_inh,  cmap='gist_rainbow')
    ax22d.set_title('UMAP clusters for inhibitory neurons 2D')
    plt.legend(legend)

    plt.show()
