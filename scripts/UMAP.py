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
import pickle
from PCA import * 
import umap.umap_ as umap

def plot_UMAP(data_inh,data_exc,c_exc,c_inh,neighbours,distance,condition_inh,condition_exc,figsize=None,random_state=0):
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
    fig = plt.figure(figsize=figsize)
    # ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax12d = fig.add_subplot(1,2,1)
    neighbours = neighbours
    dist = distance
    # clusterable_embedding3d = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
    #     n_components=3,random_state=42,).fit_transform(data_exc_umap)
    clusterable_embedding2d = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
        n_components=2,random_state=random_state,).fit_transform(data_exc_umap)
    df_2d = {'Dim1':clusterable_embedding2d[:, 0],
             'Dim2':clusterable_embedding2d[:, 1],
             'condition':condition_exc}
    sns.scatterplot(data=df_2d,x='Dim1',y='Dim2',hue='condition',  cmap='gist_rainbow',ax=ax12d)
    ax12d.set_title('UMAP excitatory')
    # plt.show()

    # fig = plt.figure(figsize=figsize)
    # ax2 = fig.add_subplot(1,2,1,projection='3d')
    ax22d = fig.add_subplot(1,2,2)
    
    # clusterable_embedding3d = umap.UMAP(n_neighbors=neighbours, min_dist=dist,
    #                                     n_components=3,random_state=42).fit_transform(data_inh_umap)

    clusterable_embedding2d = umap.UMAP(n_neighbors=neighbours, min_dist=dist,
                                        n_components=2,random_state=random_state).fit_transform(data_inh_umap)
    df_2d = {'Dim1':clusterable_embedding2d[:, 0],
             'Dim2':clusterable_embedding2d[:, 1],
             'condition':condition_inh}


    # sns.scatterplot(data=df_3d,x='Dim1',y='Dim2',hue='condition',  cmap='gist_rainbow',ax=ax22d)

    # ax2.scatter(clusterable_embedding3d[:, 0], clusterable_embedding3d[:, 1], clusterable_embedding3d[:, 2],c=c_inh,  cmap='gist_rainbow')
    # ax2.set_title('UMAP clusters for inhibitory neurons 3D')
    sns.scatterplot(data=df_2d,x='Dim1',y='Dim2',hue='condition',  cmap='gist_rainbow',ax=ax22d)
    ax22d.set_title('UMAP inhibitory')


    plt.show()

def plot_UMAP_clusters(data_inh,data_exc,neighbours,distance,condition_inh,condition_exc,k_exc,k_inh,random_state):
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
    fig = plt.figure(figsize=[10,4])
    ax12d = fig.add_subplot(1,2,1)
    neighbours = neighbours
    dist = distance

    clusterable_embedding2d_exc = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
        n_components=2,random_state=random_state,).fit_transform(data_exc_umap)

    kmeans_exc = KMeans(n_clusters=k_exc).fit(clusterable_embedding2d_exc)
    centroids_exc = kmeans_exc.cluster_centers_
    labels_exc = kmeans_exc.labels_.astype(float)

    df_2d_exc = {'Dim1':clusterable_embedding2d_exc[:, 0],
             'Dim2':clusterable_embedding2d_exc[:, 1],
             'condition':condition_exc,
             'labels':labels_exc}


    sns.scatterplot(data=df_2d_exc,x='Dim1',y='Dim2',hue='labels',  cmap='gist_rainbow',ax=ax12d)
    ax12d.set_title('UMAP clusters for excitatory neurons 2D')
    plt.show()

    fig = plt.figure(figsize=[10,4])
    # ax2 = fig.add_subplot(1,2,1,projection='3d')
    ax22d = fig.add_subplot(1,2,2)
    
    clusterable_embedding3d_inh = umap.UMAP(n_neighbors=neighbours, min_dist=dist,
                                        n_components=3,random_state=random_state).fit_transform(data_inh_umap)

    clusterable_embedding2d_inh = umap.UMAP(n_neighbors=neighbours, min_dist=dist,
                                        n_components=2,random_state=random_state).fit_transform(data_inh_umap)

    kmeans_inh = KMeans(n_clusters=k_inh,random_state=random_state).fit(clusterable_embedding2d_inh)
    centroids_inh = kmeans_inh.cluster_centers_
    labels_inh = kmeans_inh.labels_.astype(float)

    df_2d_inh = {'Dim1':clusterable_embedding2d_inh[:, 0],
                'Dim2':clusterable_embedding2d_inh[:, 1],
                'condition':condition_inh,
                'labels':labels_inh}


    # sns.scatterplot(data=df_3d,x='Dim1',y='Dim2',hue='condition',  cmap='gist_rainbow',ax=ax22d)

    # ax2.scatter(clusterable_embedding3d[:, 0], clusterable_embedding3d[:, 1], clusterable_embedding3d[:, 2],c=c_inh,  cmap='gist_rainbow')
    # ax2.set_title('UMAP clusters for inhibitory neurons 3D')
    sns.scatterplot(data=df_2d_inh,x='Dim1',y='Dim2',hue='labels',  cmap='gist_rainbow',ax=ax22d)
    ax22d.set_title('UMAP clusters for inhibitory neurons 2D')



    plt.show()
    return labels_exc, labels_inh

def plot_UMAP_combined(data_exc,data_inh,neighbours,distance,labels,random_state,figsize):

    data_scaler = StandardScaler()
    data_all = np.concatenate((data_inh,data_exc))
    data_all = data_scaler.fit_transform(data_all)
    data_all = normalize(data_all)
    fig = plt.figure(figsize=figsize)
    # ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax12d = fig.add_subplot(1,1,1)
    neighbours = neighbours
    dist = distance
    clusterable_embedding2d = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
        n_components=3,random_state=random_state,).fit_transform(data_all)
    df_2d = pd.DataFrame({'Dim1':clusterable_embedding2d[:, 0],
             'Dim2':clusterable_embedding2d[:, 1],
             'Dim3':clusterable_embedding2d[:, 2],
             'type':labels})
    sns.scatterplot(data=df_2d,x='Dim1',y='Dim2', hue='type', cmap='gist_rainbow',ax=ax12d,alpha=1.,markers=['x','.'])
    ax12d.set_title('UMAP excitatory and Inhibitory')
  
    plt.show()
