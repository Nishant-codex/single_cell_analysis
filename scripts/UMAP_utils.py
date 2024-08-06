
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
# from analyze_single_cell import collect_drug_and_acsf
# from ephys_set import return_all_ephys_dict
from sknetwork.clustering import Louvain,get_modularity

import pickle
from PCA import * 
import umap.umap_ as umap

def plot_UMAP(data_inh,data_exc,c_exc,c_inh,neighbours,distance,condition_inh,condition_exc,figsize=None,random_state=0,save=False):
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

    if save:
        plt.savefig('C:/Users/Nishant Joshi/Documents/DNM/umap_20.png')
    else:
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


def plot_UMAP_clusters_single(data_inh, neighbours, distance, res_louvain, random_state,size =0.2,annotate=False,norm=True, c_list=None ,savepath=None,save=False):
    """plots UMAP for excitatory and inhibitory cells 

    Args:
        data_inh (ndarray): cell X feature matrix for inhibitory cells
        data_exc (ndarray): cell X feature matrix for excitatory cells
        c_exc (array): _description_
        c_inh (array): _description_
        neighbours (int): number of neares neighbours
        distance (float): minimum distance between points
    """
    data_umap_scaler = StandardScaler()
    data_umap = data_umap_scaler.fit_transform(data_inh)
    if norm:
        data_umap = normalize(data_umap)
    
    neighbours = neighbours
    dist = distance
    reducer = umap.UMAP(n_neighbors=neighbours,min_dist=dist,random_state=random_state)
    mapper = reducer.fit(data_umap)
    
    fig = plt.figure(figsize=[8,8])
    ax12d = fig.add_subplot(1,1,1)
    
    louvain = Louvain(resolution=res_louvain,random_state=random_state)
    adjacency = mapper.graph_
    labels_exc = louvain.fit_predict(adjacency)

    print(len(set(labels_exc)))
    clusterable_embedding2d = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
        n_components=2,random_state=random_state,).fit_transform(data_umap)



    df_2d_exc = {'UMAP1':clusterable_embedding2d[:, 0],
             'UMAP2':clusterable_embedding2d[:, 1],
             'class':labels_exc}

    ax12d.set_xticks([])
    ax12d.set_yticks([])

    sns.scatterplot(data=df_2d_exc,x='UMAP1',y='UMAP2',hue='class',palette=c_list[:len(set(labels_exc))],ax=ax12d,s=size)
    if annotate:
        for i in range(len(clusterable_embedding2d)):

            plt.annotate(str(i),(clusterable_embedding2d[i,0]+0.05,clusterable_embedding2d[i,1]+0.05))

        # sns.scatterplot(data=df_2d_exc,x='UMAP1',y='UMAP2',hue='class',ax=ax12d)

        # ax12d.set_title('UMAP clusters for excitatory neurons 2D')
        ax12d.legend()
    if save:
        plt.tight_layout=True
        plt.savefig(savepath,dpi=200)
    
    plt.show()

    return labels_exc,mapper,reducer




def plot_UMAP_values(data_inh, values,neighbours=20, distance=0.1, random_state=42,annotate=False, c_list=None ,savepath=None,save=False):
    """plots UMAP for excitatory and inhibitory cells

    Args:
        data_inh (ndarray): cell X feature matrix for inhibitory cells
        data_exc (ndarray): cell X feature matrix for excitatory cells
        c_exc (array): _description_
        c_inh (array): _description_
        neighbours (int): number of neares neighbours
        distance (float): minimum distance between points
    """

    data_umap_scaler = StandardScaler()
    data_umap = data_umap_scaler.fit_transform(data_inh)

    neighbours = neighbours
    dist = distance
    reducer = umap.UMAP(n_neighbors=neighbours,min_dist=dist,random_state=random_state)
    mapper = reducer.fit(data_umap)

    fig = plt.figure(figsize=[4,4])
    ax12d = fig.add_subplot(1,1,1)

    clusterable_embedding2d = umap.UMAP(n_neighbors=neighbours,min_dist=dist,
        n_components=2,random_state=random_state,).fit_transform(data_umap)



    df_2d_exc = {'UMAP1':clusterable_embedding2d[:, 0],
             'UMAP2':clusterable_embedding2d[:, 1]}

    ax12d.set_xticks([])
    ax12d.set_yticks([])

    ax_ = ax12d.scatter(data=df_2d_exc,x='UMAP1',y='UMAP2',c=values)
    plt.colorbar(ax_)
    if annotate:
        for i in range(len(clusterable_embedding2d)):

            plt.annotate(str(i),(clusterable_embedding2d[i,0]+0.05,clusterable_embedding2d[i,1]+0.05))

        # sns.scatterplot(data=df_2d_exc,x='UMAP1',y='UMAP2',hue='class',ax=ax12d)

        # ax12d.set_title('UMAP clusters for excitatory neurons 2D')
        ax12d.legend()
    if save:
        plt.savefig(savepath,dpi=200)

    plt.show()



def return_confusion_matrix(df1,df2,label1_name,label2_name,shuffle = False):
    np.random.seed(42)
    if shuffle:
        fig,[ax1,ax2] = plt.subplots(1,2,figsize = [12,5])
        df = pd.DataFrame(columns=['label1','label2'])
        df['exp_name1'] = df1.exp_name
        df['exp_name2'] = df2.exp_name
        
        label1 = list(df1[label1_name])
        np.random.shuffle(label1)
        label2 = list(df2[label2_name])
        np.random.shuffle(label2)

        df['label1_sh'] = label1
        df['label2_sh'] = label2

        df['label1'] = np.array(df1[label1_name]) 
        df['label2'] = np.array(df2[label2_name])

        mat_orig = np.zeros((len(set(df1[label1_name])),len(set(df2[label2_name]))))

        for i in np.unique(df.label1):
            data_ = np.unique(df[df.label1==i]['label2'],return_counts=True)
            mat_orig[i,data_[0]] =(data_[1]/np.sum(data_[1]))*100 

        mat_sh = np.zeros((len(set(df['label1_sh'])),len(set(df['label2_sh']))))

        for i in np.unique(df.label1_sh):
            data_ = np.unique(df[df.label1_sh==i]['label2_sh'],return_counts=True)
            mat_sh[i,data_[0]] =(data_[1]/np.sum(data_[1]))*100 


        sns.heatmap(mat_orig,cmap='BrBG_r',annot=True,ax=ax1,vmin=0,vmax=100) 
        sns.heatmap(mat_sh,cmap='BrBG_r',annot=True,ax=ax2,vmin=0,vmax=100) 

        plt.show()
    else:
        df = pd.DataFrame(columns=['label1','label2'])
        df['exp_name1'] = df1.exp_name
        df['exp_name2'] = df2.exp_name
        df['label1'] = np.array(df1[label1_name])
        df['label2'] = np.array(df2[label2_name])

        mat = np.zeros((len(set(df1[label1_name])),len(set(df2[label2_name]))))

        for i in np.unique(df.label1):
            data_ = np.unique(df[df.label1==i]['label2'],return_counts=True)
            mat[i,data_[0]] =(data_[1]/np.sum(data_[1]))*100 

        sns.heatmap(mat,cmap='BrBG_r',annot=True,vmin=0,vmax=100) 

        plt.show()