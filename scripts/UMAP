#%%
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
#%%
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

#%%
data_d2 = pickle.load(open('G:/My Drive/all_ephys_d2.p','rb'))
data_d1 = pickle.load(open('G:/My Drive/all_ephys_d1.p','rb'))
data_sag = pickle.load(open('G:/My Drive/all_ephys_sag.p','rb'))
data_acsf_exc = pickle.load(open('G:/My Drive/all_ephys_exc_NC_acsf_imp.p','rb'))
data_acsf_inh = pickle.load(open('G:/My Drive/all_ephys_inh_NC_acsf_imp.p','rb'))

ind_feat = [0,3,4,6,9,17,20]

# perplexity_ = np.arange(5,100,10)
# fig, ax_inh = plt.subplots(1,len(perplexity_),figsize=[40,4])
# fig, ax_exc = plt.subplots(1,len(perplexity_),figsize=[40,4])



data_all_inh = np.vstack([  remove_nans_and_infs(np.squeeze(np.array(data_d2['inh'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_sag['inh'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf'])))[:,ind_feat]])

data_all_exc = np.vstack([  remove_nans_and_infs(np.squeeze(np.array(data_d2['exc'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['exc'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_sag['exc'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf'])))[:,ind_feat],
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf'])))[:,ind_feat]])

colors_inh = [np.repeat('red',len(remove_nans_and_infs(np.squeeze(np.array(data_d2['inh']))))),
np.repeat('green',len(remove_nans_and_infs(np.squeeze(np.array(data_sag['inh']))))),
np.repeat('blue',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh']))))),
np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))))),
np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))))),
np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf'])))))]

colors_exc = [np.repeat('red',len(remove_nans_and_infs(np.squeeze(np.array(data_d2['exc']))))),
np.repeat('green',len(remove_nans_and_infs(np.squeeze(np.array(data_sag['exc']))))),
np.repeat('blue',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc']))))),
np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf']))))),
np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf']))))),
np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf'])))))]
neighbours  = np.arange(5,50,5)
distances = np.arange(0.05,0.5,0.05)
for n in neighbours:
    for dist in distances:
        print(n, dist)
        plot_UMAP(data_acsf_inh['all'][:,ind_feat],data_acsf_exc['all'][:,ind_feat],'blue','blue',n,dist,['acsf'])
        # plot_UMAP(data_all_inh,data_all_exc,c_inh=np.hstack(colors_inh),c_exc=np.hstack(colors_exc),neighbours=n,distance=dist,legend=['sag','D1','D2','acsf'])
plot_UMAP(data_all_inh,data_all_exc,c_inh=np.hstack(colors_inh),c_exc=np.hstack(colors_exc))
plt.legend(['sag','D1','D2','acsf'])
plt.show() 
    
# %%
