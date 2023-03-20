#%%
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
from PCA import *
#%%
def plot_tsne(data_inh:dict,data_exc:dict):
    """_summary_

    Args:
        data_inh (dict): _description_
        data_exc (dict): _description_
    """
    scalar_inh_tsne = StandardScaler()
    scalar_exc_tsne = StandardScaler()
    min_size = min(data_inh.shape[0],data_exc.shape[0])


    data_inh_tsne = scalar_inh_tsne.fit_transform(data_inh)
    # data_inh_tsne = normalize(data_inh_tsne)
    # data_inh_tsne = data_inh


    data_exc_tsne = scalar_exc_tsne.fit_transform(data_exc)
    # data_exc_tsne = normalize(data_exc_tsne)
    # data_exc_tsne = data_exc
    data_inh_tsne = data_inh_tsne[:min_size,] 

    n_components = 2
    perplexity_ = np.arange(5,100,10)
    fig, ax = plt.subplots(1,len(perplexity_),figsize=[40,4])

    for p,j in enumerate(perplexity_):
        tsne = manifold.TSNE(
            n_components=n_components,
            init="pca",
            random_state=0,
            perplexity=j,
            learning_rate="auto",
            n_iter=1000,)

        reduced_data = tsne.fit_transform(np.array(data_inh_tsne))
        ax[p].scatter(reduced_data[:,0], reduced_data[:,1], c='blue', s=50, alpha=0.5,marker = 'o')
        ax[p].set_title('Inhibitory p='+str(j))
        ax[p].get_xaxis().set_visible(False)
        ax[p].get_yaxis().set_visible(False)    
    plt.show()

    data_exc_tsne = data_exc_tsne[:min_size,] 

    n_components = 2
    perplexity_ = np.arange(5,100,10)
    fig, ax = plt.subplots(1,len(perplexity_),figsize=[40,4])

    for p,j in enumerate(perplexity_):
        tsne = manifold.TSNE(
            n_components=n_components,
            init="pca",
            random_state=0,
            perplexity=j,
            learning_rate="auto",
            n_iter=400,)

        reduced_data = tsne.fit_transform(np.array(data_exc_tsne))
            
        ax[p].scatter(reduced_data[:,0], reduced_data[:,1], c='blue', s=50, alpha=0.5,marker = 'o')
        ax[p].set_title('Excitatory p='+str(j))
        ax[p].get_xaxis().set_visible(False)
        ax[p].get_yaxis().set_visible(False)
    plt.show()  

def plot_tsne_with_conditions(data_inh:list,data_exc:list,ax_inh,ax_exc,c='blue'):
    """_summary_

    Args:
        data_inh (dict): _description_
        data_exc (dict): _description_
    """
    scalar_inh_tsne = StandardScaler()
    scalar_exc_tsne = StandardScaler()
    min_size = min(data_inh.shape[0],data_exc.shape[0])


    data_inh_tsne = scalar_inh_tsne.fit_transform(data_inh)
    data_exc_tsne = scalar_exc_tsne.fit_transform(data_exc)
    data_inh_tsne = data_inh_tsne[:min_size,] 

    n_components = 2
    perplexity_ = np.arange(5,100,10)
    # fig, ax = plt.subplots(1,len(perplexity_),figsize=[40,4])
    ax = ax_inh
    for p,j in enumerate(perplexity_):
        tsne = manifold.TSNE(
            n_components=n_components,
            init="pca",
            random_state=0,
            perplexity=j,
            learning_rate="auto",
            n_iter=1000,)

        reduced_data = tsne.fit_transform(np.array(data_inh_tsne))
        ax[p].scatter(reduced_data[:,0], reduced_data[:,1], c=c, s=50, alpha=0.5,marker = 'o')
        ax[p].set_title('Inhibitory p='+str(j))
        ax[p].get_xaxis().set_visible(False)
        ax[p].get_yaxis().set_visible(False)    
    # plt.show()

    data_exc_tsne = data_exc_tsne[:min_size,] 

    n_components = 2
    perplexity_ = np.arange(5,100,10)
    # fig, ax = plt.subplots(1,len(perplexity_),figsize=[40,4])
    ax = ax_exc
    for p,j in enumerate(perplexity_):
        tsne = manifold.TSNE(
            n_components=n_components,
            init="pca",
            random_state=0,
            perplexity=j,
            learning_rate="auto",
            n_iter=400,)

        reduced_data = tsne.fit_transform(np.array(data_exc_tsne))
            
        ax[p].scatter(reduced_data[:,0], reduced_data[:,1], c=c, s=50, alpha=0.5,marker = 'o')
        ax[p].set_title('Excitatory p='+str(j))
        ax[p].get_xaxis().set_visible(False)
        ax[p].get_yaxis().set_visible(False)
    # plt.show()  

def plot_tsne_oneoff(data_inh:dict,data_exc:dict):
    """_summary_

    Args:
        data_inh (dict): _description_
        data_exc (dict): _description_

    Returns:
        None: _description_
    """

    data_inh_tsne_scaler = StandardScaler() 
    data_inh_tsne = data_inh_tsne_scaler.fit_transform(data_inh['all'])
    min_size = min(data_inh['all'].shape[0],data_exc['all'].shape[0])

    def normalize(data):
        return (data-np.mean(data))/np.std(data)
    for m in range(21):
        data_inh_ = data_inh_tsne[:min_size,] 

        #shuffle 
        data_inh_ = shuffle_prams(data_inh_tsne[:min_size,] ,m)

        # Project the data in 2D
        n_components = 2
        perplexity_ = np.arange(5,100,10)
        fig, ax = plt.subplots(1,len(perplexity_),figsize=[32,4])

        for p,j in enumerate(perplexity_):
            tsne = manifold.TSNE(
                n_components=n_components,
                init="pca",
                random_state=0,
                perplexity=j,
                learning_rate="auto",
                n_iter=1000,)

            reduced_data = tsne.fit_transform(np.array(data_inh_))


            kmeans = KMeans(n_clusters=5).fit(reduced_data)
            centroids = kmeans.cluster_centers_
            label = kmeans.labels_.astype(float)
            xs = np.repeat('x',281)
            os = np.repeat('o',418-281)
            mark = np.concatenate((xs,os))
         
            ax[p].scatter(reduced_data[:,0], reduced_data[:,1], c='blue', s=50, alpha=0.5,marker = 'o')
            ax[p].set_title('Inhibitory p='+str(j))
            ax[p].get_xaxis().set_visible(False)
            ax[p].get_yaxis().set_visible(False)   

        plt.show()    

# acsf,drug = collect_drug_and_acsf(path_files ='C:/Users/Nishant Joshi/Google Drive/Analyzed/' ,condition= ['sag'])        
#%%

# data_d2 = pickle.load(open('G:/My Drive/all_ephys_d2.p','rb'))
# data_d1 = pickle.load(open('G:/My Drive/all_ephys_d1.p','rb'))
# data_sag = pickle.load(open('G:/My Drive/all_ephys_sag.p','rb'))

# data_acsf_exc = pickle.load(open('G:/My Drive/all_ephys_exc_NC_acsf_imp.p','rb'))
# data_acsf_inh = pickle.load(open('G:/My Drive/all_ephys_inh_NC_acsf_imp.p','rb'))
# ind_feat = [0,3,4,6,9,17,20]
# # plot_tsne(remove_nans_and_infs(np.squeeze(np.array(data_acsf_exc['all'][:,ind_feat]))),
# #           remove_nans_and_infs(np.squeeze(np.array(data_acsf_inh['all'][:,ind_feat]))))  
# perplexity_ = np.arange(5,100,10)
# fig, ax_inh = plt.subplots(1,len(perplexity_),figsize=[40,4])
# fig, ax_exc = plt.subplots(1,len(perplexity_),figsize=[40,4])

data_all_inh = np.vstack([remove_nans_and_infs(np.squeeze(np.array(data_d2['inh']))),
                            remove_nans_and_infs(np.squeeze(np.array(data_sag['inh']))),
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh']))),
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))),
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))),
                            remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf'])))])
# data_all_exc = np.vstack([remove_nans_and_infs(np.squeeze(np.array(data_d2['exc']))),
# remove_nans_and_infs(np.squeeze(np.array(data_sag['exc']))),
# remove_nans_and_infs(np.squeeze(np.array(data_d1['exc']))),
# remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf']))),
# remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf']))),
# remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf'])))])

# colors = [np.repeat('red',len(remove_nans_and_infs(np.squeeze(np.array(data_d2['inh']))))),
# np.repeat('green',len(remove_nans_and_infs(np.squeeze(np.array(data_sag['inh']))))),
# np.repeat('blue',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh']))))),
# np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))))),
# np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))))),
# np.repeat('orange',len(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf'])))))]

# plot_tsne_with_conditions(data_all_inh,data_all_exc,ax_inh,ax_exc,c=np.hstack(colors))

# plt.legend(['sag','D1','D2','acsf'])
# plt.show()

# %%
# data_d2 = pickle.load(open('G:/My Drive/all_ephys_d2.p','rb'))
# data_d1 = pickle.load(open('G:/My Drive/all_ephys_d1.p','rb'))
# data_sag = pickle.load(open('G:/My Drive/all_ephys_sag.p','rb'))
# data_acsf_exc = pickle.load(open('G:/My Drive/all_ephys_exc_NC_acsf_imp.p','rb'))
# data_acsf_inh = pickle.load(open('G:/My Drive/all_ephys_inh_NC_acsf_imp.p','rb'))
# ind_feat = [0,3,4,6,9,17,20]
# plot_tsne(remove_nans_and_infs(np.squeeze(np.array(data_acsf_exc['all']))),
#           remove_nans_and_infs(np.squeeze(np.array(data_acsf_inh['all']))))  
# perplexity_ = np.arange(5,100,10)
# fig, ax_inh = plt.subplots(1,len(perplexity_),figsize=[40,4])
# fig, ax_exc = plt.subplots(1,len(perplexity_),figsize=[40,4])


# plt.legend(['sag','D1','D2'])
# plt.show()
# %%
