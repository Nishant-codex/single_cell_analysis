import pickle
from typing import Set
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import append 
from scipy.io import loadmat, savemat
import importlib.util
import matplotlib.pyplot as plt
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
from sklearn import manifold, datasets
from sklearn.decomposition import PCA,IncrementalPCA,SparsePCA
from sklearn.preprocessing import StandardScaler, normalize

def remove_nans_and_infs(data:list):
    """_summary_

    Args:
        data (list): _description_

    Returns:
        _type_: _description_
    """
    nan = np.isnan(data)
    nan = np.where(nan==True)
    nan_rem = np.delete(data, [nan[0]],axis=0)
    nan = np.isnan(nan_rem)
    nan = np.where(nan==True)
    data = nan_rem

    inf = np.isinf(data)
    inf = np.where(inf==True)
    inf_rem = np.delete(data, [inf[0]],axis=0)
    inf = np.isnan(inf_rem)
    inf = np.where(inf==True)
    data = inf_rem
    return data

def takeRandomSamples(data,mask_level=1.0):
    """_summary_

    Args:
        data (_type_): _description_
        mask_level (float, optional): _description_. Defaults to 1.0.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    if mask_level>1 or mask_level<0:
        raise ValueError('Value must a float between 0 and 1')
    data_ = data
    zero_vals = np.zeros_like(data_)
    rand_vals = np.random.rand(zero_vals.shape[0],zero_vals.shape[1])
    ind = np.where(rand_vals<mask_level )
    zero_vals[ind] = 1
    return zero_vals

def shuffle_prams(data,col,all=False):
    """_summary_

    Args:
        data (_type_): _description_
        col (_type_): _description_
        all (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    rng = np.random.default_rng()
    data_o = data
    col_shuffled = data[:,col]
    if all:
        col_shuffled = data
        rng.shuffle(col_shuffled)
        data_o=col_shuffled      
        return data_o
    rng.shuffle(col_shuffled)
    data_o[:,col]  = col_shuffled
    return data_o  

def plot_pca(data:dict):
    """_summary_

    Args:
        data (list): _description_
    """
    features = ['Vm_avg','dvdt_p','dvdt_n','resistance','thr','adaptation',
    'isi','peak','peak_adaptation','ap_width','hyp_value','fist_spike','up_down_ratio',
    'isi_adaptation','thr_adp_ind','psth','int_fr','fr','sub_thr','spk_fr_adp','imp']

    min_size = min(np.array(data['inh']).shape[0],np.array(data['exc']).shape[0])

    scalar_inh = StandardScaler()
    scalar_exc = StandardScaler()
    data_inh_pca = scalar_inh.fit_transform(remove_nans_and_infs(np.squeeze(data['inh'])))
    data_inh_pca = normalize(data_inh_pca) 
    print('herer')
    data_exc_pca = scalar_exc.fit_transform(remove_nans_and_infs(np.squeeze(data['exc'])))
    data_exc_pca = normalize(data_exc_pca) 

    pca_x = PCA(whiten=True,random_state=40)

    fig, ax = plt.subplots(1,3,figsize=[24,8])

    # Project the data in 2D

    reduced_data_inh = pca_x.fit_transform(data_inh_pca[:min_size,:])
    exp_var_inh = pca_x.explained_variance_ratio_
    loadings = pca_x.components_.T * np.sqrt(pca_x.explained_variance_)

    n_components = 3

    kmeans = KMeans(n_clusters=5).fit(reduced_data_inh)
    centroids_inh = kmeans.cluster_centers_
    label = kmeans.labels_.astype(float)

    ax[0].scatter(reduced_data_inh[:,0], reduced_data_inh[:,1], c=label, s=50, alpha=0.5,marker = 'o')
    ax[0].scatter(centroids_inh[:, 0], centroids_inh[:, 1],c='black', s=50,marker = 'x')
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[0].set_title('Inhibitory')

    for i, feature in enumerate(features):
    # if (abs(loadings[i, 0])+abs(loadings[i, 1]))>0.5:
        ax[1].plot([0,loadings[i, 0]],[0,loadings[i, 1]])
        ax[1].annotate(feature, xy = [loadings[i, 0], loadings[i, 1]])
    ax[2].scatter(np.arange(len(exp_var_inh)),exp_var_inh)

    plt.show()

    fig, ax = plt.subplots(1,3,figsize=[24,8])

    pca_x_exc = PCA(whiten=True,random_state=40)
    # Project the data in 2D
    reduced_data_exc = pca_x_exc.fit_transform(data_exc_pca[:min_size,:])
    exp_var_exc = pca_x_exc.explained_variance_ratio_
    print(sum(exp_var_exc[:3]))
    loadings = pca_x_exc.components_.T * np.sqrt(pca_x_exc.explained_variance_)
    n_components = 2

    kmeans = KMeans(n_clusters=5).fit(reduced_data_exc)
    centroids_exc = kmeans.cluster_centers_
    label = kmeans.labels_.astype(float)

    ax[0].Projection ='3d'                   
    ax[0].scatter(reduced_data_exc[:,0], reduced_data_exc[:,1], c=label, s=50, alpha=0.5,marker = 'o')
    ax[0].scatter(centroids_exc[:, 0], centroids_exc[:, 1],c='black', s=50,marker = 'x')
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[0].set_title('Excitatory')

    for i, feature in enumerate(features):
    # if (abs(loadings[i, 0])+abs(loadings[i, 1]))>0.4:
        ax[1].plot([0,loadings[i, 0]],[0,loadings[i, 1]])
        ax[1].annotate(feature, xy = [loadings[i, 0], loadings[i, 1]])


    ax[2].scatter(np.arange(len(exp_var_exc)),exp_var_exc)
    plt.show()

def plot_pca_with_loadings(data:dict,features:list):
    """_summary_

    Args:
        data (dict): _description_
        features (list): _description_
    """
    features = features
    scalar = StandardScaler()

    data = scalar.fit_transform(data)
    data = normalize(data) 

    pca_x = PCA(whiten=True,random_state=40)
    # Project the data in 2D
    reduced_data = pca_x.fit_transform(data)
    exp_var = pca_x.explained_variance_ratio_
    loadings = pca_x.components_.T * np.sqrt(pca_x.explained_variance_)
    n_components = 2


    fig = plt.figure(figsize=[30,10])

    ax3d = fig.add_subplot(1,3,1,projection='3d',)
    axloadings = fig.add_subplot(1,3,2)
    ax2d = fig.add_subplot(1,3,3)

    ax2d.scatter(reduced_data[:,0], reduced_data[:,1],s=50, alpha=0.5,marker = 'o')
    ax2d.set_xlabel('PC1')
    ax2d.set_ylabel('PC2')
    ax2d.set_title('Inhibitory')


    for i, feature in enumerate(features):
        # if (abs(loadings[i, 0])+abs(loadings[i, 1]))>0.5:
        axloadings.plot([0,loadings[i, 0]],[0,loadings[i, 1]])
        axloadings.annotate(feature, xy = [loadings[i, 0], loadings[i, 1]])


    ax3d.scatter(reduced_data[:,0], reduced_data[:,1],reduced_data[:,2], s=50,  alpha=0.5,marker = 'o')
    ax3d.set_xlabel('PC1')
    ax3d.set_ylabel('PC2')
    ax3d.set_zlabel('PC3')
    ax3d.set_title('Inhibitory')
    plt.show()
    plt.scatter(np.arange(len(exp_var)),exp_var)
    plt.xlabel('PC components')
    plt.ylabel('Fraction Explained variance')
    plt.show()

def plot_PCA_oneoff(data_inh,data_exc):

    #@title plot PCA one off 
    scalar_inh = StandardScaler()
    scalar_inh.fit(data_inh['all'])
    scalar_exc = StandardScaler()
    scalar_exc.fit(data_exc['all'])
    data_inh_pca = scalar_inh.transform(data_inh['all'])
    data_exc_pca = scalar_exc.transform(data_exc['all'])
    size_inh = data_inh['all'].shape
    size_exc = data_exc['all'].shape
    min_size = min(size_inh[0],size_exc[0])

    for m in range(21):
        data_inh_pca = np.array(data_inh_pca[:min_size,:])
        data_exc_pca = np.array(data_exc_pca[:min_size,:])

    pca_x = PCA(n_components=10,whiten=True)

    fig = plt.figure(figsize=[14,7])

    ax = fig.add_subplot(1, 2, 1, ) #projection='3d'

    # Project the data in 2D
    reduced_data_inh = pca_x.fit_transform(data_inh_pca)
    n_components = 2

    kmeans = KMeans(n_clusters=5).fit(reduced_data_inh)
    centroids_inh = kmeans.cluster_centers_
    label = kmeans.labels_.astype(float)
    labels  = [] 

    for i in  kmeans.labels_.astype(float):
        if i ==0:
            labels.append('r')
        if i ==1:
            labels.append('b')
        if i ==2:
            labels.append('green')
        if i ==3:
            labels.append('cyan')
        if i ==4:
            labels.append('purple')    

    ax.scatter(reduced_data_inh[:,0], reduced_data_inh[:,1], c='red', s=50, alpha=0.5,marker = 'o') #,reduced_data_inh[:,2]
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Inhibitory w/ '+features[m])
    
    ## Excitatory plot 

    pca_x = PCA(whiten=True)
    # Project the data in 2D
    reduced_data_exc = pca_x.fit_transform(data_exc_pca)
    n_components = 2
    ax = fig.add_subplot(1, 2, 2, ) #projection='3d'
    kmeans = KMeans(n_clusters=5).fit(reduced_data_exc)
    centroids_exc = kmeans.cluster_centers_
    label = kmeans.labels_.astype(float)
    labels  = [] 
    for i in  kmeans.labels_.astype(float):
        if i ==0:
            labels.append('r')
        if i ==1:
            labels.append('b')
        if i ==2:
            labels.append('green')
        if i ==3:
            labels.append('cyan')
        if i ==4:
            labels.append('purple')                
    ax.scatter(reduced_data_exc[:,0], reduced_data_exc[:,1], c='blue', s=50, alpha=0.5,marker = 'o') #,reduced_data_exc[:,2]
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Excitatory w/ '+features[m])
    plt.show()
