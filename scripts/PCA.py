#%%
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
#%%
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

def plot_pca(data_inh,data_exc,plot_loadings=False,feature_sub=None ):
    """_summary_

    Args:
        data (list): _description_
    """
    features = ['Vm_avg','dvdt_p','dvdt_n','resistance','thr','adaptation',
    'isi','peak','peak_adaptation','ap_width','hyp_value','fist_spike','up_down_ratio',
    'isi_adaptation','thr_adp_ind','psth','int_fr','fr','sub_thr','spk_fr_adp','imp']
    if feature_sub !=None:
        features = np.array(features)[feature_sub]

    print(np.array(data_inh).shape)
    min_size = min(np.array(data_inh).shape[0],np.array(data_exc).shape[0])

    scalar_inh = StandardScaler()
    scalar_exc = StandardScaler()
    data_inh_pca = scalar_inh.fit_transform(remove_nans_and_infs(np.squeeze(data_inh)))
    data_inh_pca = normalize(data_inh_pca) 
    data_exc_pca = scalar_exc.fit_transform(remove_nans_and_infs(np.squeeze(data_exc)))
    data_exc_pca = normalize(data_exc_pca) 

    pca_x = PCA(whiten=True,random_state=40)


    # Project the data in 2D

    reduced_data_inh = pca_x.fit_transform(data_inh_pca[:min_size,:])
    exp_var_inh = pca_x.explained_variance_ratio_
    loadings = pca_x.components_.T * np.sqrt(pca_x.explained_variance_)

    n_components = 3

    kmeans = KMeans(n_clusters=5).fit(reduced_data_inh)
    centroids_inh = kmeans.cluster_centers_
    label = kmeans.labels_.astype(float)


    if plot_loadings:
        fig, ax = plt.subplots(1,3,figsize=[24,8])

        ax[0].scatter(reduced_data_inh[:,0], reduced_data_inh[:,1], c=label, s=50, alpha=0.5,marker = 'o')
        ax[0].scatter(centroids_inh[:, 0], centroids_inh[:, 1],c='black', s=50,marker = 'x')
        ax[0].set_xlabel('PC1')
        ax[0].set_ylabel('PC2')
        ax[0].set_title('Inhibitory')        
        for i, feature in enumerate(features):
            ax[1].plot([0,loadings[i, 0]],[0,loadings[i, 1]])
            ax[1].annotate(feature, xy = [loadings[i, 0], loadings[i, 1]])
        ax[2].scatter(np.arange(len(exp_var_inh)),exp_var_inh)
    else:
        plt.scatter(reduced_data_inh[:,0], reduced_data_inh[:,1], c=label, s=50, alpha=0.5,marker = 'o')
        plt.scatter(centroids_inh[:, 0], centroids_inh[:, 1],c='black', s=50,marker = 'x')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Inhibitory')           
    plt.show()

   

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


    if plot_loadings:
        fig, ax = plt.subplots(1,3,figsize=[24,8])
        ax[0].Projection ='3d'                   
        ax[0].scatter(reduced_data_exc[:,0], reduced_data_exc[:,1], c=label, s=50, alpha=0.5,marker = 'o')
        ax[0].scatter(centroids_exc[:, 0], centroids_exc[:, 1],c='black', s=50,marker = 'x')
        ax[0].set_xlabel('PC1')
        ax[0].set_ylabel('PC2')
        ax[0].set_title('Excitatory')

        for i, feature in enumerate(features):
            ax[1].plot([0,loadings[i, 0]],[0,loadings[i, 1]])
            ax[1].annotate(feature, xy = [loadings[i, 0], loadings[i, 1]])
        ax[2].scatter(np.arange(len(exp_var_exc)),exp_var_exc)
    else:
        plt.Projection ='3d'                   
        plt.scatter(reduced_data_exc[:,0], reduced_data_exc[:,1], c=label, s=50, alpha=0.5,marker = 'o')
        plt.scatter(centroids_exc[:, 0], centroids_exc[:, 1],c='black', s=50,marker = 'x')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Excitatory')
    plt.show()

def plot_pca_multiple_conditions(data,ax,plot_loadings=False,c='blue',type_c=None):
    """_summary_

    Args:
        data (list): _description_
    """
    features = ['Vm_avg','dvdt_p','dvdt_n','resistance','thr','adaptation',
    'isi','peak','peak_adaptation','ap_width','hyp_value','fist_spike','up_down_ratio',
    'isi_adaptation','thr_adp_ind','psth','int_fr','fr','sub_thr','spk_fr_adp','imp']
    print(np.array(data).shape)

    scalar_inh = StandardScaler()
    scalar_exc = StandardScaler()
    data_inh_pca = scalar_inh.fit_transform(remove_nans_and_infs(np.squeeze(data)))
    data_inh_pca = normalize(data_inh_pca) 


    pca_x = PCA(whiten=True,random_state=40)


    # Project the data in 2D

    reduced_data_inh = pca_x.fit_transform(data_inh_pca)
    exp_var_inh = pca_x.explained_variance_ratio_
    loadings = pca_x.components_.T * np.sqrt(pca_x.explained_variance_)

    n_components = 3

    kmeans = KMeans(n_clusters=5).fit(reduced_data_inh)
    centroids_inh = kmeans.cluster_centers_
    label = kmeans.labels_.astype(float)


    ax.scatter(reduced_data_inh[:,0], reduced_data_inh[:,1], c=c, s=50, alpha=0.5,marker = 'o')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(type_c)           

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
#%%
# data_d2 = pickle.load(open('G:/My Drive/all_ephys_d2.p','rb'))
# data_d1 = pickle.load(open('G:/My Drive/all_ephys_d1.p','rb'))
# data_sag = pickle.load(open('G:/My Drive/all_ephys_sag.p','rb'))
# data_acsf_exc = pickle.load(open('G:/My Drive/all_ephys_exc_NC_acsf_imp.p','rb'))
# data_acsf_inh = pickle.load(open('G:/My Drive/all_ephys_inh_NC_acsf_imp.p','rb'))
# ind_feat = [0,3,4,6,9,17,20]
# fig, ax_inh = plt.subplots(1,1,figsize=[10,8])

# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d2['inh']))),ax_inh,c='red',type_c='Inhibitory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_sag['inh']))),ax_inh,c='blue',type_c='Inhibitory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh']))),ax_inh,c='orange',type_c='Inhibitory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))),ax_inh,c='green',type_c='Inhibitory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))),ax_inh,c='green',type_c='Inhibitory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['inh_acsf']))),ax_inh,c='green',type_c='Inhibitory')

# plt.legend(['D2','sag','D1','acsf'])

# plt.show()

# fig, ax_exc = plt.subplots(1,1,figsize=[10,8])

# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d2['exc'])))[:,ind_feat],ax_exc,c='red',type_c='Excitatory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_sag['exc'])))[:,ind_feat],ax_exc,c='blue',type_c='Excitatory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc'])))[:,ind_feat],ax_exc,c='orange',type_c='Excitatory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf']))),ax_exc,c='green',type_c='Excitatory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf']))),ax_exc,c='green',type_c='Excitatory')
# plot_pca_multiple_conditions(remove_nans_and_infs(np.squeeze(np.array(data_d1['exc_acsf']))),ax_exc,c='green',type_c='Excitatory')

# plt.legend(['D2','sag','D1','acsf'])

# plt.show()

# %%
# data_d2 = pickle.load(open('G:/My Drive/all_ephys_d2.p','rb'))
# data_d1 = pickle.load(open('G:/My Drive/all_ephys_d1.p','rb'))
# data_sag = pickle.load(open('G:/My Drive/all_ephys_sag.p','rb'))
# data_acsf_exc = pickle.load(open('G:/My Drive/all_ephys_exc_NC_acsf_imp.p','rb'))
# data_acsf_inh = pickle.load(open('G:/My Drive/all_ephys_inh_NC_acsf_imp.p','rb'))
# ind_feat = [0,3,4,6,9,17,20]
# plot_pca(data_acsf_inh['all'],data_acsf_exc['all'],plot_loadings=True)
# print(data_acsf_inh['all'][:,ind_feat].shape)
# print(data_acsf_exc['all'][:,ind_feat].shape)

# %%
