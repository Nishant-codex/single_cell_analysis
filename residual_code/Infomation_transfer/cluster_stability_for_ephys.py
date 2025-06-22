import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
from sknetwork.clustering import Louvain,get_modularity
import sys 
import os 
from Cluster_stability import *
import pickle
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler



if __name__ == "__main__":

    path_save = os.getcwd()
    data_umap_scaler = StandardScaler()

    ephys = pd.read_csv(os.getcwd()+"/Cluster_stability/ephys_EI_class.csv")
    biophys = pd.read_csv(os.getcwd()+"/Cluster_stability/biophys_EI_class.csv")
    cols_to_compare_ephys = ephys.columns[1:-5]
    cols_to_compare_biophys = ['tau_m (ms)', 'R (MOhm):', 'C (nF):', 'gl (nS):', 'El (mV):', 'Vr (mV):', 'Vt* (mV):', 'DV (mV):']


    ephys_E = ephys[ephys.labels_wave ==1]
    ephys_I = ephys[ephys.labels_wave ==0]

    biophys_E = biophys[biophys.labels_wave ==1]
    biophys_I = biophys[biophys.labels_wave ==0]

    # ephys_e_cluster_data = find_optimum_res_with_cols(data_umap_scaler.fit_transform(normalize(ephys_E[cols_to_compare_ephys].to_numpy())),
    #                                                   list(cols_to_compare_ephys))

    # with open(path_save+"/cluster_stablity_ephys_E.pkl",'wb') as f:
    #     pickle.dump(ephys_e_cluster_data, file=f)


    # ephys_i_cluster_data = find_optimum_res_with_cols(data_umap_scaler.fit_transform(normalize(ephys_I[cols_to_compare_ephys].to_numpy())),
    #                                                   list(cols_to_compare_ephys))

    # with open(path_save+"/cluster_stablity_ephys_I.pkl",'wb') as f:
    #     pickle.dump(ephys_i_cluster_data, file=f)


    biophys_e_cluster_data = find_optimum_res_with_cols(data_umap_scaler.fit_transform(normalize(biophys_E[cols_to_compare_biophys].to_numpy())),
                                                        list(cols_to_compare_biophys))

    with open(path_save+"/cluster_stablity_biophys_E.pkl",'wb') as f:
        pickle.dump(biophys_e_cluster_data, file=f)


    biophys_i_cluster_data = find_optimum_res_with_cols(data_umap_scaler.fit_transform(normalize(biophys_I[cols_to_compare_biophys].to_numpy())),
                                                        list(cols_to_compare_biophys))


    with open(path_save+"/cluster_stablity_biophys_I.pkl",'wb') as f:
        pickle.dump(biophys_i_cluster_data, file=f)
        
