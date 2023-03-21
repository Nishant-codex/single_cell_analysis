import os
import pickle
from typing import Set
import numpy as np
from numpy.lib.function_base import append
import scipy.io as spio
from scipy.io import loadmat, savemat
import importlib.util
from sklearn import datasets, linear_model
from scipy.sparse import data
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from plotnine.data import mtcars
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import SparsePCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from pylab import *
from scipy.signal import find_peaks


def plot_single_cell(path_cc: str, filename: str,plottype=None):
    """ to plot single CC cell, voltage and input current side by side 

    Args:
        path_cc (str): path to all CC files 
        filename (str): name of the file to be plotted 
    """
    if plottype==None:
        raise('Invalid value, it should be \'stacked\' or \'expanded\'')

    elif plottype == 'stacked':
        stacked=True
        expand = False

    elif plottype == 'expanded':
        stacked= False
        expand = True

    def order_list(data:list)->list:
        """ somtimes the list of keys are ordered such that the last value is at the beginning. This function sorts it. 

        Args:
            data (_type_): the list of keys  

        Returns:
            list : returns an ordered list 
        """
        if data[0][-3] == '0':
            temp_list = data[2:]
            rear_vals = data[:2]
            return temp_list+rear_vals
        else:
            return data
    data = loadmat(path_cc+filename)
    keys = data.keys()
    keys = list(keys)[3:]
    dt = 1/20000
    searchthreshold = 0
    thresholdwindow = [1, 0.25]
    refractory_period = 3
    derthreshold = [[1, 0.000008], [2, 0.000008]]
    # nwindow = [np.round(thresholdwindow[0]*dt), np.round(thresholdwindow[1]/dt)]
    nwindow = [30, 70]
    trials = len(keys)//20
    return_dict = {}
    trial_steps = []
    for t in range(trials):
        trial_step = order_list(keys[len(keys)//trials*t:len(keys)//trials*(t+1)])
        trial_steps.append(trial_step)

    return_dict = {}
    for ind_trial, trial in enumerate(trial_steps):
        Is, tI, Vs, tV = [], [], [], []
        fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        for index, step in enumerate(trial):
            if step[-1] == '2':

                Vs.append(data[step][:, 1])
                tV.append(data[step][:, 0])
                if stacked:
                    ax[0].plot(data[step][:, 0],data[step][:, 1])
            elif step[-1] == '1':
                I = data[step][:, 1]
                Is.append(data[step][:, 1])
                tI.append(data[step][:, 0])
                if stacked:
                    ax[1].plot(data[step][:, 0],data[step][:, 1])
        if expand:
            V = Vs
            # spk_ind, thr, thr_ind = get_threshold_fontaine(np.expand_dims(np.array(V).flatten(),axis=1),dt=dt,searchthreshold = searchthreshold,windown = nwindow,refractory_period = refractory_period,derthreshold =derthreshold )
            ax[0].plot(np.array(V).flatten())
            # V = np.array(V).flatten()
            # ax[0].plot(spk_ind,V[spk_ind],'x', markersize=12)
            # ax[0].plot(thr_ind,V[thr_ind],'o', markersize=4)
            ax[1].plot(np.array(Is).flatten())
            plt.show()


def returnVsandIs(path_cc: str, filename: str)->dict:
    """ returns all the Vs and Is combined for all steps  

    Args:
        path_cc (str): path to all CC files 
        filename (str): filename

    Returns:
        dict: a dictionary object containing all Vs and Is for all steps 
    """
    def order_list(data):
        if data[0][-3] == '0':
            temp_list = data[2:]
            rear_vals = data[:2]
            return temp_list+rear_vals
        else:
            return data
    data = loadmat(path_cc+filename)
    keys = data.keys()
    keys = list(keys)[3:]

    trials = len(keys)//20
    return_dict = {}
    trial_steps = []
    for t in range(trials):
        trial_step = order_list(
            keys[len(keys)//trials*(t):len(keys)//trials*(t+1)])
        trial_steps.append(trial_step)

    return_dict = {}
    for ind_trial, trial in enumerate(trial_steps):
        Is, tI, Vs, tV = [], [], [], []
        for index, step in enumerate(trial):
            if step[-1] == '2':
                Vs.append(data[step][:, 1])
                tV.append(data[step][:, 0])
            elif step[-1] == '1':
                Is.append(data[step][:, 1])
                tI.append(data[step][:, 0])
        return_dict[str(ind_trial+1)] = {'V': Vs, 'I': Is, 'tV': tV, 'tI': tI}
    return return_dict


def plot_steps(path_cc,filename):
    def order_list(data):
        if data[0][-3] == '0':
            temp_list = data[2:]
            rear_vals = data[:2]
            return temp_list+rear_vals
        else:
            return data
    data = loadmat(path_cc+filename)
    keys = data.keys()
    keys = list(keys)[3:]
    dt = 1/20000
    searchthreshold = 0
    thresholdwindow = [1, 0.25]
    refractory_period = 3
    derthreshold = [[1, 0.000008], [2, 0.000008]]
    # nwindow = [np.round(thresholdwindow[0]*dt), np.round(thresholdwindow[1]/dt)]
    nwindow = [30, 70]
    trials = len(keys)//20
    return_dict = {}
    trial_steps = []
    for t in range(trials):
        trial_step = order_list(
            keys[len(keys)//trials*t:len(keys)//trials*(t+1)])
        trial_steps.append(trial_step)

    return_dict = {}
    for ind_trial, trial in enumerate(trial_steps):
        Is, tI, Vs, tV = [], [], [], []
        fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        for index, step in enumerate(trial):
            if step[-1] == '2':

                Vs.append(data[step][:, 1])
                tV.append(data[step][:, 0])
            elif step[-1] == '1':
                I = data[step][:, 1]
                Is.append(data[step][:, 1])
                tI.append(data[step][:, 0])
    V = Vs
    spk_ind, thr, thr_ind = get_threshold_fontaine(np.expand_dims(np.array(V).flatten(
    ), axis=1), dt=dt, searchthreshold=searchthreshold, windown=nwindow, refractory_period=refractory_period, derthreshold=derthreshold)
    ax[0].plot(np.array(V).flatten())
    V = np.array(V).flatten()
    ax[0].plot(spk_ind, V[spk_ind], 'x', markersize=12)
    ax[0].plot(thr_ind, V[thr_ind], 'o', markersize=4)
    ax[1].plot(np.array(Is).flatten())
    plt.show()

    return_dict[str(ind_trial+1)] = {'V': Vs, 'I': Is, 'tV': tV, 'tI': tI}


def get_threshold_fontaine(membrane_potential, dt, searchthreshold, windown, refractory_period=0, derthreshold=[1, 5], plotyn=0):

    # Based on Fontaine, Peña, Brette, PLoS Comp Bio 2014: Spike-Threshold
    # Adaptation Predicted by Membrane Potential Dynamics In Vivo.

    # First value of which each of the derivatives specified in derthreshold
    # exceeds the specified value

    [Nder, _] = np.shape(derthreshold)
    _, Nvolt = membrane_potential.shape
    if Nvolt == 1:
        membrane_potential = membrane_potential.T

    DerMat = np.nan*np.zeros((Nder, membrane_potential.size))
    for nd in range(Nder):

        derorder = np.array(derthreshold, dtype=np.int32)[nd, 0]
        der = np.diff(membrane_potential, derorder)
        if derorder == 1:
            add_ = np.zeros((1, 1))
            add_[:, :] = np.nan
            der = np.concatenate((der, add_), axis=1)
        elif derorder % 2 == 1:
            der = np.concatenate(
                (np.zeros((1, derorder-1)), der, np.zeros((1, derorder))), axis=1)
        else:
            der = np.concatenate(
                (np.zeros((1, derorder-1)), der, np.zeros((1, derorder-1))), axis=1)
        DerMat[nd, :] = der

    # _, spikeindices = findspikes(dt, membrane_potential, searchthreshold, 0, refractory_period)

    spikeindices, _ = find_peaks(
        membrane_potential[0, :], height=0, distance=20)
    spikeindices = np.expand_dims(spikeindices, axis=0)
    if spikeindices.size > 1:
        spikeindices = np.array(spikeindices[0, :], dtype=np.int32)
        Nspikes = np.size(spikeindices)
    else:
        spikeindices = np.array([spikeindices])
        Nspikes = np.size(spikeindices)
    thresholds = np.zeros_like(spikeindices, dtype=np.float32)

    if thresholds.size > 1:
        thresholds[:] = np.nan
    thresholdindices = np.zeros_like(spikeindices, dtype=np.float32)
    if thresholdindices.size > 1:

        thresholdindices[:] = np.nan

    for ns in range(Nspikes):

        if (spikeindices[ns] - windown[0]) > 0:

            nn = np.arange(int(
                spikeindices[ns] - windown[0]), int(spikeindices[ns]+windown[1]), dtype=np.int64)
        elif (spikeindices[ns] - windown[1]) > 0:
            nn = np.arange(1, spikeindices[ns]+windown[1], dtype=np.int64)
        else:
            thresholdindices[ns] = 1
            thresholds[ns] = np.nan
            continue

        vm_rel = membrane_potential[0, nn]
        for nd in range(Nder):
            threshold_temp = np.array(derthreshold)[nd, 1]
            der = DerMat[nd, nn]
            vm_rel[der < threshold_temp] = np.nan

        thresholdindex = np.where(~np.isnan(vm_rel))
        # print(thresholdindex)
        if thresholdindex[0].size > 0:
            thresholdindices[ns] = spikeindices[ns] - \
                windown[0]+thresholdindex[0][0]
            thresholds[ns] = vm_rel[thresholdindex[0][0]]
            if thresholds[ns] > 0.0:
                thresholds[ns] = np.nan
            elif thresholds[ns] < -0.08:
                thresholds[ns] = np.nan
        else:
            thresholds[ns] = np.nan

    return np.array(spikeindices, dtype=np.int32), thresholds, np.array(thresholdindices, dtype=np.int32)


def check_for_faultycell(val_dict, name, exceptions=None):
    """_summary_

    Args:
        val_dict (_type_): _description_
        name (_type_): _description_
        exceptions (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if exceptions == None:
        trials = len(val_dict.keys())
        if trials == 1:
            for v in val_dict['1']['V']:
                if len(v[v < -0.08]) > 10:
                    print(name + ' is faulty')
                    return True
                else:
                    return False
        else:
            for k, vals in val_dict.items():
                for v in vals['V']:
                    if len(v[v < -0.08]) > 10:
                        return True
                    else:
                        return False
    elif name == exceptions:
        return True


def collect_all_spike_data(path_cc, df_CC_exp , condition):
    """_summary_

    Args:
        condition (_type_): _description_

    Returns:
        _type_: _description_
    """
    dt = 1/20000
    searchthreshold = 0
    thresholdwindow = [1, 0.25]
    refractory_period = 3
    derthreshold = [[1, 0.000008], [2, 0.000008]]
    # nwindow = [np.round(thresholdwindow[0]*dt), np.round(thresholdwindow[1]/dt)]
    nwindow = [30, 70]
    cond = condition
    file_cond = df_CC_exp[df_CC_exp['condition'] ==
                          cond][df_CC_exp['drug'] == False]['CC_files']
    file_cond_acsf = df_CC_exp[df_CC_exp['condition']
                               == cond][df_CC_exp['drug'] == True]['CC_files']

    # print(len(file_cond),len(file_cond_acsf))
    files_with_spks_and_thresholds_drug = []
    files_with_spks_and_thresholds_acsf = []
    # print(file_cond_acsf)
    for drug_file, acsf_file in zip(np.array(file_cond), np.array(file_cond_acsf)):
        print(drug_file)
        I_means_drug = []
        spikes_and_thrs_drug = []
        value_dict_drug = returnVsandIs(path_cc , drug_file)
        if check_for_faultycell(value_dict_drug, drug_file):
            print('faulty')
            pass
        else:
            if len(value_dict_drug.keys()) == 1:
                for i in value_dict_drug['1']['I']:
                    I_means_drug.append(np.mean(i))
            else:
                for k, vals in value_dict_drug.items():
                    tempI = []
                    for j in vals['I']:
                        tempI.append(np.mean(j))
                    I_means_drug.append(tempI)
            try:

                for trial, vals in value_dict_drug.items():
                    spikes_and_thrs_trials = []
                    for steps in vals['V']:
                        V = steps
                        if drug_file == 'NC_16-8-17-E1-CCSTEP-DRUG.mat':
                            plt.plot(V)
                            plt.show()

                        spk_ind, thr, thr_ind = get_threshold_fontaine(np.expand_dims(
                            V, axis=1), dt=dt, searchthreshold=searchthreshold, windown=nwindow, refractory_period=refractory_period, derthreshold=derthreshold)
                        spikes_and_thrs_trials.append({'spks': spk_ind,
                                                       'thrs': thr,
                                                       'thr_ind': thr_ind})
                    spikes_and_thrs_drug.append(spikes_and_thrs_trials)
            except:
                print('problem with '+drug_file)

                # spikes_and_thrs_drug.append({})
            files_with_spks_and_thresholds_drug.append(
                {'avg_I': I_means_drug, 'spike_info': spikes_and_thrs_drug, 'filename': drug_file})

        I_means_acsf = []
        spikes_and_thrs_acsf = []

        value_dict_acsf = returnVsandIs(path_cc, acsf_file)
        if check_for_faultycell(value_dict_acsf, acsf_file):
            pass
        else:
            if len(value_dict_acsf.keys()) == 1:
                for i in value_dict_acsf['1']['I']:
                    I_means_acsf.append(np.mean(i))
            else:
                for k, vals in value_dict_acsf.items():
                    tempI = []
                    for j in vals['I']:
                        tempI.append(np.mean(j))
                    I_means_acsf.append(tempI)
            try:

                for trial, vals in value_dict_acsf.items():
                    spikes_and_thrs_trials = []
                    for steps in vals['V']:
                        V = steps
                        spk_ind, thr, thr_ind = get_threshold_fontaine(np.expand_dims(
                            V, axis=1), dt=dt, searchthreshold=searchthreshold, windown=nwindow, refractory_period=refractory_period, derthreshold=derthreshold)
                        spikes_and_thrs_trials.append({'spks': spk_ind,
                                                       'thrs': thr,
                                                       'thr_ind': thr_ind})
                    spikes_and_thrs_acsf.append(spikes_and_thrs_trials)
            except:
                print('problem with '+acsf_file)

            files_with_spks_and_thresholds_acsf.append(
                {'avg_I': I_means_acsf, 'spike_info': spikes_and_thrs_acsf, 'filename': acsf_file})
    return [files_with_spks_and_thresholds_drug, files_with_spks_and_thresholds_acsf]


def return_name_date_exp_fn(string):
    if 'NC' in string:
        string_broken = string.split('_')
        name = string_broken[0]
        date = string_broken[1]
        exp = string_broken[-1]
        year = date[:2]
        month = date[2:4]

        if month[0] == '0':
            month = month[1]
        day = date[4:]
        if day[0] == '0':
            day = day[1]
        date = day+month+year
        return name+'_'+date+'_'+exp
    elif 'xuan' in string:
        broken_str = string.split('_')
        name = broken_str[0]
        date = broken_str[1].replace('-', '')
        exp = broken_str[2]
        return name+'_'+date+'_'+exp
    elif 'asli' in string:
        broken_str = string.split('_')
        name = broken_str[0]
        date = broken_str[1].replace('-', '')
        exp = broken_str[2]
        return name+'_'+date+'_'+exp
    elif 'Payam' in string:
        broken_str = string.split('_')
        name = broken_str[0].lower()
        date = broken_str[1].split('-')
        exp = broken_str[2]
        day = date[0]
        month = date[1]
        year = date[2]
        if day[0] == '0':
            day = day[1]
        date = day+month+year
        return name+'_'+date+'_'+exp
