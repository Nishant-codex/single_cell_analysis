#%%
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from scipy.sparse import data
from scipy.spatial.distance import cdist
from matplotlib.ticker import NullFormatter
from sklearn.cluster import KMeans
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.decomposition import SparsePCA
from sklearn import manifold, datasets
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import isnan
from utils import *
from analyze_single_cell import collect_drug_and_acsf
from impedance import *

def remove_nan(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = ~np.isnan(data)
    data_ = data[ind]
    return data_

def rolling_avg(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    start = 0
    length = len(data)//10
    end = len(data)
    finish = False
    avg = []

    if length == 0:
        return np.mean(data[start:end])
    else:
        while finish != True:
            if start+length < end:
                avg.append(np.mean(data[start:start+length]))
                start += length
            else:
                avg.append(np.mean(data[start:end]))
                finish = True

        return avg

def get_Vm(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    Vm = []
    V = data['membrane_potential']
    thr = data['thresholds']
    thr_ind = data['thresholdindices']
    spikes = data['spikeindices']
    ind = ~np.isnan(thr)
    spikes = spikes[ind]
    thr = thr[ind]
    thr_ind = thr_ind[ind]

    for i, j in zip(thr_ind, thr):
        Vm.append(V[int(i)+1:int(i)+50])
    return np.mean(Vm), Vm, np.mean(V)

def get_dvdt(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    dvdt_p = []
    dvdt_n = []

    for i in data:
        dv_ = np.diff(i)
        posp = np.where(dv_ > 0)
        dvdt_ind = np.zeros_like(dv_, dtype=bool)
        dvdt_ind[posp] = True
        posp = dvdt_ind
        posn = ~posp
        dvdt_p.append(np.mean(dv_[posp]))
        dvdt_n.append(np.mean(dv_[posn]))

    return np.mean(dvdt_p), np.mean(dvdt_n)

def sub_threhold_resistance(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    spikes = data['spikeindices']
    V = data['membrane_potential'][:spikes[0]]
    I = data['input_current'][:spikes[0]]
    R = np.mean(V/I)
    return R

def get_thresholds(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = ~np.isnan(data['thresholds'])
    return np.mean(data['thresholds'][ind])

def get_isi(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = ~np.isnan(data['thresholds'])
    return np.mean(np.diff(data['spikeindices'][ind]))

def get_adaptation(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = ~np.isnan(data['thresholds'])
    return np.mean(np.diff(data['thresholds'][ind]))

def get_AP_peak(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    max_v = []
    for i in data:
        max_v.append(np.max(i))
    return np.mean(max_v)

def get_AP_peak_adaptation(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    max_v = []
    for i in data:
        max_v.append(np.max(i))
    return np.mean(np.diff(max_v))

def get_AP_width(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    thr_ind = data['thresholdindices']
    thr = data['thresholds']
    ind = ~np.isnan(thr_ind)
    spks = data['spikeindices'][ind]
    thr = thr[ind]
    thr_ind = thr_ind[ind]
    V = data['membrane_potential']
    peak = 0
    width = []
    for i, j in zip(spks, thr_ind):
        try:
            spike_wf = V[int(j):int(j)+100]
            # plt.plot(spike_wf)
            left = V[int(j):i]
            right_ind = i-int(j)
            right = spike_wf[right_ind:]
            half_height = V[i]/2
            left_first = np.where(left <= half_height)
            right_first = np.where(right <= half_height)
            width.append((int(i-j)+right_first[0][0]+1)-(left_first[0][-1]))
        except:
            pass
    return np.mean(width)

def hyperpolarized_value(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.min(data['membrane_potential'])

def first_spike(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return data['thresholdindices'][0]

def get_up_down_ratio(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(data['Analysis']) > 1 and type(data['Analysis']) == list:
        avg_up = []
        avg_down = []
        for i in data['Analysis']:
            avg_up.append(i['nup'])
            avg_down.append(i['ndown'])
    else:
        avg_up = data['Analysis']['nup']
        avg_down = data['Analysis']['ndown']
    return np.nanmean(np.array(avg_up)/np.array(avg_down))

def subthreshold(data, subthreshold=False, plot=False):
    """_summary_

    Args:
        data (_type_): _description_
        subthreshold (bool, optional): _description_. Defaults to False.
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    V = data['membrane_potential']
    I = data['input_current']
    spikes = data['spikeindices']
    if data['input_generation_settings']['tau'] == 250:
        left = 20
        right = 30
    else:
        left = 20
        right = 20
    empty_cell = []
    for i in spikes:
        empty_cell.append([np.arange(i-left, i+right)])
        V1 = V[i-left]
        V2 = V[i+right]
        if V1 != V2:
            div = (V2-V1)/(left+right)
            V[i-left:i+right] = np.arange(V1, V2, div)[:left+right]
        else:
            V[i-left:i+right] = np.ones((1, left+right))*V1
        I1 = I[i-left]
        I2 = I[i+right]
        if I1 != I2:
            divI = (I2-I1)/(left+right)
            I[i-left:i+right] = np.arange(I1, I2, divI)[:left+right]
        else:
            I[i-left:i+right] = np.ones((1, left+right))*I1
    V_ = V
    I_ = I
    tailored_spikes = np.ndarray.flatten(np.array(empty_cell))
    zero_spikes = np.zeros(np.size(V))
    zero_spikes[tailored_spikes] = True
    if subthreshold == True:
        zero_spikes = ~zero_spikes
    return np.mean(V_)

def ISI_adaptation_index(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = ~np.isnan(data['thresholds'])
    ISI = np.diff(data['spikeindices'][ind])
    len_isi = len(ISI)
    fac = len_isi//10
    avgs = []
    for i in range(10):
        avg_ = np.mean(ISI[i*fac:(i+1)*fac])
        avgs.append(avg_)
    factors = []
    for j in range(9):
        factors.append(avgs[j]/avgs[j+1])
    return (np.mean(factors))

def threshold_adaptation_index(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    ind = ~np.isnan(data['thresholds'])
    thr = np.diff(data['thresholds'][ind])
    len_thr = len(thr)
    fac = len_thr//10
    avgs = []
    for i in range(10):
        avg_ = np.mean(thr[i*fac:(i+1)*fac])
        avgs.append(avg_)
    factors = []
    for j in range(9):
        factors.append(avgs[j]/avgs[j+1])
    return (np.mean(factors))

def PSTH(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    V_zero = np.zeros_like(data['membrane_potential'])
    thr = data['thresholdindices']
    ind = ~np.isnan(thr)
    spks = data['spikeindices'][ind]
    V_zero[spks] = 1
    count_spk = []
    start = 0
    width = 50000
    end = len(V_zero)
    run = True
    while run:
        if start+width > end:
            count_spk.append(sum(V_zero[start:end]))
            run = False
        else:
            count_spk.append(sum(V_zero[start:start+width]))
            start = start+width
    return np.mean(count_spk)

def get_inst_fr(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.mean(1/(np.diff(data['spikeindices'])))

def get_firing_rate(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.mean(data['firing_rate'])

def spike_frequency_adaptation(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    V_zero = np.zeros_like(data['membrane_potential'])
    thr = data['thresholdindices']
    ind = ~np.isnan(thr)
    spks = data['spikeindices'][ind]
    V_zero[spks] = 1
    count_spk = []
    start = 0
    width = 50000
    end = len(V_zero)
    run = True
    while run:
        if start+width > end:
            count_spk.append(sum(V_zero[start:end]))
            run = False
        else:
            count_spk.append(sum(V_zero[start:start+width]))
            start = start+width
    return np.mean(np.diff(count_spk))

def get_impedence(data:list)->float:
    """_summary_

    Args:
        data (list): _description_

    Returns:
        float: _description_
    """
    I_acsf = data['input_current']
    V_acsf = data['membrane_potential']
    spk_acsf, V_acsf, I_acsf = return_stiched_spike_train(data)
    imp = overdracht_wytse(0.01, I_acsf, V_acsf, 20001, 20001, 1)
    return np.mean(imp)

def get_ephys_vals(data_i):
    """_summary_

    Args:
        data_i (dict): _description_

    Returns:
        _type_: _description_
    """

    # try:
    Vm_avg, Vm, _ = get_Vm(data_i)
    dvdt_p, dvdt_n = get_dvdt(Vm)
    resistance = sub_threhold_resistance(data_i)
    thr = get_thresholds(data_i)
    adaptation = get_adaptation(data_i)
    isi = get_isi(data_i)
    peak = get_AP_peak(Vm)
    peak_adaptation = get_AP_peak_adaptation(Vm)
    ap_width = get_AP_width(data_i)
    hyp_value = hyperpolarized_value(data_i)
    fist_spike = first_spike(data_i)
    up_down_ratio = get_up_down_ratio(data_i)
    isi_adaptation = ISI_adaptation_index(data_i)
    thr_adp_ind = threshold_adaptation_index(data_i)
    psth = PSTH(data_i)
    int_fr = get_inst_fr(data_i)
    fr = get_firing_rate(data_i)
    sub_thr = subthreshold(data_i)
    spk_fr_adp = spike_frequency_adaptation(data_i)
    imp = get_impedence(data_i)

    ephys_data =      [Vm_avg, #
                        dvdt_p,
                        dvdt_n,
                        resistance, #
                        thr,#
                        adaptation,#
                        isi,#
                        peak,
                        peak_adaptation,
                        ap_width, #
                        hyp_value,
                        fist_spike,
                        up_down_ratio,
                        isi_adaptation,
                        thr_adp_ind,
                        psth,
                        int_fr,
                        fr, #
                        sub_thr,
                        spk_fr_adp,
                        imp] #

    return ephys_data

def return_all_ephys_dict(cond:list, experimenter:str=None)->dict:
    """returns a dictonary with all the ephys properties for each cell for the 
    condition provided along with the aCSF counterpart. 
    Exc and inhibitory cells are segregated.
    
    Args:
        cond (list): a list containing the condion to be analyzed
        experimenter (str, optional): if a specific experimenter needs to aanlyzed seperately. Defaults to None.

    Raises:
        ValueError:  'condition should be a list even if a single value is provided'

    Returns:
        dict: dictionary containing all e-phys features for each cell  
    """

    all_ephys_with_cond = {}
    path_i = 'C:/Users/Nishant Joshi/Google Drive/Analyzed/'
    if type(cond) != list:
        raise ValueError(
            'condition should be a list even if a single value is provided')
    cond_i = cond  
    new_a = join_conditions(list_cond=cond_i)
    if experimenter != None:
        new_a = new_a.groupby('experimenter').get_group(experimenter)
    new_a_inh = new_a.groupby('tau').get_group(50)
    new_a_exc = new_a.groupby('tau').get_group(250)

    exp_name_inh = np.array(new_a_inh['experimentname'])
    trials_inh = np.array(new_a_inh['trialnr'])
    exp_name_exc = np.array(new_a_exc['experimentname'])
    trials_exc = np.array(new_a_exc['trialnr'])
    all_ephys_data_inh = []
    all_ephys_data_exc = []
    all_ephys_data_inh_acsf = []
    all_ephys_data_exc_acsf = []
    problem_cell = []
    count = 0

    for i, j in zip(exp_name_exc, trials_exc-1):
        count += 1
        print(count)
        try:
            data = loadmatInPy(path_i + i + '_analyzed.mat')
        except:
            data = loadmatInPy(path_i + 'Copy of ' + i + '_analyzed.mat')
        for instance in data:
            if instance['input_generation_settings']['condition'] in ['ACSF', 'aCSF']:
                all_ephys_data_exc_acsf.append(get_ephys_vals(instance))
        try:
            data_i = data[j]
        except:
            if len(data) == 1:
                data_i = data
            else:
                problem_cell.append(i)
                pass
        print(i, data_i['input_generation_settings']['condition'])
        all_ephys_data_exc.append(get_ephys_vals(data_i))

    for i, j in zip(exp_name_inh, trials_inh-1):
        count += 1
        try:
            data = loadmatInPy(path_i + i + '_analyzed.mat')
        except:
            data = loadmatInPy(path_i + 'Copy of ' + i + '_analyzed.mat')
        for instance in data:
            if instance['input_generation_settings']['condition'] in ['ACSF', 'aCSF']:
                all_ephys_data_inh_acsf.append(get_ephys_vals(instance))
        try:
            data_i = data[j]
        except:
            if len(data) == 1:
                data_i = data
            else:
                problem_cell.append(i)
                pass
        print(i, data_i['input_generation_settings']['condition'])
        all_ephys_data_inh.append(get_ephys_vals(data_i))

    all_ephys_with_cond['exc'] = all_ephys_data_exc
    all_ephys_with_cond['inh'] = all_ephys_data_inh
    all_ephys_with_cond['exc_acsf'] = all_ephys_data_exc_acsf
    all_ephys_with_cond['inh_acsf'] = all_ephys_data_inh_acsf
    all_ephys_with_cond['cond'] = cond
    return all_ephys_with_cond


# %%
data  = return_all_ephys_dict(['aCSF','acsf'])