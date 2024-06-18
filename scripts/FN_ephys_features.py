#%%
import pandas as pd
import numpy as np
import seaborn as sns
import os 
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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
from sklearn.linear_model import LinearRegression
import neo
from quantities import *
from elephant import statistics
from elephant.kernels import GaussianKernel
from elephant.statistics import isi, cv
from elephant.statistics import time_histogram, instantaneous_rate
from elephant import sta



#%%



class EphysSet:

    def __init__(self,data,cond,exp_name,trialnr):

        self.data = data
        self.cond = cond
        self.exp_name = exp_name
        self.trialnr = trialnr
        self.V = self.data['membrane_potential']
        self.dt = 1/20 

    def remove_nan(self,data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(data)
        data_ = data[ind]
        return data_

    def rolling_avg(self,data):
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

    def get_Vm(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        Vm = []
        V = self.data['membrane_potential']
        thr = self.data['thresholds']
        thr_ind = self.data['thresholdindices']
        spikes = self.data['spikeindices']
        ind = ~np.isnan(thr)
        spikes = spikes[ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]

        for i, j in zip(thr_ind, thr):
            Vm.append(V[int(i)-20*4:int(i)+20*5])
        if return_mean:
            return np.mean(Vm), Vm, np.mean(V)
        else:
            return Vm
    
    def get_dvdt(self,data,return_mean=True):
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
        if return_mean:
            return np.mean(dvdt_p), np.mean(dvdt_n),np.max(dvdt_p), np.max(dvdt_n),np.min(dvdt_p), np.min(dvdt_n)
        else:
            return dvdt_p, dvdt_n
    
    def sub_threshold_resistance(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        spikes = self.remove_nan(self.data['thresholdindices'])

        V = self.data['membrane_potential'][:int(spikes[0])-100]
        I = self.data['input_current'][:int(spikes[0])-100].reshape((-1, 1))
        
        model = LinearRegression()
        model.fit(I, V)

        R = model.coef_

        return R

    def get_thresholds(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        if return_mean:
            return np.mean(self.data['thresholds'][ind])
        else:
            return self.data['thresholds'][ind]
        
    def get_isi(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        if return_mean:
            return np.mean(np.diff(self.data['spikeindices'][ind])*dt)
        else:
            return np.diff(self.data['spikeindices'][ind])*dt
        
    def get_threshold_adaptation(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        if return_mean:
            return np.mean(np.diff(self.data['thresholds'][ind]))
        else: 
            return np.diff(self.data['thresholds'][ind])

    def get_AP_peak(self,spike_waves,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_v = []
        for i in spike_waves:
            max_v.append(np.max(i))
        if return_mean:
            return np.mean(max_v)
        else:
            return max_v

    def get_AP_peak_adaptation(self,spike_waves,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_v = []
        for i in spike_waves:
            max_v.append(np.max(i))
        if return_mean:
            return np.mean(np.diff(max_v))
        else:
            return np.diff(max_v)

    def get_AP_width(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        thr_ind = self.data['thresholdindices']
        thr = self.data['thresholds']
        ind = ~np.isnan(thr_ind)
        spks = self.data['spikeindices'][ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]
        V = self.data['membrane_potential']
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
        if return_mean:
            return np.mean(width)
        else:
            return width

    def hyperpolarized_value(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.min(self.data['membrane_potential'])

    def first_spike(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.data['thresholdindices'][0]

    def get_up_down_ratio(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(self.data['Analysis']) > 1 and type(self.data['Analysis']) == list:
            avg_up = []
            avg_down = []
            for i in self.data['Analysis']:
                avg_up.append(i['nup'])
                avg_down.append(i['ndown'])
        else:
            avg_up = self.data['Analysis']['nup']
            avg_down = self.data['Analysis']['ndown']
        return np.nanmean(np.array(avg_up)/np.array(avg_down))

    def subthreshold(self, subthreshold=False, plot=False,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_
            subthreshold (bool, optional): _description_. Defaults to False.
            plot (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        V = self.data['membrane_potential']
        I = self.data['input_current']
        spikes = self.data['spikeindices']
        if self.data['input_generation_settings']['tau'] == 250:
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
        if return_mean:
            return np.mean(V_)
        else:
            return V_
        
    def isi_adaptation_index(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        ISI = np.diff(self.data['spikeindices'][ind])
        len_isi = len(ISI)
        fac = len_isi//10
        avgs = []
        for i in range(10):
            avg_ = np.mean(ISI[i*fac:(i+1)*fac])
            avgs.append(avg_)
        factors = []
        for j in range(9):
            factors.append(avgs[j]/avgs[j+1])
        if return_mean:                
            return (np.mean(factors))
        else:
            return factors

    def threshold_adaptation_index(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        thr = np.diff(self.data['thresholds'][ind])
        len_thr = len(thr)
        fac = len_thr//10
        avgs = []
        for i in range(10):
            avg_ = np.mean(thr[i*fac:(i+1)*fac])
            avgs.append(avg_)
        factors = []
        for j in range(9):
            factors.append(avgs[j]/avgs[j+1])
        if return_mean:        
            return (np.mean(factors))
        else:
            return factors

    def psth(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        V_zero = np.zeros_like(self.data['membrane_potential'])
        thr = self.data['thresholdindices']
        ind = ~np.isnan(thr)
        spks = self.data['spikeindices'][ind]
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
        if return_mean:        
            return np.mean(count_spk)
        else:
            return count_spk

    def get_inst_fr(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if return_mean:        
            return np.mean(1/(np.diff(self.data['spikeindices'])))
        else:
            return 1/(np.diff(self.data['spikeindices']))

    def get_MI(self,return_mean=True):
        if type(self.data['Analysis']) == list:
            return np.mean([i['FI'] for i in self.data['Analysis']])
        else:
            return self.data['Analysis']['FI']
    
    def get_firing_rate(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if return_mean:        
            return np.mean(self.data['firing_rate'])
        else:
            return self.data['firing_rate']

    def spike_frequency_adaptation(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        V_zero = np.zeros_like(self.data['membrane_potential'])
        thr = self.data['thresholdindices']
        ind = ~np.isnan(thr)
        spks = self.data['spikeindices'][ind]
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

        if return_mean:        
            return np.mean(np.diff(count_spk))
        else:
            return np.diff(count_spk)

    def get_impedence(self,return_mean=True):
        """_summary_

        Args:
            data (list): _description_

        Returns:
            float: _description_
        """
        I_acsf = self.data['input_current']
        V_acsf = self.data['membrane_potential']
        spk_acsf, V_acsf, I_acsf = return_stiched_spike_train(self.data)
        imp,fas = overdracht_wytse(0.01, I_acsf, V_acsf, 20001, 20001, 1)
        if return_mean:
            return np.mean(imp)
        else:
            return imp,fas
    
    def get_ephys_vals(self):
        """_summary_

        Args:
            data_i (dict): _description_

        Returns:
            _type_: _description_
        """

        Vm_avg, Vm, avg_V = self.get_Vm()
        dvdt_p, dvdt_n = self.get_dvdt(Vm)
        resistance = self.sub_threshold_resistance()
        thr = self.get_thresholds()
        adaptation = self.get_threshold_adaptation()
        isi = self.get_isi()
        peak = self.get_AP_peak(Vm)
        peak_adaptation = self.get_AP_peak_adaptation(Vm)
        ap_width = self.get_AP_width()
        hyp_value = self.hyperpolarized_value()
        fist_spike = self.first_spike()
        up_down_ratio = self.get_up_down_ratio()
        isi_adaptation = self.isi_adaptation_index()
        thr_adp_ind = self.threshold_adaptation_index()
        psth = self.psth()
        mi = self.get_MI()
        int_fr = self.get_inst_fr()
        fr = self.get_firing_rate()
        sub_thr = self.subthreshold()
        spk_fr_adp = self.spike_frequency_adaptation()
        imp = self.get_impedence()

        ephys_data =      [Vm, 
                           Vm_avg, #
                            dvdt_p,
                            dvdt_n,
                            avg_V,
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
                            mi,
                            spk_fr_adp,
                            imp,
                            self.exp_name,
                            self.cond,
                            self.trialnr] #
        return ephys_data

class EphysSet_niccolo:

    def __init__(self,data,cond,exp_name,trialnr,run_half=False,compute_spikes=False):

        self.data = data
        self.cond = cond.lower()
        self.exp_name = exp_name
        self.trialnr = trialnr
        self.compute_spikes = compute_spikes
        self.wavelen_left = 5
        self.wavelen_right = 5
        if run_half:
            self.V = self.data['membrane_potential']
            total_length =len(self.V)
            self.I = self.data['input_current']
            self.h = self.data['hidden_state']
            if compute_spikes==False:
                self.spikeindices = data['spikeindices']
                self.spikeindices = data['spikeindices']
                self.thresholdindices = data['thresholdindices']
                self.thresholdindices = data['thresholdindices'][:len(self.spikeindices)]
                self.thresholds = data['thresholds'][:len(self.spikeindices)]
            else:
                self.compute_spikes_and_thresholds()                   

        else:
            self.V = self.data['membrane_potential']
            self.I = self.data['input_current']
            self.h = self.data['hidden_state']
            if compute_spikes==False:
                self.thresholds = data['thresholds']
                self.thresholdindices = data['thresholdindices']
                self.spikeindices = data['spikeindices']  
            else:
                self.compute_spikes_and_thresholds()    
            
        self.tau = self.data['input_generation_settings']['tau']
        self.dt = 1/20 

    def remove_nan(self,data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(data)
        data_ = data[ind]
        return data_
    
    def get_Vm(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.compute_spikes:
            if return_mean:
                # plt.plot(self.waveforms.T)
                return self.waveforms,np.mean(self.waveforms,axis=0),np.mean(self.V)
            
            else:
                return self.waveforms
        
        else:
            Vm = []
            V = self.V
            thr = self.thresholds
            thr_ind = self.thresholdindices
            spikes = self.spikeindices
            ind = ~np.isnan(thr)
            spikes = spikes[ind]
            thr = thr[ind]
            thr_ind = thr_ind[ind]

            for i, j in zip(thr_ind, thr):
                Vm.append(V[int(i)-20*self.wavelen_left:int(i)+20*self.wavelen_right])
            if return_mean:
                return Vm, np.mean(Vm,axis=0), np.mean(V)
            else:
                return Vm
        
    def compute_thresholds(self,waveforms):
        threshold_inds = []
        thresholds = [] 

        for wave,spk_i in zip(waveforms,self.spikeindices):
            thr = np.where((np.diff(wave)*20)>25)[0]

            if len(thr>0):

                thr_val = wave[thr[0]]
                thr_ind = (spk_i-self.wavelen_left*20)+thr[0]

                threshold_inds.append(thr_ind)
                thresholds.append(thr_val)
            else:
                threshold_inds.append(100.) #since thresholds cannot be calculated, a high place holder value is added
                thresholds.append(100.)

        self.bool_inds = np.array(thresholds)>0 
        self.thresholdindices = np.array(threshold_inds)[~self.bool_inds]
        self.thresholds = np.array(thresholds)[~self.bool_inds]
        self.spikeindices = self.spikeindices[~self.bool_inds]

    def compute_spikes_and_thresholds(self):
        V = self.V
        self.spikeindices =  find_peaks(V,height=30,distance=4*20)[0]
        waveforms = []

        for i in self.spikeindices:
            waveforms.append(V[i-self.wavelen_left*20:i+self.wavelen_right*20])
        self.compute_thresholds(waveforms)
        self.waveforms = np.array(waveforms)[~self.bool_inds]



    def rolling_avg(self,data):
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

    def return_quant_divided_by_time(self,divisions,quant):
        total_duration = len(self.V)/20
        quantity = quant
        time_ranges = np.arange(0,total_duration+1,total_duration//divisions)
        spike_times = self.spikeindices/20
        time_divided_spiketimes = [np.array(spike_times[np.where(np.logical_and(spike_times>time_ranges[i],spike_times<=time_ranges[i+1] ))[0]]*20,dtype=np.int32)
        for i in range(len(time_ranges)-1)]
        last = 0
        vals_divided_by_time = []
        for i in time_divided_spiketimes:
            vals_divided_by_time.append(quantity[last:last+len(i)])
            last = len(i)
        return vals_divided_by_time  

    def fano_factor(self,divisions):
        total_duration = len(self.V)/20
        time_ranges = np.arange(0,total_duration+1,total_duration//divisions)
        spike_times = self.spikeindices/20
        spike_counts = [len(np.array(spike_times[np.where(np.logical_and(spike_times>time_ranges[i],spike_times<=time_ranges[i+1] ))[0]]*20,dtype=np.int32))
        for i in range(len(time_ranges)-1)]

        return np.var(spike_counts)/np.mean(spike_counts)  

    def cv(self,data):
        return np.std(data)/np.mean(data)
    
    def get_MI(self,return_mean=True):
        if type(self.data['Analysis']) == list:
            return np.mean([i['FI'] for i in self.data['Analysis']])
        else:
            return self.data['Analysis']['FI']
        
    #values
    def get_current_at_first_spike(self):

        firstspike_ind = self.spikeindices[0]
        return self.I[firstspike_ind]
    
    def get_ap_count(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """

        thr = self.thresholds
        thr_ind = self.thresholdindices
        spikes = self.spikeindices
        ind = ~np.isnan(thr)
        spikes = spikes[ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]
        return len(spikes)
        
    def get_firing_rate(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if return_mean:        
            return np.mean(self.data['firing_rate'])
        else:
            return self.data['firing_rate']

    def get_inst_fr(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if return_mean:        
            return np.mean(1/(np.diff(self.spikeindices*self.dt)))
        else:
            return 1/(np.diff(self.spikeindices))

    def get_time_to_first_spike(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.thresholdindices[0]*self.dt

    def get_isi(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.thresholds)
        isi =np.diff(self.spikeindices[ind]*self.dt)
        if return_mean:
            try:
                return np.mean(isi),np.median(isi),np.max(isi),np.min(isi)
            except:
                return np.nan,np.nan,np.nan,np.nan
        else:
            return np.diff(self.spikeindices[ind])*self.dt

    def get_thresholds(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.thresholds)
        if return_mean:
            return self.thresholds[ind][0], np.mean(self.thresholds[ind]),np.median(self.thresholds[ind]),np.min(self.thresholds[ind]),np.max(self.thresholds[ind])
        else:
            return self.thresholds[ind]
              
    def get_AP_width(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        thr_ind = self.thresholdindices
        thr = self.thresholds
        ind = ~np.isnan(thr_ind)
        spks = self.spikeindices[ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]
        V = self.V
        peak = 0
        width = []
        for i, j in zip(spks, thr_ind):
            try:
                # print(j,)
                spike_wf = V[int(j):int(j)+20*5]
                # plt.plot(spike_wf)
                # plt.show()
                left = V[int(j):i]
                right_ind = i-int(j)
                right = spike_wf[right_ind:]
                half_height = V[i]/2
                left_first = np.where(left <= half_height)
                right_first = np.where(right <= half_height)
                width.append(((int(i-j)+right_first[0][0]+1)-(left_first[0][-1]))/40)
            except:
                pass
        if return_mean:
            return np.mean(width),np.median(width),np.max(width),np.min(width)
        else:
            return width  

    def get_AP_peak(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_v = []
        thr_ind = self.thresholdindices
        thr = self.thresholds
        ind = ~np.isnan(thr_ind)
        spks = self.spikeindices[ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]
        V = self.V
        for i, j in zip(spks, thr_ind):
            try:
                spike_wf = V[int(j):int(j)+100]
                max_v.append(np.max(spike_wf)-V[int(j)])
            except:
                pass

        if return_mean:
            return np.mean(max_v),np.median(max_v),np.min(max_v),np.max(max_v)
        else:
            return max_v

    def get_ephys_vals(self):
        """_summary_

        Args:
            data_i (dict): _description_

        Returns:
            _type_: _description_
        """
        waveform,average_waveform,average_v = self.get_Vm(return_mean=True)
        # average_waveform = np.mean(waveform,axis=0)
        current_at_first_spike= self.get_current_at_first_spike()
        tau = self.tau
        ap_count = self.get_ap_count()
        fr = self.get_firing_rate()
        inst_fr = self.get_inst_fr()
        time_to_first_spike=self.get_time_to_first_spike()
        mean_isi,median_isi,max_isi,min_isi = self.get_isi()
        first_thr, mean_thr, median_thr, min_thr, max_thr = self.get_thresholds()
        mean_width,median_width,max_width,min_width = self.get_AP_width()
        mean_amplitude,median_amplitude,min_amplitude,max_amplitude = self.get_AP_peak()

        ephys_data = [average_waveform,
                      current_at_first_spike,
                      ap_count,
                      fr,
                      inst_fr,
                      time_to_first_spike,
                      mean_isi,
                      median_isi,
                      max_isi,
                      min_isi,
                      first_thr,
                      mean_thr, 
                      median_thr, 
                      min_thr, 
                      max_thr,
                      mean_width,
                      median_width,
                      max_width,
                      min_width,
                      mean_amplitude,
                      median_amplitude,
                      min_amplitude,
                      max_amplitude,
                      tau,
                      self.exp_name,
                      self.cond,
                      self.trialnr] #
        return ephys_data

    def get_ephys_vals_for_comparison(self):

        tau = self.tau
        isi = self.get_isi(return_mean=False)
        thresholds = self.get_thresholds(return_mean=False)
        AP_widths = self.get_AP_width(return_mean=False)
        AP_peaks = self.get_AP_peak(return_mean=False)
        MI = self.get_MI(return_mean=False)
        ephys_data =[tau,
                      isi,
                      thresholds,
                      AP_widths,
                      AP_peaks,
                      MI,
                      self.exp_name,
                      self.cond,
                      self.trialnr] #
        return ephys_data

    def get_sta(self):        

            sampling_rate = 1/20
            spks = self.spikeindices*(sampling_rate)
            V = self.V
            I = self.I

            spiketrain = neo.SpikeTrain(spks, t_stop=len(V)*(sampling_rate), units='ms')
            signal = neo.AnalogSignal(np.array([I]).T, units='pA',sampling_rate=20/ms) 
            sta_ = sta.spike_triggered_average(signal, spiketrain, (-100 * ms, 0 * ms))
            return sta_.magnitude   
    
    def get_sta_h(self):        

        sampling_rate = 1/20
        spks = self.spikeindices*(sampling_rate)
        h = self.h
        V = self.V

        spiketrain = neo.SpikeTrain(spks, t_stop=len(V)*(sampling_rate), units='ms')
        signal = neo.AnalogSignal(np.array([h]).T, units='pA',
                                    sampling_rate=20/ms) 
        sta_ = sta.spike_triggered_average(signal, spiketrain, (-100 * ms, 0 * ms))
        return sta_.magnitude   


def test_single_exp(path_files, exp_name,compute_spikes=False):

    all_ephys_with_cond = {}
    all_ephys_data = []
    
    data = loadmatInPy(path_files + exp_name + '_analyzed.mat')
    for instance in data:
        cond = instance['input_generation_settings']['condition']
        trialnr = 0#instance['input_generation_settings']['trialnr']
        ephys_obj = EphysSet_niccolo(data=instance,cond=cond,exp_name=exp_name,trialnr=trialnr,compute_spikes=compute_spikes)
        all_ephys_data.append(ephys_obj.get_ephys_vals())

    return all_ephys_data  

def return_all_ephys_dict_with_just_files(path_to_analyzed_files,just_NC=False, compute_spikes=False):
    files = os.listdir(path_to_analyzed_files)
    all_ephys_data = []
    for f in files:
            # f = 'NC_170815_aCSF_D1ago_E3_analyzed.mat'
            data = loadmatInPy(path_to_analyzed_files+f)
            for trial, instance in enumerate(data):
                try:
                    cond = instance['input_generation_settings']['condition'].lower()
                    # trialnr = instance['input_generation_settings']['trialnr']

                    exp =  f.split('.')[0]
                    if just_NC:
                        exp = return_name_date_exp_fn_NC_data(exp)
                    else:
                        exp = return_name_date_exp_fn(exp)

                    print(exp, trial, cond)
                    ephys_obj = EphysSet_niccolo(data=instance,cond=cond,exp_name=exp,trialnr=trial,run_half=False,compute_spikes=compute_spikes)
                    all_ephys_data.append(ephys_obj.get_ephys_vals())
                except:
                        print('problem with ',f[:-13],' trial ',trial)
            # break

    return all_ephys_data

def return_all_impedance(path_to_analyzed_files):
    files = os.listdir(path_to_analyzed_files)
    all_ephys_data = []
    for f in files:
            # f = 'NC_170815_aCSF_D1ago_E3_analyzed.mat'
            data = loadmatInPy(path_to_analyzed_files+f)
            for trial, instance in enumerate(data):
                # try:
                    cond = instance['input_generation_settings']['condition'].lower()
                    # trialnr = instance['input_generation_settings']['trialnr']

                    exp =  f.split('.')[0]

                    exp = return_name_date_exp_fn(exp)

                    print(exp, trial, cond)
                    ephys_obj = EphysSet(data=instance,cond=cond,exp_name=exp,trialnr=trial)
                    imp,fas = ephys_obj.get_impedence(return_mean=False)

                    # all_ephys_data.append(imp)
                    # all_ephys_data.append(exp)
                    # all_ephys_data.append(trial)
                    # all_ephys_data.append(cond)
                # except:
                #         print('problem with ',f[:-13],' trial ',trial)
            break

    return fas

def return_partitioned_data(data,partitions):
    input_settings = data['input_generation_settings']
    total_length = len(data['membrane_potential'])
    V = data['membrane_potential']
    I = data['input_current']
    h = data['hidden_state']
    thresholds = data['thresholds']
    idx = ~np.isnan(thresholds)
    thresholds = thresholds[idx]
    spks_acsf = data['spikeindices'][idx]
    threshold_idx = data['thresholdindices'][idx]
    data_partitions = []

    for i in range(partitions):
        V_ = V[(total_length//partitions)*i:(total_length//partitions)*(i+1)]
        I_ = I[(total_length//partitions)*i:(total_length//partitions)*(i+1)]
        h_ = h[(total_length//partitions)*i:(total_length//partitions)*(i+1)]
        idx_temp = np.where((spks_acsf>=(total_length//partitions)*(i))&(spks_acsf<(total_length//partitions)*(i+1)))
        spks_ = spks_acsf[idx_temp]-((i)*total_length//partitions)

        thresholds_ = thresholds[idx_temp]
        threshold_idx_ = threshold_idx[idx_temp]-((i)*total_length//partitions)

        data_idx = {'membrane_potential':V_,
                    'input_current':I_,
                    'hidden_state':h_,
                    'spikeindices':spks_,
                    'thresholds':thresholds_,
                    'thresholdindices':threshold_idx_,
                    'input_generation_settings':input_settings,
                    'firing_rate':len(spks_)*(input_settings['sampling_rate']*1000)/(total_length//partitions)
                    }
        data_partitions.append(data_idx)

    return data_partitions

def return_all_ephys_dict_with_just_files_partitioned(path_to_analyzed_files,partitions,compute_spikes):
    files = os.listdir(path_to_analyzed_files)
    all_ephys_data = []
    for f in files:
            # f = 'NC_170815_aCSF_D1ago_E3_analyzed.mat'
            try:
                data = loadmatInPy(path_to_analyzed_files+f)
                for trial, instance in enumerate(data):
                    cond = instance['input_generation_settings']['condition']
                    # trialnr = instance['input_generation_settings']['trialnr']

                    exp =  f.split('.')[0]
                    exp = return_name_date_exp_fn(exp)
                    print(exp, trial, cond)
                    instance_paritions = return_partitioned_data(instance,2)
                    all_partitions = []
                    for i in range(partitions):
                        if compute_spikes:
                            ephys_obj = EphysSet_niccolo(data=instance_paritions[i],cond=cond,exp_name=exp,trialnr=trial,run_half=False,compute_spikes=True)
                        else:
                            ephys_obj = EphysSet_niccolo(data=instance_paritions[i],cond=cond,exp_name=exp,trialnr=trial,run_half=False)

                        all_partitions.append(ephys_obj.get_ephys_vals())
                    all_ephys_data.append(all_partitions)
                # break
            except:
                print('problem with ',exp, trial, cond)
                pass
    return all_ephys_data

def return_all_waveforms():
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

    new_a = join_conditions()

    new_a_inh = new_a.groupby('tau').get_group(50)
    new_a_exc = new_a.groupby('tau').get_group(250)

    exp_name_inh = np.unique(np.array(new_a_inh['experimentname']))
    exp_name_exc = np.unique(np.array(new_a_exc['experimentname']))
    all_ephys_data_inh = []
    all_ephys_data_exc = []
    problem_cell = []
    
    count = 0
    for exp in exp_name_exc:
        count += 1
        print(count,exp)
        try:
            data = loadmatInPy(path_i + exp + '_analyzed.mat')
        except:
            data = loadmatInPy(path_i + 'Copy of ' + exp + '_analyzed.mat')
        for instance in data:
            cond = instance['input_generation_settings']['condition']
            ephys_obj = EphysSet(data=instance,cond=cond,exp_name=exp)
            waves_exc = list(np.mean(ephys_obj.get_Vm(return_mean=False),axis=0))
            waves_exc.append(ephys_obj.cond)
            waves_exc.append(ephys_obj.exp_name)
            all_ephys_data_exc.append(waves_exc)

    count = 0
    for exp in exp_name_inh:
        count += 1
        print(count,exp)
        try:
            data = loadmatInPy(path_i + exp + '_analyzed.mat')

        except:
            data = loadmatInPy(path_i + 'Copy of ' + exp + '_analyzed.mat')
        for instance in data:
            cond = instance['input_generation_settings']['condition']
            ephys_obj = EphysSet(data=instance,cond=cond,exp_name=exp)
            waves_inh = list(np.mean(ephys_obj.get_Vm(return_mean=False),axis=0))
            waves_inh.append(ephys_obj.cond)
            waves_inh.append(ephys_obj.exp_name)

            all_ephys_data_inh.append(waves_inh)

    all_ephys_with_cond['exc'] = all_ephys_data_exc
    all_ephys_with_cond['inh'] = all_ephys_data_inh
    return all_ephys_with_cond

def return_all_waveforms_DB(path):
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

    waves_all  = [] 
    files = os.listdir(path)[1:]
    for f in files:
        try:
            data = loadmatInPy(path +f)
            exp = f[:-13]
            print(exp)
            for trial,instance in enumerate(data):
                waves = []
                cond = instance['input_generation_settings']['condition']
                ephys_obj = EphysSet(data=instance,cond=cond,exp_name=exp,trialnr=trial)
                waves = list(np.mean(ephys_obj.get_Vm(return_mean=False),axis=0))
                waves.append(ephys_obj.cond)
                waves.append(ephys_obj.exp_name)
                waves.append(trial)
                waves_all.append(waves)
        except:
            print('problem with ',f[:-13])
    return waves_all

def return_all_STA_db(path,compute_spikes=False):
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

    sta_all  = [] 
    files = os.listdir(path)
    for f in files:
        try:
            data = loadmatInPy(path +f)
            exp = f[:-13]
            exp = return_name_date_exp_fn(exp)
            
            for trial,instance in enumerate(data):
                sta = []
                cond = instance['input_generation_settings']['condition']
                print(exp, trial, cond)
                ephys_obj = EphysSet_niccolo(data=instance,cond=cond,exp_name=exp,trialnr=trial,compute_spikes=compute_spikes)
                sta = list(ephys_obj.get_sta())
                sta.append(ephys_obj.cond)
                sta.append(ephys_obj.exp_name)
                sta.append(trial)
                sta_all.append(sta)
        except:
            print('problem with ',f[:-13])
    return sta_all

def return_all_STA_h_db(path):
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

    sta_all  = [] 
    files = os.listdir(path)[1:]
    for f in files:
        try:
            data = loadmatInPy(path +f)
            exp = f[:-13]
            
            for trial,instance in enumerate(data):
                sta = []
                cond = instance['input_generation_settings']['condition']
                print(exp, trial, cond)
                ephys_obj = EphysSet_niccolo(data=instance,cond=cond,exp_name=exp,trialnr=trial)
                sta = list(ephys_obj.get_sta_h())
                sta.append(ephys_obj.cond)
                sta.append(ephys_obj.exp_name)
                sta.append(trial)
                sta_all.append(sta)
        except:
            print('problem with ',f[:-13])

    return sta_all

def run_and_save(func,savepath,save=True,**args):
    
    feats = ['waveform',
         'current_at_first_spike',
         'ap_count',
         'fr',
         'inst_fr',
         'time_to_first_spike',
         'mean_isi',
         'median_isi',
         'max_isi',
         'min_isi',
         'first_thr', 
         'mean_thr', 
         'median_thr', 
         'min_thr', 
         'max_thr',
         'mean_width',
         'median_width',
         'max_width',
         'min_width',
         'mean_amplitude',
         'median_amplitude',
         'min_amplitude',
         'max_amplitude',
         'tau',
         'exp_name',
         'cond',
         'trialnr']

    if func.__name__ == 'return_all_ephys_dict_with_just_files_partitioned':
        data = func("D:/Analyzed/",2,compute_spikes=True)
        data_1 = np.array(data)[:,0]
        data_2 = np.array(data)[:,1]

        df1 = pd.DataFrame(columns=feats)
        df2 = pd.DataFrame(columns=feats)

        for i in range(len(data_1)):
            df1.loc[i,'waveform'] = np.array(data_1)[i][0]
            df1.loc[i,feats[1:]]  = np.array(data_1)[i][1:]

        for i in range(len(data_2)):
            df2.loc[i,'waveform'] = np.array(data_2)[i][0]
            df2.loc[i,feats[1:]]  = np.array(data_2)[i][1:]

        if save:
            df1.to_pickle(savepath+'Ephys_collection_all_exps_all_conds_first.pkl')
            df2.to_pickle(savepath+'Ephys_collection_all_exps_all_conds_second.pkl')
        else:
            return df1,df2
    
    elif func.__name__ == 'return_all_ephys_dict_with_just_files':
        if args['compute_spikes']==True:
            data = func("D:/Analyzed/",compute_spikes=True)
        if args['compute_spikes']==False:
            data = func("D:/Analyzed/")

        df = pd.DataFrame(columns=feats)

        for i in range(len(data)):
            df.loc[i,'waveform'] = np.array(data)[i][0]
            df.loc[i,feats[1:]]  = np.array(data)[i][1:]

        if save:
            df.to_pickle(savepath+'Ephys_collection_all_exps_all_conds.pkl')
        else:
            return df1,df2       



# %%xuan_29319_E1
data = loadmatInPy("D:/Analyzed/xuan_29-3-19_E1_analyzed.mat")
# data = return_all_ephys_dict_with_just_files("D:/Analyzed/",compute_spikes=True)
# data = return_all_ephys_dict_with_just_files_partitioned("D:/Analyzed/",2,compute_spikes=True)

# data = return_all_ephys_dict_with_just_files("D:/CurrentClamp/FN_analyzed/",just_NC=True,compute_spikes=True)

# "D:\Analyzed\NC_170821_aCSF_D1ago_E4_analyzed.mat"
# data = test_single_exp("D:/Analyzed/",'NC_170821_aCSF_D1ago_E4',compute_spikes=True)
# imps = return_all_impedance("D:/Analyzed/")
# waves = return_all_waveforms_DB("D:/Analyzed/")
# stas = return_all_STA_db("D:/Analyzed/",compute_spikes=True)
# stas = return_all_STA_h_db("D:/Analyzed/")

 #%% Fr saving all STAs
df = pd.DataFrame(columns=['sta','cond','exp_name','trial'])
for i in range(len(stas)):
    df.loc[i,'sta'] = np.array(np.hstack(stas[i])[:-3],dtype=np.float32)
    df.loc[i,['cond','exp_name','trial']] = np.hstack(stas[i])[-3:] 
df.to_pickle('D:/CurrentClamp/all_stas_hidden_spikes_computed.pkl')

# %% For saving all ephys features for clustering 

feats = ['waveform',
         'current_at_first_spike',
         'ap_count',
         'fr',
         'inst_fr',
         'time_to_first_spike',
         'mean_isi',
         'median_isi',
         'max_isi',
         'min_isi',
         'first_thr', 
         'mean_thr', 
         'median_thr', 
         'min_thr', 
         'max_thr',
         'mean_width',
         'median_width',
         'max_width',
         'min_width',
         'mean_amplitude',
         'median_amplitude',
         'min_amplitude',
         'max_amplitude',
         'tau',
         'exp_name',
         'cond',
         'trialnr']

data_1 = np.array(data)[:,0]
data_2 = np.array(data)[:,1]

df1 = pd.DataFrame(columns=feats)
df2 = pd.DataFrame(columns=feats)

for i in range(len(data_1)):
    df1.loc[i,'waveform'] = np.array(data_1)[i][0]
    df1.loc[i,feats[1:]]  = np.array(data_1)[i][1:]

for i in range(len(data_2)):
    df2.loc[i,'waveform'] = np.array(data_2)[i][0]
    df2.loc[i,feats[1:]]  = np.array(data_2)[i][1:]

df1.to_pickle('D:/FN_analysed_feat_set/Ephys_collection_all_exps_all_conds_first_spks_calculated.pkl')
df2.to_pickle('D:/FN_analysed_feat_set/Ephys_collection_all_exps_all_conds_second_spks_calculated.pkl')

#%%

feats = ['waveform',
         'current_at_first_spike',
         'ap_count',
         'fr',
         'inst_fr',
         'time_to_first_spike',
         'mean_isi',
         'median_isi',
         'max_isi',
         'min_isi',
         'first_thr', 
         'mean_thr', 
         'median_thr', 
         'min_thr', 
         'max_thr',
         'mean_width',
         'median_width',
         'max_width',
         'min_width',
         'mean_amplitude',
         'median_amplitude',
         'min_amplitude',
         'max_amplitude',
         'tau',
         'exp_name',
         'cond',
         'trialnr']
df = pd.DataFrame(columns=feats)
for i in range(len(data)):
    df.loc[i,'waveform'] = np.array(data)[i][0]
    df.loc[i,feats[1:]]  = np.array(data)[i][1:]

# df.to_pickle('D:/FN_analysed_feat_set/Ephys_collection_all_exps_all_conds_spikes_calculated_5ms.pkl')

# df.to_pickle("D:/Data For Publication/FN_files_NC.pkl")

#%% For saving essential features for significance test
feats = ['tau',
         'isi',
         'thresholds',
         'AP_widths',
         'AP_peaks',
         'MI',
         'exp_name',
         'cond',
         'trialnr']

df = pd.DataFrame(columns=feats)

for i in range(len(data)):
    df.loc[i,'tau'] = np.array(data[i][0])
    df.loc[i,'isi'] = np.array(data[i][1])
    df.loc[i,'thresholds'] = np.array(data[i][2])
    df.loc[i,'AP_widths'] = np.array(data[i][3])
    df.loc[i,'AP_peaks'] = np.array(data[i][4])
    df.loc[i,'MI'] = np.array(data[i][5])
    df.loc[i,'exp_name'] = np.array(data[i][6])
    df.loc[i,'cond'] = np.array(data[i][7])
    df.loc[i,'trialnr'] = np.array(data[i][8])
df.to_pickle('D:/FN_analysed_feat_set/val_collection_all_exps_all_conds.pkl')


#%%
# feats = ['waveforms','cond','exp_name','trial']
# # waves = np.vstack(waves)
# df = pd.DataFrame(columns=feats)
# # df['waveform'] = np.vstack(waves)[:,:-3]
# # df[['cond','exp_name','trial']] =  np.vstack(waves)[:,:-3]
# for i in range(len(waves)):
#     print(i)
#     df.loc[i,'waveforms'] = np.float32(np.array(waves)[i,:-3])
#     df.loc[i,['cond','exp_name','trial']] = np.array(waves)[i,-3:]
# df.to_pickle('D:/CurrentClamp/all_waveforms_entire.pkl')

# %%
df.to_csv('D:/CurrentClamp/all_waveforms_entire.pkl')


# %%
data = loadmatInPy("D:/CurrentClamp/FN_analyzed/170628_NC_33_FN_analyzed.mat")

spks = data[0]['spikeindices']*(1/20)
V = data[0]['membrane_potential']
I = data[0]['input_current']
sampling_rate = 1/20

# spiketrain = neo.SpikeTrain(spks, t_stop=len(V)*(sampling_rate), units='ms')
# signal = neo.AnalogSignal(np.array([I]).T, units='pA',
#                             sampling_rate=20/ms) 

# sta_ = sta.spike_triggered_average(signal, spiketrain, (-100 * ms, 0.01* ms))


# plt.plot(sta_.magnitude)
# for i in data[0]['spikeindices'][:10]:
#     plt.plot(I[data[0]['spikeindices'][i]-100*20:data[0]['spikeindices'][i]])


# %%


feats = ['impedance','exp_name','trial','cond']
imps_vals = imps[::4]
exps = imps[1::4]
trials = imps[2::4]
conds = imps[3::4]
df = pd.DataFrame(columns=feats)

for i in range(len(imps_vals)):
    print(i)
    df.loc[i,'impedance'] = imps_vals[i]
    df.loc[i,['exp_name','trial','cond']] = [exps[i],trials[i],conds[i]]

df.to_pickle('D:/CurrentClamp/Impedance.pkl')


# %%
