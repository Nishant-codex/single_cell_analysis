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
from sklearn.linear_model import LinearRegression

# parameters{1} = 'Current at first spike'; @
# parameters{3} = 'AP Count - Start'; @
# parameters{5} = 'Absolute Firing Rate - Start'; @
# parameters{7} = 'Inst Firing Rate - Start'; @
# parameters{9} = 'AP Latency First - Start'; @
# parameters{11} = 'AP Latency Last - Start'; @
# parameters{13} = 'AP Latency Window - Start';
# parameters{15} = 'AP Latency Compression - Start';
# parameters{17} = 'ISI Min - Start'; @
# parameters{19} = 'ISI Max - Start'; @
# parameters{21} = 'ISI Mean - Start'; @
# parameters{23} = 'ISI Median - Start'; @
# parameters{25} = 'ISI Adapt Rate - Start'; 
# parameters{27} = 'AP Threshold First - Start'; @
# parameters{29} = 'AP Threshold Last - Start'; @ 
# parameters{31} = 'AP Threshold Min - Start'; @ 
# parameters{33} = 'AP Threshold Max - Start'; @
# parameters{35} = 'AP Threshold Mean - Start'; @
# parameters{37} = 'AP Threshold Median - Start'; @
# parameters{39} = 'AP Threshold AdaptationRate - Start';
# parameters{41} = 'AP HalfWidth First - Start'; @
# parameters{43} = 'AP HalfWidth Last - Start'; @
# parameters{45} = 'AP HalfWidth Min - Start'; @
# parameters{47} = 'AP HalfWidth Max - Start'; @
# parameters{49} = 'AP HalfWidth Mean - Start'; @
# parameters{51} = 'AP HalfWidth Median - Start'; @
# parameters{53} = 'AP HalfWidth AdaptationRate - Start';
# parameters{55} = 'AP Amplitude First - Start'; @
# parameters{57} = 'AP Amplitude Last - Start'; @
# parameters{59} = 'AP Amplitude Min - Start'; @
# parameters{61} = 'AP Amplitude Max - Start'; @
# parameters{63} = 'AP Amplitude Mean - Start'; @
# parameters{65} = 'AP Amplitude Median - Start'; @
# parameters{67} = 'AP Amplitude AdaptationRate - Start';



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
            Vm.append(V[int(i)-70:int(i)+100])
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
        imp = overdracht_wytse(0.01, I_acsf, V_acsf, 20001, 20001, 1)
        if return_mean:
            return np.mean(imp)
        else:
            return imp
    
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

    def __init__(self,data,cond,exp_name,trialnr):

        self.data = data
        self.cond = cond
        self.exp_name = exp_name
        self.trialnr = trialnr
        self.V = self.data['membrane_potential']
        self.I = self.data['input_current']
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

    def return_quant_divided_by_time(self,divisions,quant):
        total_duration = len(self.data['membrane_potential'])/20
        quantity = quant
        time_ranges = np.arange(0,total_duration+1,total_duration//divisions)
        spike_times = self.data['spikeindices']/20
        time_divided_spiketimes = [np.array(spike_times[np.where(np.logical_and(spike_times>time_ranges[i],spike_times<=time_ranges[i+1] ))[0]]*20,dtype=np.int32)
        for i in range(len(time_ranges)-1)]
        last = 0
        vals_divided_by_time = []
        for i in time_divided_spiketimes:
            vals_divided_by_time.append(quantity[last:last+len(i)])
            last = len(i)
        return vals_divided_by_time  

    def fano_factor(self,divisions):
        total_duration = len(self.data['membrane_potential'])/20
        time_ranges = np.arange(0,total_duration+1,total_duration//divisions)
        spike_times = self.data['spikeindices']/20
        spike_counts = [len(np.array(spike_times[np.where(np.logical_and(spike_times>time_ranges[i],spike_times<=time_ranges[i+1] ))[0]]*20,dtype=np.int32))
        for i in range(len(time_ranges)-1)]

        return np.var(spike_counts)/np.mean(spike_counts)  

    def cv(self,data):
        return np.std(data)/np.mean(data)
    
    #values
    def get_current_at_first_spike(self):

        firstspike_ind = self.data['spikeindices'][0]
        return self.I[firstspike_ind]
    
    def get_ap_count(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """

        thr = self.data['thresholds']
        thr_ind = self.data['thresholdindices']
        spikes = self.data['spikeindices']
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
            return np.mean(1/(np.diff(self.data['spikeindices']*self.dt)))
        else:
            return 1/(np.diff(self.data['spikeindices']))

    def get_time_to_first_spike(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.data['thresholdindices'][0]*self.dt

    def get_isi(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        isi =np.diff(self.data['spikeindices'][ind]*self.dt)
        if return_mean:
            try:
                return np.mean(isi),np.median(isi),np.max(isi),np.min(isi)
            except:
                return np.nan,np.nan,np.nan,np.nan
        else:
            return np.diff(self.data['spikeindices'][ind])*self.dt

    def get_thresholds(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        if return_mean:
            return self.data['thresholds'][ind][0], np.mean(self.data['thresholds'][ind]),np.median(self.data['thresholds'][ind]),np.min(self.data['thresholds'][ind]),np.max(self.data['thresholds'][ind])
        else:
            return self.data['thresholds'][ind]
              
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
        thr_ind = self.data['thresholdindices']
        thr = self.data['thresholds']
        ind = ~np.isnan(thr_ind)
        spks = self.data['spikeindices'][ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]
        V = self.data['membrane_potential']
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

        current_at_first_spike= self.get_current_at_first_spike()
        ap_count = self.get_ap_count()
        fr = self.get_firing_rate()
        inst_fr = self.get_inst_fr()
        time_to_first_spike=self.get_time_to_first_spike()
        mean_isi,median_isi,max_isi,min_isi = self.get_isi()
        first_thr, mean_thr, median_thr, min_thr, max_thr = self.get_thresholds()
        mean_width,median_width,max_width,min_width = self.get_AP_width()
        mean_amplitude,median_amplitude,min_amplitude,max_amplitude = self.get_AP_peak()

        ephys_data =      [current_at_first_spike,
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
                           self.exp_name,
                           self.cond,
                           self.trialnr] #
        return ephys_data


def test_single_exp(exp_name):

    all_ephys_with_cond = {}
    all_ephys_data = []
    path_i = 'C:/Users/Nishant Joshi/Google Drive/Analyzed/'
    try:
        data = loadmatInPy(path_i + exp_name + '_analyzed.mat')
    except:
        data = loadmatInPy(path_i + 'Copy of ' + exp_name + '_analyzed.mat')
    for instance in data:
        cond = instance['input_generation_settings']['condition']
        trialnr = instance['input_generation_settings']['trialnr']
        ephys_obj = EphysSet_niccolo(data=instance,cond=cond,exp_name=exp_name,trialnr=trialnr)
        all_ephys_data.append(ephys_obj.get_ephys_vals())

    return all_ephys_data  

def return_all_ephys_dict_old(cond:list, experimenter:str=None)->dict:
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

def load_single_cell_test(exp):
    path_i = 'C:/Users/Nishant Joshi/Google Drive/Analyzed/'
    try:
        data = loadmatInPy(path_i + exp + '_analyzed.mat')
    except:
        data = loadmatInPy(path_i + 'Copy of ' + exp + '_analyzed.mat')
    return data        

def return_all_ephys_dict():
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
            trialnr = instance['input_generation_settings']['trialnr']
            ephys_obj = EphysSet_niccolo(data=instance,cond=cond,exp_name=exp,trialnr=trialnr)
            all_ephys_data_exc.append(ephys_obj.get_ephys_vals())

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
            trialnr = instance['input_generation_settings']['trialnr']
            ephys_obj = EphysSet_niccolo(data=instance,cond=cond,exp_name=exp,trialnr=trialnr)
            all_ephys_data_inh.append(ephys_obj.get_ephys_vals())

    all_ephys_with_cond['exc'] = all_ephys_data_exc
    all_ephys_with_cond['inh'] = all_ephys_data_inh
    all_ephys_with_cond['cond'] = cond
    return all_ephys_with_cond

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


# %%

