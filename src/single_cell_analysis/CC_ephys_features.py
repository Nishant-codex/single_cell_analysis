
import numpy as np 
import pandas as pd 
import os
from typing import Set
from numpy.lib.function_base import append 
import scipy.io as spio
from scipy.io import loadmat, savemat
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle as pkl 
from scipy.signal import find_peaks
import sys

from single_cell_analysis.utils import *
from single_cell_analysis.CC_analysis_utils import * 

path_cc_NC = 'D:/NC_CC_files/mat_analyzed/'
path_cc_NC_DB = 'D:/CurrentClamp/StepProtocol/'
all_cc = pd.read_csv("G:\My Drive\lists\Verification_list_all_CC_FN.csv")
path_cc = 'D:/Step-and-Hold Protocol (Xuan, Asli, NC, Payam)/'
df_acsf = all_cc[all_cc.drug==False]


class Ephys_CC:
    '''Ephys_CC is a class that extracts electrophysiological features from current clamp data.
    Parameters:
    --------------
    VI_data: the voltage and current data
    spikedata: the spike data
    
    Methods:
    --------------  
    get_current_at_first_spike: returns the current at the first spike
    get_AP_count: returns the number of action potentials in the last trial
    fi_curve: plots the firing rate against the input current
    get_abs_firing_rate: returns the absolute firing rate in Hz
    get_inst_firing_rate: returns the instantaneous firing rate in Hz
    get_time_to_first_spike: returns the time to the first spike in ms
    get_isi_stats: returns statistics about the inter-spike intervals
    get_threshold_stats: returns statistics about the threshold values
    get_halfwidth_stats: returns statistics about the half-width values
    get_amplitude_stats: returns statistics about the amplitude values
    get_all_ephys_vals: returns all the ephys values as a list
    
    '''
    def __init__(self,VI_data,spikedata):
        self.V_I_data = VI_data
        self.spikedata = spikedata
        self.onset = 2003  
        self.offset = 12002
 
         
    def get_current_at_first_spike(self):
        '''get_current_at_first_spike returns the current at the first spike.
        Parameters:
        --------------
        None
        Returns:
        --------------
        I_s[0]: the current at the first spike
        '''
        spike_inds = []
        # for trial in self.spikedata:
        spike_ind = 0
        spike_ind_val = 0
        for ind, each in enumerate(self.spikedata):
            if len(each['spks'])>0:
                spike_ind = ind
                spike_ind_val = each['spks'][0]
                break
        spike_inds.append({str(spike_ind):spike_ind_val})  
        # print(spike_inds)            
        I_s = []
        for ind1,ind2 in enumerate(spike_inds):

            I_s.append(self.V_I_data['I'][int(list(ind2.keys())[0])][ind2[list(ind2.keys())[0]]])
        return I_s[0]

    def get_AP_count(self):
        '''get_AP_count returns the number of action potentials in the last trial.
        Parameters:
        --------------
        None
        Returns:
        --------------
        spike_inds: the number of action potentials in the last trial
        '''
        spike_inds = []
        # for trial in self.spikedata:
        spike_inds  = len(self.spikedata[-1]['spks'])
        return spike_inds

    def fi_curve(self):
        '''fi_curve plots the firing rate against the input current.
        Parameters:
        --------------
        None
        Returns:
        --------------
        Is: the input currents
        '''
        time = self.offset-self.onset
        frs =[1000*(len(self.spikedata[i]['spks'])/(time/20)) for i in range(len(self.spikedata))]
        Is = [100*(np.mean(self.V_I_data['I'][i][self.onset:self.offset])/1e-10) for i in range(len(self.V_I_data['I']))]
        plt.scatter(Is,frs)
        plt.xlabel('I(pA)')
        plt.ylabel('Firing Rate (Hz)')

        plt.show()
        return Is
    
    def get_abs_firing_rate(self):
        '''get_abs_firing_rate returns the absolute firing rate in Hz.
        Parameters:
        --------------
        None
        Returns:
        --------------
        fr*1000: the absolute firing rate in Hz
        '''
        time = self.offset-self.onset

        fr = len(self.spikedata[-1]['spks'])/(time/20)
        return fr*1000

    def get_inst_firing_rate(self):
        '''get_inst_firing_rate returns the instantaneous firing rate in Hz.
        Parameters:
        --------------
        None
        Returns:
        --------------
        inst_fr: the instantaneous firing rate in Hz
        '''
        isi = []

        isi = np.diff(self.spikedata[-1]['spks'])/20
        inst_fr = np.mean(1/isi)                  
        return inst_fr

    def get_time_to_first_spike(self):
        '''get_time_to_first_spike returns the time to the first spike in ms.
        Parameters:
        --------------
        None
        Returns:
        --------------
        spike_inds: the time to the first spike in ms
        '''
        spike_ind = 0
        spike_ind_val = 0
        for ind, each in enumerate(self.spikedata):
            if len(each['spks'])>0:
                spike_ind = ind
                spike_ind_val = each['spks'][0]-self.onset
                break
        spike_inds = spike_ind_val/20  

        return spike_inds     

    def get_isi_stats(self):
        '''get_isi_stats returns the mean, max, min, and median inter-spike intervals (ISI) in ms.
        Parameters:
        --------------
        None
        Returns:
        --------------
        mean_isi: the mean ISI in ms
        max_isi: the maximum ISI in ms
        min_isi: the minimum ISI in ms
        median_isi: the median ISI in ms
        '''
        isi = np.diff(self.spikedata[-1]['spks'])/20
        mean_isi = np.mean(isi)
        max_isi = np.max(isi)
        min_isi = np.min(isi)
        median_isi = np.median(isi)                    
        return mean_isi,max_isi,min_isi,median_isi    

    def get_threshold_stats(self):
        '''get_threshold_stats returns the first, mean, max, min, and median threshold values.
        Parameters:
        --------------
        None
        Returns:
        --------------
        first_thrs: the first threshold value
        mean_thrs: the mean threshold value
        max_thrs: the maximum threshold value
        min_thrs: the minimum threshold value
        median_thrs: the median threshold value
        '''
        thrs = self.spikedata[-1]['thrs']
        mean_thrs = np.nanmean(thrs)
        max_thrs = np.nanmax(thrs)
        min_thrs = np.nanmin(thrs)
        median_thrs = np.nanmedian(thrs)                    
        return thrs[0], mean_thrs,max_thrs,min_thrs,median_thrs      

    def get_halfwidth_stats(self):
        '''get_halfwidth_stats returns the first, mean, max, min, and median half-width values.
        Parameters:
        --------------
        None
        Returns:
        --------------
        first_hwidth: the first half-width value
        mean_hwidth: the mean half-width value
        median_hwidth: the median half-width value
        max_hwidth: the maximum half-width value
        min_hwidth: the minimum half-width value
        '''
        hwidths = []
        for ind, step in enumerate(self.spikedata):
            if len(step['spks'])>1:
                for sp,thr in zip(step['spks'],step['thr_ind']):
                    V = self.V_I_data['V'][ind]
                    half_amp = (V[sp]+V[thr])/2
                    half_width = np.where(V[thr:thr+100]>=half_amp)[0]
                    hwidths.append(len(half_width)/20)
                break
        return hwidths[0], np.mean(hwidths),np.median(hwidths),np.max(hwidths),np.min(hwidths)   

    def get_amplitude_stats(self):
        '''get_amplitude_stats returns the first, mean, max, min, and median amplitude values.
        Parameters:
        --------------
        None
        Returns:
        --------------
        first_amp: the first amplitude value
        mean_amp: the mean amplitude value
        median_amp: the median amplitude value
        max_amp: the maximum amplitude value
        min_amp: the minimum amplitude value
        '''
        amp = []
        amp_ = []
        for ind, step in enumerate(self.spikedata):
            if len(step['spks'])>0:
                for sp in step['spks']:
                    V = self.V_I_data['V'][ind]
                    arg_ahp = sp+np.argmin(V[sp:sp+20*5])
                    arg_min = np.argmin(V[sp-20*2:sp+20*4])
                    arg_max = np.argmax(V[sp-20*2:sp+20*4])

                    # plt.plot(V[sp:sp+100])
                    # plt.scatter(np.argmin(V[sp:sp+100]),V[np.argmin(V[sp:sp+100])])
                    amp_i = arg_max - arg_min

                    amp_.append(amp_i)
                break
        amp.append(amp_)        
        return amp[0][0], np.mean(amp),np.median(amp),np.max(amp),np.min(amp)    

    def get_all_ephys_vals(self):
        '''get_all_ephys_vals returns all the ephys values as a list.
        Parameters:
        --------------
        None
        Returns:
        --------------
        List of all ephys values
        '''
        current_first_spike = self.get_current_at_first_spike()
        ap_count = self.get_AP_count()
        abs_firing_rate = self.get_abs_firing_rate()
        inst_firing_rate = self.get_inst_firing_rate()
        time_to_first_spike = self.get_time_to_first_spike()
        mean_isi,max_isi,min_isi,median_isi = self.get_isi_stats() 
        first_thrs,mean_thrs,max_thrs,min_thrs,median_thrs = self.get_threshold_stats()

        first_hwidths,mean_hwidths,median_hwidths,max_hwidths,min_hwidths = self.get_halfwidth_stats()
        first_amp,mean_amp,median_amp,max_amp,min_amp = self.get_amplitude_stats()
        return [current_first_spike,ap_count,abs_firing_rate,
                inst_firing_rate,time_to_first_spike,mean_isi,max_isi,
                min_isi,median_isi,first_thrs,mean_thrs,
                max_thrs,min_thrs,median_thrs,first_hwidths,
                mean_hwidths,median_hwidths,max_hwidths,
                min_hwidths,first_amp,mean_amp,median_amp,max_amp,min_amp]



def return_all_ephys_cc(path_cc,df_cc,already_analyzed):
    '''
    return_all_ephys_cc returns all the ephys values for the given path and dataframe.
    Parameters:
    --------------
    path_cc: the path to the cell data
    df_cc: the dataframe containing the cell information
    already_analyzed: a list of already analyzed cells
    Returns:
    --------------
    all_cc: a list of all the ephys values
    prob_cell: a list of cells that had problems
    '''
    df_cc = df_cc[df_cc.all_cc_names_and_dates==df_cc.fn_matches]
    file_cond = list(df_cc['CC_files'])
    cond = list(df_cc['condition'])
    drug = list(df_cc['drug'])
    exp_name = list(df_cc['all_cc_names_and_dates'])
    all_cc = []
    prob_cell = []
    for ind_, f in enumerate(file_cond):
        try:
            VI_data = returnVsandIs(path_cc,f) 
            spikedata = collect_singlecell_spike_data(path_cc,f,already_analyzed)
            all_cc_trial = []
            for ind,VI_data_ in enumerate(VI_data):
                # print(VI_data[VI_data_])
                ephys = Ephys_CC(VI_data[VI_data_],spikedata[ind])
                ephys_set_i = ephys.get_all_ephys_vals()
                ephys_set_i.append(cond[ind_])
                ephys_set_i.append(drug[ind_])
                ephys_set_i.append(exp_name[ind_])
                ephys_set_i.append(ind)

                all_cc_trial.append(ephys_set_i)
            all_cc.append(all_cc_trial)
        except:
            print('problem with inside '+f)
            prob_cell.append(f)
    return all_cc,prob_cell

def return_all_ephys_cc_analyzed(path_cc, already_analyzed,running_data_with_drug=False):
    '''
    return_all_ephys_cc_analyzed returns all the ephys values for the given path and already analyzed cells.
    Parameters:
    --------------
    path_cc: the path to the cell data
    already_analyzed: a list of already analyzed cells
    running_data_with_drug: a boolean indicating whether to include drug information
    Returns:
    --------------
    all_cc: a list of all the ephys values
    prob_cell: a list of cells that had problems
    '''
    files  = os.listdir(path_cc)[:-1] 
    all_cc = []
    prob_cell = []
    for ind_, f in enumerate(files):
            exp = f[:-4]
            try:
                VI_data = returnVsandIs_analyzed_DB(path_cc,f) 
                spikedata = collect_singlecell_spike_data(path_cc,f,already_analyzed)
                all_cc_trial = []
                for ind,VI_data_ in enumerate(VI_data):
                    # print(VI_data[VI_data_])
                    print(ind_,' ',exp,' ', ind)

                    wave = get_waveforms(spikedata[ind][-1]['spks'],VI_data[str(ind+1)]['V'][-1])

                    ephys = Ephys_CC(VI_data[VI_data_],spikedata[ind])

                    ephys_set_i = ephys.get_all_ephys_vals()
                    if len(set(np.isnan(ephys_set_i)))>1:
                        print(ephys_set_i)
                        pass
                    else:
                        ephys_set_i.append(ind+1)
                        if running_data_with_drug:
                            exp_name,drug_cond = standardize_exp_string_CC(exp)
                            drug_cond = drug_cond=='DRUG'
                            ephys_set_i.append(exp_name)
                            ephys_set_i.append(drug_cond)
                        else:
                            ephys_set_i.append(exp)

                        ephys_set_i.append(wave)
                    # all_cc_trial.append(ephys_set_i)
                        all_cc.append(ephys_set_i)
            except:
                print('problem with inside '+f)
                prob_cell.append(f)


    return all_cc,prob_cell

def return_waveforms(path):
    '''return_waveforms returns the waveforms for the given path.
    Parameters:
    --------------
    path: the path to the cell data
    Returns:
    --------------
    waves: a list of all the waveforms
    problem_cells: a list of cells that had problems
    '''
    waves = []
    problem_cells = []
    for i in os.listdir(path):
        # if 'NC' in i:
            exp = i[:-4]
            # if exp == '180809_ME_5_CC':
            try:
                spiks = collect_singlecell_spike_data(path,i,True)
                VI = returnVsandIs_analyzed_DB(path,i)
                
                for trial in range(len(spiks)):
                    wave = []
                    print(exp, trial)
                    wave.append(get_waveforms(spiks[trial][-1]['spks'],VI[str(trial+1)]['V'][-1]))
                    wave.append(trial+1)
                    wave.append(exp)
                    waves.append(wave) 
            
            except:
                print(exp, 'is fauly')
                problem_cells.append(exp)
            # break

    return waves,problem_cells



 