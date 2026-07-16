
""" 
Created on Wed Mar  1 12:06:26 2023
By Nishant Joshi
This script contains utility functions for analyzing neuromodulator effects on single cells
"""

import pandas as pd 
import seaborn as sns 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def plot_FI_vs_FR(data_acsf,data_drug,ax=None, plot=True, c = 'blue',x_lim = None, y_lim = None, save=False, savepath = None):
    ''' Plots the change in FI vs change in FR for a given drug condition.
    
    Parameters:
    --------------
    data_acsf: dataframe containing the acsf condition data
    data_drug: dataframe containing the drug condition data
    ax: matplotlib axis object to plot on
    plot: boolean, whether to plot the data or not
    c: color of the points
    x_lim: tuple, limits for the x-axis
    y_lim: tuple, limits for the y-axis
    save: boolean, whether to save the plot or not
    savepath: string, path to save the plot
    
    Returns:
    --------------
    df_temp: dataframe containing the change in FI and FR for each cell     
    '''
    ind_na = np.array(data_drug['FI'].isna())
    c_labels = []
    data_acsf = data_acsf[~ind_na]
    data_drug = data_drug[~ind_na]

    for i in data_acsf.ei_labels:
        if i==1.0:
            c_labels.append('Exc')
        else:
            c_labels.append('Inh')

    c = c_labels
    delta_fi = (np.array(data_drug['FI']) - np.array(data_acsf['FI']))/np.array(data_acsf['FI'])
    delta_fr = (np.array(data_drug['fr']) - np.array(data_acsf['fr']))/np.array(data_acsf['fr'])
    df_temp = pd.DataFrame({'$\Delta$f/f':np.float32(delta_fr),
                                '$\Delta$FI/FI':np.float32(delta_fi),
                                'color':c})

    df_temp = df_temp[(df_temp['$\Delta$FI/FI']>=y_lim[0]) &(df_temp['$\Delta$FI/FI']<=y_lim[1])]
    df_temp = df_temp[(df_temp['$\Delta$f/f']>=x_lim[0]) &(df_temp['$\Delta$f/f']<=x_lim[1])]

    if plot:
        if ax is None:

            g = sns.JointGrid(data=df_temp, x="$\Delta$f/f", y="$\Delta$FI/FI",hue='color',palette=['blue','red'])
            g.plot(sns.scatterplot, sns.violinplot)
            g.ax_joint.axhline(y=0, color='gray', linestyle='--')  # Horizontal line at y=16
            g.ax_joint.axvline(x=0, color='gray', linestyle='--')  # Horizontal line at y=16

            if x_lim !=None and y_lim !=None: 
                g.ax_joint.set_xlim(x_lim)
                g.ax_joint.set_ylim(y_lim)
            # plt.legend(['inh','exc'])

            plt.xlabel('$\Delta$fr/fr')
            plt.ylabel('$\Delta$FI/FI')

            if save:
                plt.savefig(savepath,dpi=200)
            plt.show()
        else:
            ax.scatter(delta_fr[~ind_na], delta_fi[~ind_na],c=c) 

    return df_temp

def binarize_EI_labels(labels, e_vals):
    ''' Binarizes the EI labels based on the given excitatory values.
    Parameters:
    --------------
    labels: list of EI labels
    e_vals: list of excitatory values to be considered as 1, all others will be 0
    Returns:    
    --------------
    temp_labels: list of binarized EI labels
    '''
    temp_labels = labels
    for idx,vals in enumerate(labels):
        if vals in e_vals:
            temp_labels[idx] =1
        else:
            temp_labels[idx] =0 

    return temp_labels

def return_correct_names_for_neuromods(df):
    ''' Returns the correct names for the neuromodulator conditions in the dataframe.
    Parameters:
    --------------
    df: dataframe containing the neuromodulator conditions
    Returns:
    --------------
    neuromod_list: dataframe with corrected neuromodulator condition names
    '''
    neuromod_list = df.replace({'d1ago':'d1',
                                     'dop':'dopamine',
                                     'dopa':'dopamine',
                                     'm1-ag':'m1-ago',
                                     'm1-ant+ago':'m1-ago+ant',
                                     'agoanta':'m1-ago+ant'})
    return neuromod_list

def neumericalize_neurmods(df):
    ''' Returns the numerical values for the neuromodulator conditions in the dataframe.
    Parameters:
    --------------
    df: dataframe containing the neuromodulator conditions
    Returns:
    --------------
    numeric_list: list of numerical values for the neuromodulator conditions
    '''
    numeric_list = df.cond.replace({'acsf'     :0,
                    'm1-ago'    :1,
                    'm1-ant'    :2,
                    'm1-ago+ant':3,
                    'dopamine'  :4,
                    'd1'        :5,
                    'acsf_bic'  :6,
                    'd2'        :7,
                    'sag'       :8,
                    'cirazoline':9,
                    'agoanta'   :10})
    return numeric_list

def exponential_smoothing(data, alpha):
    """
    Implements Simple Exponential Smoothing from scratch.
    
    Parameters:
        data (list or numpy array): The time series data to smooth.
        alpha (float): The smoothing factor (0 < alpha < 1).
    
    Returns:
        numpy array: Smoothed values.
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]  # Initialize with the first data point
    
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    
    return smoothed

def return_paired_t_test(data_frame):
    ''' Returns the paired t-test results for the given dataframe.
    Parameters:
    --------------
    data_frame: dataframe containing the data for the t-test
    Returns:
    --------------
    None
    '''
    print(data_frame.cond.unique())
    t_statistic, p_value = stats.ttest_rel(data_frame[data_frame.cond=='acsf']['norm_peak_distance'], data_frame[data_frame.cond!='acsf']['norm_peak_distance'])
    print('norm_peak_distance', t_statistic, p_value)

    t_statistic, p_value = stats.ttest_rel(data_frame[data_frame.cond=='acsf']['decay_time'], data_frame[data_frame.cond!='acsf']['decay_time'])
    print('decay_time', t_statistic, p_value)

def rise_time(data):
    ''' Returns the rise time and peak value for the given data.
    Parameters:
    --------------
    data: numpy array containing the data
    Returns:
    --------------
    decay_time: float, the rise time of the data
    max_val: float, the peak value of the data  
    '''
    # data = data
    argmax = np.argmax(data)
    data_mod = data[argmax:]
    max_val = np.max(data_mod)
    min_val = np.min(data_mod)
    
    val_10 = min_val + 0.1*(max_val-min_val) 
    val_90 = min_val + 0.9*(max_val-min_val)     
    arg_10 = np.where(data_mod<val_10)[0][0]
    arg_90 = np.where(data_mod<val_90)[0][0]
    decay_time = (arg_10-arg_90)/20
    return decay_time, max_val

def return_peak_and_decay(df):
    ''' Returns the peak and decay values for the given dataframe.
    Parameters:
    --------------
    df: dataframe containing the data
    Returns:
    --------------
    df: dataframe with added peak and decay columns
    '''
    peak_vals_drug = []
    decay_vals_drug = []

    for y_drug in df['sta'].to_numpy():
        # try:
            y_smooth_drug =exponential_smoothing(np.flip(y_drug),0.2)  

            decay_drug, peak_drug = rise_time(y_smooth_drug)
            peak_vals_drug.append(peak_drug)
            decay_vals_drug.append(decay_drug)
    
            # df['peak'] = peak_vals_drug
        # except:
        #     pass
    df['decay_time'] = decay_vals_drug
    return df

def return_acsf_and_drug(df,cond,joint=False,remove_duplicates=True,sample=None):
    ''' Returns the acsf and drug dataframes for the given condition.
    Parameters: 
    --------------
    df: dataframe containing the data   
    cond: string, the condition to filter the dataframe
    joint: boolean, whether to return a joint dataframe or separate dataframes
    remove_duplicates: boolean, whether to remove duplicate experiments or not
    sample: int, number of samples to return from each dataframe
    Returns:
    --------------  
    df_acsf: dataframe containing the acsf condition data
    df_drug: dataframe containing the drug condition data
    '''
    exps = list(set(df[df.cond.isin(cond)]['exp_name']))
    df_new = df[df.exp_name.isin(exps)]
    df_acsf = df_new[df_new.cond == 'acsf']
    df_drug = df_new[df_new.cond == cond[0]]
    if remove_duplicates:
        common_exps = set(df_acsf.exp_name) & set(df_drug.exp_name)
        df_acsf = df_acsf.drop_duplicates('exp_name')
        df_acsf = df_acsf[df_acsf.exp_name.isin(common_exps)]
        df_drug = df_drug.drop_duplicates('exp_name')
        df_drug = df_drug[df_drug.exp_name.isin(common_exps)]

    if joint:
        df_acsf.reset_index(inplace=True)
        df_acsf = df_acsf.drop(columns='index')

        df_drug.reset_index(inplace=True)
        df_drug = df_drug.drop(columns='index')

        return pd.concat([df_acsf,df_drug])
    else:
        if sample!=None:    
            df_acsf = df_acsf.sample(n=sample)
            df_drug = df_drug.sample(n=sample)
        return df_acsf, df_drug
    
def heterogeniety_for_drug(df,cond,cols=None):
    ''' Returns the heterogeneity for the given drug condition.
    Parameters:
    --------------
    df: dataframe containing the data
    cond: string, the drug condition to analyze
    cols: list of columns to consider for heterogeneity calculation
    Returns:
    --------------
    cosine_mat: numpy array containing the cosine similarity matrix
    '''
    wave_drug = df[df.cond==cond]
    wave_acsf = df[df.cond=='acsf']
    cosine_mat = np.zeros((2,2))

    if cols!=None:
        cosine_mat[0,0] = np.mean(cosine_similarity(normalize(np.vstack(wave_drug[cols].to_numpy())),normalize(np.vstack(wave_drug[cols].to_numpy()))))
        cosine_mat[1,0] = np.mean(cosine_similarity(normalize(np.vstack(wave_drug[cols].to_numpy())),normalize(np.vstack(wave_acsf[cols].to_numpy()))))
        cosine_mat[0,1] = np.mean(cosine_similarity(normalize(np.vstack(wave_acsf[cols].to_numpy())),normalize(np.vstack(wave_drug[cols].to_numpy()))))
        cosine_mat[1,1] = np.mean(cosine_similarity(normalize(np.vstack(wave_acsf[cols].to_numpy())),normalize(np.vstack(wave_acsf[cols].to_numpy()))))
    else:
        cosine_mat[0,0] = np.mean(cosine_similarity(normalize(np.vstack(wave_drug['waveform'].to_numpy())),normalize(np.vstack(wave_drug['waveform'].to_numpy()))))
        cosine_mat[1,0] = np.mean(cosine_similarity(normalize(np.vstack(wave_drug['waveform'].to_numpy())),normalize(np.vstack(wave_acsf['waveform'].to_numpy()))))
        cosine_mat[0,1] = np.mean(cosine_similarity(normalize(np.vstack(wave_acsf['waveform'].to_numpy())),normalize(np.vstack(wave_drug['waveform'].to_numpy()))))
        cosine_mat[1,1] = np.mean(cosine_similarity(normalize(np.vstack(wave_acsf['waveform'].to_numpy())),normalize(np.vstack(wave_acsf['waveform'].to_numpy()))))


    sns.heatmap(cosine_mat,vmax=1,vmin=-1,annot=True,cmap = 'Spectral',annot_kws={'fontsize':14})    