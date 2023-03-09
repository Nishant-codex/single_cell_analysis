# %%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import pandas as pd

from utils import *
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import numpy.matlib
# %%


def makespectrumt(trace, timelen, adcrate):
    # time is the timelength of the sub evaluation 1/time is the freq resolution!
    NFFT = timelen * adcrate

    NFFT2 = int(np.floor(NFFT/2))
    NFFT = int(2 * NFFT2)
    # (0:(1/timelen):(NFFT2-1)/timelen)'
    fas = np.arange(0, NFFT2-1, (1/timelen))
    fwdw = signal.windows.hamming(NFFT)
    start = 0
    last = len(trace)-NFFT
    count = 0
    pwr = 0
    while start < last:
        endpoint = start + NFFT
        spoor = trace[start:endpoint]-np.mean(trace[start:endpoint])
        spect = fft(fwdw*spoor, NFFT)/NFFT
        if count == 0:
            pwr = 2*np.abs(spect[1:NFFT2])
        else:
            pwr = pwr + 2*np.abs(spect[1:NFFT2])
        start = start + NFFT2
        count += 1
    pwr = pwr/count
    return pwr, fas


def overdracht_wytse(par1, spoordac, spooradc, dacrate, adcrate, par2=None):

    time = par1
    if par2 is None:
        nf = 0
    else:
        nf = par2
    [pwrdac, fasdac] = makespectrumt(spoordac, time, dacrate)
    [pwradc, fasadc] = makespectrumt(spooradc, time, adcrate)
    lendac = len(pwrdac)
    lenadc = len(pwradc)
    if lendac > lenadc:
        pwrdac = pwrdac[1:lenadc]
        fas = fasdac[1:lenadc]
    elif lenadc > lendac:
        pwradc = pwradc[1:lendac]
        fas = fasadc[1:lendac]
    else:
        fas = fasdac

    if nf > 1:
        pwrdac = signal.filtfilt(np.matlib.repmat(1/nf, nf, 1), 1, pwrdac)
        pwradc = signal.filtfilt(np.matlib.repmat(1/nf, nf, 1), 1, pwradc)
    ovr = pwradc/pwrdac

    y = ovr
    return y


def spectrum_wytse(spoor, rate, par1, par2=None):
    time = par1
    if par2 == None:
        nf = 0
    else:
        nf = par2

    [pwr, fas] = makespectrumt(spoor, time, rate)

    if nf > 1:
        pwr = signal.filtfilt(np.matlib.repmat(1/nf, nf, 1), 1, pwr)
    x = pwr
    y = fas
    return (x, y)


def return_stiched_spike_train(data, subthreshold=False, plot=False):

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
    tailored_spikes = np.ndarray.flatten(
        np.array(empty_cell))  # {cat(2, empty_cell{:})}
    zero_spikes = np.zeros(np.size(V))
    zero_spikes[tailored_spikes] = True

    if subthreshold == True:
        zero_spikes = ~zero_spikes

    return zero_spikes, V_, I_


def get_impedence(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    I_acsf = data['input_current']
    V_acsf = data['membrane_potential']
    spk_acsf, V_acsf, I_acsf = return_stiched_spike_train(data)
    imp = overdracht_wytse(0.01, I_acsf, V_acsf, 20001, 20001, 1)
    return imp


def normalizeBytheFirstValue(data_matrix):
    npmat = np.array(data_matrix)
    firstVal = npmat[0, :]
    remaining = npmat[1:, :]
    remaining = remaining/firstVal
    all = np.zeros_like(npmat)
    all[0, :] = firstVal
    all[1:, :] = remaining
    return remaining


def plot_imp_condition(cond, exp_name=None, remove_spikes=True, plot=False, path_list=None, path_analyzed_data=None):

    path_list = 'C:/Users/Nishant Joshi/Downloads/Old_code/lists/all_files_new.csv'

    path_analyzed_data = 'C:/Users/Nishant Joshi/Google Drive/Analyzed/'
    cond_i = cond
    new_a = join_conditions(list_cond=cond_i)
    new_a_inh = new_a.groupby('tau').get_group(50)
    new_a_exc = new_a.groupby('tau').get_group(250)
    if exp_name is not None:
        new_a_inh = new_a.groupby('experimenter').get_group(exp_name)
        new_a_exc = new_a.groupby('experimenter').get_group(exp_name)

    exp_name_inh = np.array(new_a_inh['experimentname'])
    trials_inh = np.array(new_a_inh['trialnr'])
    exp_name_exc = np.array(new_a_exc['experimentname'])
    trials_exc = np.array(new_a_exc['trialnr'])

    imp_inh = []
    imp_inh_acsf = []
    for i, j in zip(exp_name_inh, trials_inh):
        try:
            try:
                data = loadmatInPy(path_analyzed_data+i+'_analyzed.mat')
            except:
                data = loadmatInPy(path_analyzed_data +
                                   'Copy of '+i+'_analyzed.mat')
            # try:
            for i in data:
                # this portion saves all the acsf recordings for a particular cell
                if i['input_generation_settings']['condition'] in ['ACSF', 'aCSF']:
                    I_acsf = i['input_current']
                    V_acsf = i['membrane_potential']
                    if remove_spikes:
                        spk_acsf, V_acsf, I_acsf = return_stiched_spike_train(
                            i)
                    Imp_acsf = overdracht_wytse(
                        0.01, I_acsf, V_acsf, 20001, 20001, 1)
                    imp_inh_acsf.append(Imp_acsf)

            # here on, we save all the conditions
            data = data[j-1]
            print(data['input_generation_settings']['condition'])
            I_inh = data['input_current']
            V_inh = data['membrane_potential']
            if remove_spikes:
                spk, V_inh, I_inh = return_stiched_spike_train(data)

            Imp = overdracht_wytse(0.01, I_inh, V_inh, 20001, 20001, 1)
            imp_inh.append(Imp)
            if plot:
                fig, ax = plt.subplots(1, 2, figsize=[12, 6])
                data_norm = Imp/Imp[0]
                ax[0].plot(np.array(Imp).T)
                ax[0].set_yscale('log')
                ax[0].set_xlabel('F(Hz)')
                ax[0].set_ylabel('impedance(mOhm')
                ax[0].set_title('Inh impedance Dopamine')

                data_norm_acsf = Imp_acsf/Imp_acsf[0]
                ax[1].plot(np.array(data_norm_acsf).T)
                ax[1].set_yscale('log')
                ax[1].set_xlabel('F(Hz)')
                ax[1].set_ylabel('impedance(mOhm')
                ax[1].set_title('Inh impedance acsf')
                plt.show()

        except:
            print(i)
    imp_exc = []
    imp_exc_acsf = []

    for i, j in zip(exp_name_exc, trials_exc):
        try:
            try:
                data = loadmatInPy(path_analyzed_data+i+'_analyzed.mat')
            except:
                data = loadmatInPy(path_analyzed_data +
                                   'Copy of '+i+'_analyzed.mat')
            # try:
            for i in data:
                if i['input_generation_settings']['condition'] in ['ACSF', 'aCSF']:
                    I_acsf = i['input_current']
                    V_acsf = i['membrane_potential']
                    if remove_spikes:
                        spk_acsf, V_acsf, I_acsf = return_stiched_spike_train(
                            i)
                    Imp_acsf = overdracht_wytse(
                        0.01, I_acsf, V_acsf, 20001, 20001, 1)
                    imp_exc_acsf.append(Imp_acsf)

            data = data[j-1]
            print(data['input_generation_settings']['condition'])
            I_exc = data['input_current']
            V_exc = data['membrane_potential']
            if remove_spikes:
                spk, V_exc, I_exc = return_stiched_spike_train(data)
            Imp = overdracht_wytse(0.01, I_exc, V_exc, 20001, 20001, 1)
            imp_exc.append(Imp)
            if plot:
                fig, ax = plt.subplots(1, 2, figsize=[12, 6])
                data_dop_exc = Imp/Imp[0]
                ax[0].plot(np.array(data_dop_exc).T)
                # ax[0].set_xscale('log')
                ax[0].set_yscale('log')
                ax[0].set_xlabel('F(Hz)')
                ax[0].set_ylabel('impedance(mOhm')
                ax[0].set_title('Inh impedance Dopamine')

                data_exc_acsf = Imp_acsf/Imp_acsf[0]
                ax[1].plot(np.array(data_exc_acsf).T)
                # ax[1].set_xscale('log')
                ax[1].set_yscale('log')
                ax[1].set_xlabel('F(Hz)')
                ax[1].set_ylabel('impedance(mOhm')
                ax[1].set_title('Inh impedance acsf')
                plt.show()
        except:
            print(i)

    return imp_exc, imp_inh, imp_exc_acsf, imp_inh_acsf


# %%
