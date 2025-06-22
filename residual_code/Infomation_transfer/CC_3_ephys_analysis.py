
from CC_analysis_utils import *



def max_firing_freq(path_cc,filename):
    spk_data = collect_singlecell_spike_data(path_cc,filename)
    if spk_data == 'faulty':
        return None
    else:
        raw = returnVsandIs(path_cc,filename)
        onset = 1017
        offset = 11002
        firing_rate = []
        for trial in range(len(raw)):
            t = raw[str(trial+1)]['tV'][9]            
            spikes_t =  spk_data[trial][9]['spks']
            firing_rate.append(len(spikes_t)/(t[offset]-t[onset]))
        return firing_rate

def ap_half_width(path_cc,filename):
    spk_data = collect_singlecell_spike_data(path_cc,filename)
    if spk_data == 'faulty':
        return None
    else:
        trials = len(spk_data)
        t_unit = 5.e-05
        half_width = []
        for trial in range(trials):
            for data in spk_data[trial]:
                if len(data['spks'])>1:
                    spikes = data['spks']
                    thrs = data['thr_ind']
                    half_width.append(np.mean(spikes-thrs)*t_unit)   
                    break
        return half_width

def AHP_amp(path_cc,filename):
    spk_data = collect_singlecell_spike_data(path_cc,filename)
    if spk_data == 'faulty':
        return None
    else:  
        raw = returnVsandIs(path_cc,filename)
        trials = len(spk_data) 
        t_unit = 5.e-05
        ahp = []
        for trial in range(trials):
            ahp_temp = []
            V = raw[str(trial+1)]['V']
            
            for i, data in enumerate(spk_data[trial]) :
                if len(data['spks'])>1:
                    spikes = data['spks']
                    V_inst = V[i]
                    for j in range(len(spikes)-1):
                        ahp_temp.append(np.min(V_inst[spikes[j]:spikes[j+1]])) 
                    break      
            ahp.append(np.mean(ahp_temp))
        return ahp


def collect_data_cc(path_cc, list_dir, condition, exceptions=None):

    max_freq_acsf = []
    ap_half_width_acsf = []
    ahp_amp_acsf = []

    max_freq_drug = []
    ap_half_width_drug = []
    ahp_amp_drug = []

    path_cc = path_cc
    df = pd.read_excel(list_dir)
    exp_drug = np.array(df[df['condition']==condition]['CC_files'][df['drug']==True])
    exp_acsf = np.array(df[df['condition']==condition]['CC_files'][df['drug']==False])
    for exc in exp_acsf:
        print(exc)
        # if exceptions != None:
        #     if exc in exceptions :
        #         pass
        # else:
        max_freq_acsf_t = max_firing_freq(path_cc,exc)
        if max_freq_acsf_t != None and len(max_freq_acsf_t)>0:
            max_freq_acsf.append(max_freq_acsf_t)

        ap_half_width_acsf_t = ap_half_width(path_cc,exc)
        if ap_half_width_acsf_t != None and len(ap_half_width_acsf_t)>0:
            ap_half_width_acsf.append(ap_half_width_acsf_t)

        ahp_amp_acsf_t = AHP_amp(path_cc,exc)
        if ahp_amp_acsf_t != None and len(ahp_amp_acsf_t)>0:
            ahp_amp_acsf.append(ahp_amp_acsf_t)

    for exc in exp_drug:
        print(exc)
        # if exceptions != None:
        #     if exc in exceptions :
        #         pass
        # else:
        max_freq_drug_t = max_firing_freq(path_cc,exc)
        if max_freq_drug_t != None:
            max_freq_drug.append(max_freq_drug_t)

        ap_half_width_drug_t = ap_half_width(path_cc,exc)
        if ap_half_width_drug_t != None:
            ap_half_width_drug.append(ap_half_width_drug_t)

        ahp_amp_drug_t = AHP_amp(path_cc,exc)
        if ahp_amp_drug_t != None:
            ahp_amp_drug.append(ahp_amp_drug_t)
        
    return max_freq_acsf, ap_half_width_acsf, ahp_amp_acsf, max_freq_drug, ap_half_width_drug, ahp_amp_drug 
