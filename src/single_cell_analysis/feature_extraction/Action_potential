import pandas as pd 
import argparse 
import os 
from single_cell_analysis.FN_ephys_features import * 

parser = argparse.ArgumentParser(description='Please enter the data directory and saving directory')
parser.add_argument('-input', required=True, help='location of the data')
# parser.add_argument('-o', '--output', default='output.txt', help='Output file.')
parser.add_argument('-o', '--output', required=True, help='extracted feature storage')




data = return_all_input_data_with_just_files(parser.input,just_NC=False,compute_spikes=True)

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
        'FI',
        'tau',
        'exp_name',
        'cond',
        'trialnr']
df = pd.DataFrame(columns=feats)
for i in range(len(data)):
    df.loc[i,'waveform'] = np.array(data)[i][0]
    df.loc[i,feats[1:]]  = np.array(data)[i][1:]

df.to_pickle(parser.output+'Ephys_collection_all_exps_all_conds_spikes_calculated_5ms_with_MI.pkl')
