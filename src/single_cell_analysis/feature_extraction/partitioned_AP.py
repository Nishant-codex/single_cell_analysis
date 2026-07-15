    # For saving all ephys features for clustering 

import argparse
import numpy as np
import pandas as pd
import os   
from single_cell_analysis.FN_ephys_features import return_ephys_with_partition

def parse_args():
    parser = argparse.ArgumentParser(description='Extract action potential features from ephys data.')
    parser.add_argument('--input', dest='input_path', required=True, help='location of the input data')
    parser.add_argument('--output', dest='output_path', required=True, help='directory where extracted features will be saved')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    data = return_ephys_with_partition(args.input_path,2,compute_spikes=True)
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

    df1.to_pickle(os.path.join(args.output_path, 'Ephys_collection_all_exps_all_conds_first_spks_calculated.pkl'))
    df2.to_pickle(os.path.join(args.output_path, 'Ephys_collection_all_exps_all_conds_second_spks_calculated.pkl'))


if __name__ == '__main__':
    main()
