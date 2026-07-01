import argparse
import os

import numpy as np
import pandas as pd

from single_cell_analysis.CC_ephys_features import return_all_ephys_cc_analyzed, return_waveforms


def parse_args():
    parser = argparse.ArgumentParser(description='Extract CC ephys or waveform features from user-provided data.')
    parser.add_argument('--input', dest='input_path', required=True, help='location of the CC data directory')
    parser.add_argument('--output', dest='output_path', required=True, help='directory where extracted features will be saved')
    parser.add_argument('--mode', choices=['ephys', 'waveform', 'both'], default='both',
                        help='whether to extract ephys features, waveform features, or both')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    if args.mode in ['ephys', 'both']:
        data, cell_with_issues = return_all_ephys_cc_analyzed(args.input_path, True, running_data_with_drug=True)

        features = ['current_at_first_spike', 'ap_count', 'fr',
                    'inst_fr', 'time_to_first_spike', 'mean_isi',
                    'max_isi', 'min_isi', 'median_isi', 'first_thr',
                    'mean_thr', 'max_thr', 'min_thr', 'median_thr',
                    'first_width', 'mean_width', 'median_width',
                    'max_width', 'min_width', 'first_amplitude',
                    'mean_amplitude', 'median_amplitude', 'max_amplitude',
                    'min_amplitude', 'trialnr', 'exp_name', 'drug', 'waveforms']

        df = pd.DataFrame(data, columns=features)
        output_file = os.path.join(args.output_path, 'CC_files_all_experimenters_all_conditions.pkl')
        df.to_pickle(output_file)
        print(f'Saved ephys features to {output_file}')

    if args.mode in ['waveform', 'both']:
        waves, prob_cells = return_waveforms(args.input_path)

        feats = ['waveforms', 'exp_name', 'trial']
        df_waves = pd.DataFrame(columns=feats)

        if waves:
            waveforms = np.array(waves)[:, 0]
            for i in range(len(waves)):
                df_waves.loc[i, 'waveforms'] = waveforms[i]
                df_waves.loc[i, ['trial', 'exp_name']] = np.array(waves)[i][1:]

        df_waves = df_waves.dropna(axis=0)
        output_file = os.path.join(args.output_path, 'CC_waveforms.pkl')
        df_waves.to_pickle(output_file)
        print(f'Saved waveform features to {output_file}')


if __name__ == '__main__':
    main()

