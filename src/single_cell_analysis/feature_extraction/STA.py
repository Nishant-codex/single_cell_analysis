import argparse
import os

import pandas as pd

from single_cell_analysis.FN_ephys_features import *


def parse_args():
    parser = argparse.ArgumentParser(description='Extract STA features from ephys data.')
    parser.add_argument('--input', dest='input_path', required=True, help='location of the input data')
    parser.add_argument('--output', dest='output_path', required=True, help='directory where extracted STAs will be saved')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    stas = return_all_STA_db(args.input_path, compute_spikes=True)
    stas = return_all_STA_h_db(args.input_path)

    df = pd.DataFrame(columns=['sta', 'baseline', 'peak_distance', 'cond', 'exp_name', 'trial'])
    for i in range(len(stas)):
        df.loc[i, 'sta'] = np.array(np.hstack(stas[i])[:-3], dtype=np.float32)
        df.loc[i, ['baseline', 'peak_distance', 'cond', 'exp_name', 'trial']] = np.hstack(stas[i])[-5:]

    output_file = os.path.join(args.output_path, 'all_stas_normed_input.pkl')
    df.to_pickle(output_file)


if __name__ == '__main__':
    main()



