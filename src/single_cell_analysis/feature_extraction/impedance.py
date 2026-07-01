import argparse
import os

import pandas as pd

from single_cell_analysis.FN_ephys_features import *


def parse_args():
    parser = argparse.ArgumentParser(description='Extract impedance features from analyzed ephys files.')
    parser.add_argument('--input', dest='input_path', required=True, help='location of the analyzed data files')
    parser.add_argument('--output', dest='output_path', required=True, help='directory where impedance features will be saved')
    return parser.parse_args()


def return_all_impedance(path_to_analyzed_files):
    files = os.listdir(path_to_analyzed_files)
    all_ephys_data = []
    for f in files:
        data = loadmatInPy(os.path.join(path_to_analyzed_files, f))
        for trial, instance in enumerate(data):
            cond = instance['input_generation_settings']['condition'].lower()
            exp = f.split('.')[0]
            exp = return_name_date_exp_fn(exp)

            print(exp, trial, cond)
            ephys_obj = EphysSet(data=instance, cond=cond, exp_name=exp, trialnr=trial)
            imp, fas = ephys_obj.get_impedence(return_mean=False)

            all_ephys_data.append(imp)
            all_ephys_data.append(exp)
            all_ephys_data.append(trial)
            all_ephys_data.append(cond)
        break

    return fas


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    imps = return_all_impedance(args.input_path)

    feats = ['impedance', 'exp_name', 'trial', 'cond']
    imps_vals = imps[::4]
    exps = imps[1::4]
    trials = imps[2::4]
    conds = imps[3::4]
    df = pd.DataFrame(columns=feats)

    for i in range(len(imps_vals)):
        print(i)
        df.loc[i, 'impedance'] = imps_vals[i]
        df.loc[i, ['exp_name', 'trial', 'cond']] = [exps[i], trials[i], conds[i]]

    output_file = os.path.join(args.output_path, 'all_impedance_values.pkl')
    df.to_pickle(output_file)


if __name__ == '__main__':
    main()
