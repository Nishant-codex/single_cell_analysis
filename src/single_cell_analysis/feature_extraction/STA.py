import pandas as pd 
import argparse 
import os 
from single_cell_analysis.FN_ephys_features import * 

parser = argparse.ArgumentParser(description='Please enter the data directory and saving directory')
parser.add_argument('-input', required=True, help='location of the data')
# parser.add_argument('-o', '--output', default='output.txt', help='Output file.')
parser.add_argument('-o', '--output', required=True, help='extracted feature storage')




# For saving all STAs
stas = return_all_STA_db(parser.input,compute_spikes=True)
stas = return_all_STA_h_db(parser.input)

df = pd.DataFrame(columns=['sta','baseline','peak_distance','cond','exp_name','trial'])
for i in range(len(stas)):
    df.loc[i,'sta'] = np.array(np.hstack(stas[i])[:-3],dtype=np.float32)
    df.loc[i,['baseline','peak_distance','cond','exp_name','trial']] = np.hstack(stas[i])[-5:] 
df.to_pickle(parser.output+'all_stas_normed_input.pkl')



