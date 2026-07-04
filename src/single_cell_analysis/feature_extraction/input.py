# for saving input features for clustering
import pandas as pd
import numpy as np
from single_cell_analysis.feature_extraction.input import return_all_input_data_with_just_files

data = return_all_input_data_with_just_files("D:/Analyzed/",just_NC=False,compute_spikes=True)

feats = ['mean_I',
            'var_I',
            'exp_name',
            'trialnr',
            'cond',]

df = pd.DataFrame(columns=feats)
for i in range(len(data)):
    df.loc[i,'mean_I'] = np.array(data)[i][0]
    df.loc[i,feats[1:]]  = np.array(data)[i][1:]
df.to_pickle("D:/Data For Publication/I_data.pkl")