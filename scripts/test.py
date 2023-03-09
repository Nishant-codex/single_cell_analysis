#%%
import sys
sys.path.append("C:/Users/Nishant Joshi/Downloads/Old_code/repo/single_cell_analysis/scripts")
import os 
import time 
import utils
from utils import loadmatInPy
import time
import pickle
import json

#%%

all_files = os.listdir('D:/Analyzed')
# for file in all_files:
t1 = time.time()
datapy = loadmatInPy('D:/Analyzed/'+all_files[0])
t2 = time.time()
print(t2-t1)
# pickle.dump(datapy,open('D:/analyzed_python/'+all_files[0].split('.')[0]+'.py','wb'))

#%%

t1 = time.time()
with open('D:/analyzed_python/'+all_files[0].split('.')[0]+'.py', 'rb') as f:
    x = pickle.load(f)
t2 = time.time()
print(t2-t1)

