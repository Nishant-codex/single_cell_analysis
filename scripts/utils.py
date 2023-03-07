
import sys 
import pickle
import numpy as np 
from numpy.lib.function_base import append 
import scipy.io as spio
from scipy.io import loadmat, savemat
import importlib.util
import sklearn
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from scipy.sparse import data 
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from plotnine.data import mtcars
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import sys


'''
idea behind this script is to compile functions that are more frequantly used and are more general 

1. loadmatInPy -> loading matrices 

2. join_conditions -> loading data frame containing info about the condition

'''
def join_conditions(path:str = None, list_cond:list = None,exc_inh_sep:bool = False)->pd.DataFrame:
	"""joins condition information together. 
	To be used for analyzing the metadata and loading files for group analysis

	Args:
		path (str, optional): _description_. Defaults to None.
		list_cond (list, optional): _description_. Defaults to None.
		exc_inh_sep(bool, bool): _description_. Defaults to False.
	Returns:
		new_df (pd.DataFrame): a sub pandas DataFrame object for drug condition specified in the input
		
		if exc_inh_sep is true:
		new_df_inh (pd.DataFrame): a sub pandas DataFrame object for drug condition specified in the input. All inhibitory cells
		new_df_exc (pd.DataFrame): a sub pandas DataFrame object for drug condition specified in the input. All excitatory cells

	"""

	if path!=None:
		path = path
	else:
		path = 'C:/Users/Nishant Joshi/Google Drive/lists/all_files_new.csv'
	table = pd.read_csv(path)
	cond= np.sort(np.unique(table['condition']))
	temp = []
	
	if len(list_cond)>1:
		for i in list_cond:
			temp.append(table.groupby('condition').get_group(i))
			new_df = temp[0]
			for j in range(1,len(temp)):
				new_df = pd.concat([new_df,temp[j]])
	else:
		new_df = table.groupby('condition').get_group(list_cond[0])

	if exc_inh_sep:
		inh = new_df.groupby('tau').get_group(50)
		exc = new_df.groupby('tau').get_group(250)
		return inh, exc
	else:
		return new_df



def loadmatInPy(filename:str)->dict:
	"""	
	this function should be called instead of direct spio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	
	Args:
		filename (str): full path to the matlab data file 
	Returns:
		dict: a dictionary with all the analyzed variables included
	"""

	def _check_keys(dict):
		'''
		checks if entries in dictionary are mat-objects. If yes
		todict is called to change them to nested dictionaries
		'''
		for key in dict:
			if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
				dict[key] = _todict(dict[key])
		return dict        

	def _todict(matobj):
		
		'''
		A recursive function which constructs nested dictionaries from matobjects
		'''
		dict = {}
		try:
			for strg in matobj._fieldnames:
				elem = matobj.__dict__[strg]
				if isinstance(elem, spio.matlab.mio5_params.mat_struct):
					dict[strg] = _todict(elem)
				elif strg =='Analysis':
					temp = []
					for i in elem:

						temp.append(_todict(i))
					dict['Analysis'] = temp
				else:
					if isinstance(elem,np.ndarray):
						dict[strg] = elem.tolist()
					else:
						dict[strg] = elem
		except:
			for strg in matobj.keys():
				elem = matobj[strg]
				if isinstance(elem, spio.matlab.mio5_params.mat_struct):
					dict[strg] = _todict(elem)
				elif strg =='Analysis':
					temp = []
					for i in elem:
						temp.append(_todict(i))
					dict['Analysis'] = temp
				else:
					if isinstance(elem,np.ndarray):
						dict[strg] = elem.tolist()
					else:
						dict[strg] = elem
		return dict
	
	if ('analyzed' in filename )and ('_CC_' not in filename):
		Data = []
		data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)['Data']
		try:
			for i in range(len(data)):
				data_ = data[i]
				data_ = _todict(data_) 
				Data.append(_check_keys(data_))
			return Data
		except:
			data_ = data
			data_ = _todict(data_) 
			Data.append(_check_keys(data_))
			return Data
	elif ('analyzed' in filename) and ('_CC_' in filename):
		data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
		data = _todict(data)          
	else:
		data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)['Data']
		data = _todict(data) 
	return _check_keys(data)




