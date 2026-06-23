'''
Created on Wed Mar  1 12:06:26 2023
This script contains functions to analyze single cell data
By Nishant Joshi	

'''

import sys
from single_cell_analysis.src.single_cell_analysis.utils import *

def separate_acsf_and_drug(data_list:list) -> list:
	"""seperates and returns acsf trials from the drug trials in the cell object

	Args:
		data_list (list): a list containing all the acsf and drug trial for a cell experiment
	Returns:
		_type_: _description_
	"""

	acsf = []
	drug = []
	for i in data_list:
		if i['input_generation_settings']['condition'] in ['ACSF','aCSF']:
			acsf.append(i)
		else:
			drug.append(i)
	return acsf, drug


def collect_drug_and_acsf(path_files:str, condition:list,seperate_exc_inh:bool=False) -> list:
	""" collects the acsf and drug trials for the specified condition.
	can be used for analyzing cells together. 

	Args:
		path_files (str): the path to the analyzed files 
		condition (list): condition(s) to be accumulated

	Returns:
		list: all acsf trials 
		list: all drug trials
	"""
	if seperate_exc_inh:
		acsf_all_inh = []
		acsf_all_exc = []
		drug_all_inh = []
		drug_all_exc = []
		assert path_files[-1] =='/'
		df_inh,df_exc = join_conditions(list_cond = condition,exc_inh_sep=True)
		files_inh = df_inh['filename']
		files_exc = df_exc['filename']

		for i in files_inh:
			print(i)
			data = loadmatInPy(filename=path_files+i)
			acsf,drug = separate_acsf_and_drug(data)
			for acsf_i in acsf:
				acsf_all_inh.append(acsf_i)
			for drug_i in drug:
				drug_all_inh.append(drug_i)
		for i in files_exc:
			print(i)
			data = loadmatInPy(filename=path_files+i)
			acsf,drug = separate_acsf_and_drug(data)
			for acsf_i in acsf:
				acsf_all_exc.append(acsf_i)
			for drug_i in drug:
				drug_all_exc.append(drug_i)
		return acsf_all_inh, drug_all_inh,acsf_all_exc, drug_all_exc	
	else:		
		acsf_all = []
		drug_all = []
		assert path_files[-1] =='/'
		df = join_conditions(list_cond = condition)
		files = df['filename']
		for i in files:
			print(i)
			data = loadmatInPy(filename=path_files+i)
			acsf,drug = separate_acsf_and_drug(data)
			for acsf_i in acsf:
				acsf_all.append(acsf_i)
			for drug_i in drug:
				drug_all.append(drug_i)
		return [acsf_all, drug_all]

