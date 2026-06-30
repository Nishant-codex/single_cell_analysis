import pandas as pd 
import argparse 
import os 
from single_cell_analysis.FN_ephys_features import * 








parser = argparse.ArgumentParser(description='Please enter the data directory and saving directory')
parser.add_argument('-input', required=True, help='location of the data')
# parser.add_argument('-o', '--output', default='output.txt', help='Output file.')
parser.add_argument('-o', '--output', required=True, help='extracted feature storage')




imps = return_all_impedance("D:/Analyzed/")
