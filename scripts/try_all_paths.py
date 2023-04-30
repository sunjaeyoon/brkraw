# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:02:55 2023

@author: Timothy
"""

import sys
sys.path.append("..")
import os

import traceback
from subprocess import check_output


import time
import random


import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from brkraw.lib.parser import Parameter
from brkraw.lib.pvobj import PvDatasetDir
from brkraw.lib.utils import get_value, set_value
import brkraw as br

from raw2frame import *
from recoFunctions import *


def get_files(path):
    file_all = []
    for (root,dirs,files) in os.walk(path, topdown=True):                
        #print(len(files))
        for file in files:
            file_path = os.path.join(root, file)
            file_all.append(file_path)
    
    return file_all

def print_names(MainDir):
    rawdata = br.load(os.path.join(MainDir))
    with open('files.txt', 'a') as f2:
        for ExpNum in range(1,100):
             
                try:
                    # Raw data processing for single job
                    fid_binary = rawdata.get_fid(ExpNum)
                    acqp = rawdata.get_acqp(ExpNum)
                    meth = rawdata.get_method(ExpNum)
                    with open(os.path.join(MainDir, str(ExpNum),'pdata','1','reco'),'r') as f:
                        reco = Parameter(f.read().split('\n'))
                
                    #print(get_value(acqp, 'ACQ_protocol_name'))
                    print(os.path.join(MainDir,str(ExpNum)) )
                    path = os.path.join(MainDir,str(ExpNum))
                    f2.write(path+'\n')
                    #command = f'matlab -nodisplay -r \"cd(\'E:\TimHo\SWI_bruker\pvtools\custom_functions\'); make_dataset({path}, {ExpNum});exit'
                    #print(command)
                    #check_output(command, shell = True)
                except Exception as e:
                    #print(type(e))
                    #print(sys.exc_info()[2])
                    
                    if 'KeyError' not in traceback.format_exc():
                        print(traceback.format_exc())
        
path     = "E:\TimHo\\Tor_data_01102023"
#path     = "E:\TimHo\\CCM_data_12022022"
#path     = "E:\TimHo\CCM_data_12022022"
#path_out = "E:\TimHo\CCM_data_sorted"
folders = [i[:-23] for i in get_files(path) if "ScanProgram" in i]

for i in folders:
    print_names(i)


