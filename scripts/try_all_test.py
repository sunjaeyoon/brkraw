# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 19:45:24 2023

@author: 
"""


import os

import sys
sys.path.append("..")

from brkraw.lib.parser import Parameter
from brkraw.lib.pvobj import PvDatasetDir
from brkraw.lib.utils import get_value, set_value
import brkraw as br

from raw2frame import *
from recoFunctions import *

import os
import numpy as np
import matplotlib.pyplot as plt
import time

import traceback


def get_files(path):
    file_all = []
    for (root,dirs,files) in os.walk(path, topdown=True):                
        #print(len(files))
        for file in files:
            file_path = os.path.join(root, file)
            file_all.append(file_path)
    
    return file_all

def get_dirs(path):
  
    file_all = []
    for (root,dirs,files) in os.walk(path, topdown=True):
        for d in dirs:
            file_all.append(os.path.join(root,d))
    return file_all



def run_tthrough(MainDir):
    rawdata = br.load(os.path.join(MainDir))


    for ExpNum in range(1,50):
        try:
            # Raw data processing for single job
            fid_binary = rawdata.get_fid(ExpNum)
            acqp = rawdata.get_acqp(ExpNum)
            meth = rawdata.get_method(ExpNum)
            with open(os.path.join(MainDir, str(ExpNum),'pdata','1','reco'),'r') as f:
                reco = Parameter(f.read().split('\n'))
        
            print(get_value(acqp, 'ACQ_protocol_name'))
            print(os.path.join(MainDir,str(ExpNum)) )
        
            # test functions
            start_time = time.time()
            raw_sort = readBrukerRaw(fid_binary, acqp, meth)
            print("--- %s seconds ---" % (time.time() - start_time))
        
            start_time = time.time()
            frame = convertRawToFrame(raw_sort, acqp, meth)
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Frame\t",frame.shape)
        
            start_time = time.time()
            kdata = convertFrameToCKData(frame, acqp, meth)
            print("--- %s seconds ---" % (time.time() - start_time))
            
        
            start_time = time.time()
            image = brkraw_Reco(kdata, reco, meth, recoparts = ['quadrature', 'phase_rotate', 'FT', 'phase_corr_pi']  )
            print("--- %s seconds ---" % (time.time() - start_time))
            
            for i in range(1):
                plt.figure()
                plt.imshow(np.squeeze(np.abs(image[:,:,0,0,0,i,0])))
                plt.title(get_value(acqp, 'ACQ_protocol_name')+' '+str(ExpNum))
                plt.show()
            
            print("Kdata\t",kdata.shape)
            print("Image\t",image.shape)
            print()
       
        except Exception as e:
            #print(type(e))
            #print(sys.exc_info()[2])
            
            if 'KeyError' not in traceback.format_exc():
                print(traceback.format_exc())
                break
                

#path     = "E:\TimHo\\Tor_data_01102023"
path     = "E:\TimHo\\CCM_data_12022022"
#path     = "E:\TimHo\CCM_data_12022022"
#path_out = "E:\TimHo\CCM_data_sorted"
folder = [i[:-23] for i in get_files(path) if "ScanProgram" in i]


for i in folder: 
    run_tthrough(i)