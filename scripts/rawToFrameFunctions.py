# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:04:26 2023

@author:
"""

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

## --------------------------------------------------------------------------##

MainDir = "E:\\TimHo\\CCM_data_12022022\\20221202_Price_CCM\Price-Delaney-CCM\\Price-Delaney-CCM_20220519_PDGFb29_30_Cdh5319_320_20220913_CCM_Scr_E1_P1\\20220913_134021_Price_Delaney_CCM_20220913_CCM_Screening_1_9"
#MainDir = 'E:\\TimHo\\DOWNLOADS\\20221219_Wilson-Tim-PulseDev_Wilson-Tim-PulseDev_SPACE_E3_P1\\20221219_151737_Wilson_Tim_PulseDev_20221219_SPACE_1_4'
#MainDir = 'E:\\TimHo\\DOWNLOADS\\20221205_VFA_Wilson-Tim-PulseDev_VariableFlipAngl_E13_P1\\20221205_223925_20221205_VFA_VariableFlipAngle_1_1'
#MainDir = 'E:\TimHo\\Tor_data_01102023\\20230126_Price-Breza\\Price-Breza_Tor_Glioma_20230123_ComboEx_E23_P1\\20230123_173251_Price_Breza_20230123_ComboExp13_ctrlImmune_d10SizeMatching_1_24'
#MainDir = 'E:\\TimHo\\Tor_data_01102023\\20221207_Price-Breza\\Price-Breza_Tor_Glioma_20221205_T2FLAIR_E15_P1\\20221205_091304_Price_Breza_20221205_T2FLAIR_sham_Exp1_1_11\\'
ExpNum = 1

rawdata = br.load(os.path.join(MainDir))


#for ExpNum in range(16,17):
    #try:
# Raw data processing for single job
fid_binary = rawdata.get_fid(ExpNum)
acqp = rawdata.get_acqp(ExpNum)
meth = rawdata.get_method(ExpNum)
with open(os.path.join(MainDir, str(ExpNum),'pdata','1','reco'),'r') as f:
    reco = Parameter(f.read().split('\n'))

print(get_value(acqp, 'ACQ_protocol_name'), ExpNum)

# test functions
start_time = time.time()
raw_sort = readBrukerRaw(fid_binary, acqp, meth)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
frame = convertRawToFrame(raw_sort, acqp, meth)
print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
kdata = convertFrameToCKData(frame, acqp, meth)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
stuff = ['quadrature', 'phase_rotate', 'FT', 'phase_corr_pi']
image = brkraw_Reco(kdata, reco, meth, recoparts = 'all') #stuff)
print("--- %s seconds ---" % (time.time() - start_time))


N1 = 3
for i in range(N1):
    plt.figure()
    plt.imshow(np.squeeze(np.abs(image[:,:,0,0,0,i,0])))
    plt.title(get_value(acqp, 'ACQ_scanl_name'))
    plt.show()

    #except Exception as e:
    #    print(e)
    #    pass
            
# if get_value(acqp, 'ACQ_protocol_name') == 'T2star_map_MGE':
#     print(image[63:65,63:65,63,0,3,0])
#     plt.figure()
#     plt.plot(np.abs(image[64,64,64,0,0,:]))