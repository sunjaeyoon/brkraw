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
from brkraw.lib.reference import WORDTYPE, BYTEORDER

from raw2frame import *
from recoFunctions import *

import os
import numpy as np
import matplotlib.pyplot as plt
import time

## --------------------------------------------------------------------------##

MainDir = "E:\\TimHo\\CCM_data_12022022\\20221202_Price_CCM\Price-Delaney-CCM\\Price-Delaney-CCM_20220519_PDGFb29_30_Cdh5319_320_20220913_CCM_Scr_E1_P1\\20220913_134021_Price_Delaney_CCM_20220913_CCM_Screening_1_9"
ExpNum = 7

rawdata = br.load(os.path.join(MainDir))


# Raw data processing for single job
fid_binary = rawdata.get_fid(ExpNum)
acqp = rawdata.get_acqp(ExpNum)
meth = rawdata.get_method(ExpNum)
with open(os.path.join(MainDir, str(ExpNum),'pdata','1','reco'),'r') as f:
    reco = Parameter(f.read().split('\n'))


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


#recoFunctions.reco_phase_corr_pi(kdata[:,:,:,:,0,0,0], reco,0)


start_time = time.time()
image = brkraw_Reco(kdata, reco, meth, recoparts = 'all')
print("--- %s seconds ---" % (time.time() - start_time))

"""
N1 = 12
for i in range(N1):
    plt.figure()
    for j in range(4):
        plt.subplot(2,2,j+1)
        plt.imshow(np.squeeze(np.angle(image[:,:,i,0,j,0,0])))
"""
print(image[63:65,63:65,63,0,3,0])
plt.figure()
plt.plot(np.abs(image[64,64,64,0,0,:]))