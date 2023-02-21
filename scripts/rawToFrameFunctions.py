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

from recoFunctions import *
from raw2frame import *

import os
import numpy as np
import matplotlib.pyplot as plt


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
raw_sort = readBrukerRaw(fid_binary, acqp, meth)
frame = convertRawToFrame(raw_sort, acqp, meth)
kdata = convertFrameToCKData(frame, acqp, meth)


reco_result = np.zeros(kdata.shape).astype(kdata.dtype)
# Other stuff
RECO_ft_mode = get_value(reco, 'RECO_ft_mode')

if '360' in meth.headers['title'.upper()]:
    reco_ft_mode_new = []
    
    for i in RECO_ft_mode:
        if i == 'COMPLEX_FT' or i == 'COMPLEX_FFT':
            reco_ft_mode_new.append('COMPLEX_IFT')
        else:
            reco_ft_mode_new.append('COMPLEX_FT')
            
    reco = set_value(reco, 'RECO_ft_mode', reco_ft_mode_new)
    RECO_ft_mode = get_value(reco, 'RECO_ft_mode')


#def brkraw_Reco(kdata, Reco, recopart = 'all')

# Adapt FT convention to acquisition version.
N1, N2, N3, N4, N5, N6, N7 = kdata.shape
recopart = 'all'

dims = kdata.shape[0:4];
for i in range(4):
    if dims[i]>1:
        dimnumber = (i+1);
    
NINR=kdata.shape[5]*kdata.shape[6]

if recopart == 'all':
    recopart = ['quadrature', 'phase_rotate', 'zero_filling', 'FT', 'phase_corr_pi', 
              'cutoff',  'scale_phase_channels', 'sumOfSquares', 'transposition']

# all transposition the same?
same_transposition = True
for i in get_value(reco, 'RECO_transposition'):
    if i != get_value(reco, 'RECO_transposition')[0]:
        same_transposition = False


map_index= np.reshape( np.arange(0,kdata.shape[5]*kdata.shape[6]), (kdata.shape[6], kdata.shape[5]) )

# if 'quadrature' in recopart:
#     for NR in range(N7):
#         for NI in range(N6):
#             for chan in range(N5):
#                 reco_qopts(kdata[:,:,:,:,chan,NI,NR], reco, NI*NR)
    
# if 'phase_rotate' in recopart:
#     for NR in range(N7):
#         for NI in range(N6):
#             for chan in range(N5):
#                 reco_phase_rot(kdata[:,:,:,:,chan,NI,NR], reco, NI*NR)
  
# if 'zero_filling' in recopart:
#     zero_fill()

if 'FT' in recopart:
    for NR in range(N7):
        for NI in range(N6):
            for chan in range(N5):
                reco_result[:,:,:,:,chan,NI,NR] = reco_ft(kdata[:,:,:,:,chan,NI,NR], reco)


# if 'phase_corr_pi' in recopart:
#     reco_phase_corr_pi()
 
# if 'cutoff' in recopart:  
#     reco_cutoff()
    
# if 'scale_phase_channels' in recopart: 
#     reco_scale_phase_channels()
    
if 'sumOfSquares' in recopart:
    for NR in range(N7):
        for NI in range(N6):
            #print(reco_result[:,:,:,:,:1,NI,NR].shape)
            print(NR+1, NI+1)
            reco_result[:,:,:,:,:1,NI,NR] = reco_sumofsquares(reco_result[:,:,:,:,:,NI,NR], reco)
    reco_result = reco_result[:,:,:,:,:1,:,:]

#if 'transposition' in recopart:
#    transposition()

# from numpy.fft import fft2, ifft2,fftshift, ifftshift, fftn

# kdata = fftn(kdata[:,:,:,0,0,0,0])

# for i in range(N3):
#     plt.figure()
    
#     for j in range(N6):
#         img = np.squeeze(reco_result[:,:,i,0,0,j,0])
#         #img = fft2(img)
#         plt.subplot(2,4,j+1)
#         plt.imshow(np.abs(img))
#         plt.title(j) 
# y_max = np.abs(np.max(reco_result))
# plt.figure()
# for i in range(N1):
#     for j in range(N2):
        
#         if np.max(np.abs(reco_result[i,j,64,0,0,:,0])) > 2500: 
#             plt.plot(np.abs(reco_result[i,j,64,0,0,:,0]))
#             plt.ylim(ymax = y_max, ymin = 0)