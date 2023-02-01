# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:04:26 2023

@author: Timothy
"""

import sys
sys.path.append("..")

from brkraw.lib.parser import Parameter
from brkraw.lib.pvobj import PvDatasetDir
from brkraw.lib.utils import get_value, set_value
import brkraw as br
from brkraw.lib.reference import WORDTYPE, BYTEORDER

from recoFunctions import *

import os
import numpy as np
import matplotlib.pyplot as plt


def readBrukerRaw(fid_binary, acqp, meth):
    dt_code = np.dtype('float64') if get_value(acqp, 'ACQ_ScanPipeJobSettings')[0][1] == 'STORE_64bit_float' else np.dtype('int32') 
    fid = np.frombuffer(fid_binary, dt_code)
    
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    ACQ_size = get_value(acqp, 'ACQ_jobs')[0][0]
    numDataHighDim = np.prod(ACQ_size)
    numSelectedRecievers = get_value(acqp, 'ACQ_ReceiverSelectPerChan').count('Yes')
    nRecs = numSelectedRecievers
    
    jobScanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    dim1 = int(len(fid)/(jobScanSize*nRecs))
    
    X = np.reshape(fid, [jobScanSize, nRecs, dim1], order='F')
    X = np.transpose(X, (1, 0, 2))

    X = X[:,0::2,:] + 1j*X[:,1::2,:]

    return X


def convertRawToFrame(data, acqp, meth):
    # Turn Raw into frame
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    ACQ_phase_factor    = get_value(acqp,'ACQ_phase_factor')
    ACQ_obj_order       = get_value(acqp, 'ACQ_obj_order')
    ACQ_dim             = get_value(acqp, 'ACQ_dim')
    numSelectedReceivers=data.shape[0]
    
    acqSizes = np.zeros(ACQ_dim)
    
    scanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    acqSizes[0] = scanSize
    
    isSpatialDim = [i == 'Spatial' for i in get_value(acqp, 'ACQ_dim_desc')]
    spatialDims = sum(isSpatialDim)
    
    if isSpatialDim[0]:
        if spatialDims == 3:
            acqSizes[1:] = get_value(meth, 'PVM_EncMatrix')[1:]
        elif spatialDims == 2:
            acqSizes[1]  = get_value(meth, 'PVM_EncMatrix')[1]
    numDataHighDim=np.prod(acqSizes[1:])
    
    if np.iscomplexobj(data):
        scanSize = int(acqSizes[0]/2)
    else:
        scanSize = acqSizes[0]
    
    # Resort
    if ACQ_dim>1:
        data = data.reshape(
            int(numSelectedReceivers), int(scanSize), int(ACQ_phase_factor), 
            int(NI), int(numDataHighDim/ACQ_phase_factor), int(NR), order='F')
        data = np.transpose(data, (1, 2, 4, 0, 3, 5)) # => scansize, ACQ_phase_factor, numDataHighDim/ACQ_phase_factor, numSelectedReceivers, NI, NR
        data=  data.reshape( int(scanSize), int(numDataHighDim), int(numSelectedReceivers), int(NI), int(NR) , order='F') 
    
        if NI != len(ACQ_obj_order):
            raise 'Size of ACQ_obj_order is not equal to NI: Data from unsupported method?'
        
        data[:,:,:,ACQ_obj_order,:]=data
    else:
        data = np.reshape(data,(numSelectedReceivers, scanSize,1,NI,NR), order='F')
        data = np.transpose(data, (1, 2, 0, 3, 4))
    
    return data


def convertFrameToCKData(frame, acqp, meth):
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    ACQ_phase_factor    = get_value(acqp,'ACQ_phase_factor')
    ACQ_obj_order       = get_value(acqp, 'ACQ_obj_order')
    ACQ_dim             = get_value(acqp, 'ACQ_dim')
    numSelectedReceivers= frame.shape[2]
    
    acqSizes = np.zeros(ACQ_dim)
    
    scanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    acqSizes[0] = scanSize
    ACQ_size = acqSizes
    
    isSpatialDim = [i == 'Spatial' for i in get_value(acqp, 'ACQ_dim_desc')]
    spatialDims = sum(isSpatialDim)
    
    if isSpatialDim[0]:
        if spatialDims == 3:
            acqSizes[1:] = get_value(meth, 'PVM_EncMatrix')[1:]
        elif spatialDims == 2:
            acqSizes[1]  = get_value(meth, 'PVM_EncMatrix')[1]
    numDataHighDim=np.prod(acqSizes[1:])
    
    if np.iscomplexobj(frame):
        scanSize = int(acqSizes[0]/2)
    else:
        scanSize = acqSizes[0]
    
    if ACQ_dim==3:
       
        PVM_EncSteps2=get_value(meth, 'PVM_EncSteps2')
        assert PVM_EncSteps2 != None
           
    PVM_Matrix = get_value(meth, 'PVM_Matrix')
    PVM_EncSteps1 = get_value(meth,'PVM_EncSteps1')
    
    PVM_AntiAlias = get_value(meth, 'PVM_AntiAlias')
    if PVM_AntiAlias == None:
        # No anti-aliasing available.
        PVM_AntiAlias = np.ones((ACQ_dim))
     
    PVM_EncZf=get_value(meth, 'PVM_EncZf')
    
    if PVM_EncZf == None:
        # No zero-filling/interpolation available.
        PVM_EncZf = np.ones((ACQ_dim))
    
    # Resort
    # use also method-parameters (encoding group)
    
    frameData = frame
    
    # MGE with alternating k-space readout: Reverse every second
    # scan. 
    
    if get_value(meth, 'EchoAcqMode') != None and get_value(meth,'EchoAcqMode') == 'allEchoes':
        print('Okay, Ill figure this out later')
        #frameData(:,:,:,2:2:end,:)=flipdim(data(:,:,:,2:2:end,:),1)
    
    
    # Calculate size of Cartesian k-space
    # Step 1: Anti-Aliasing
    ckSize = np.round(np.array(PVM_AntiAlias)*np.array(PVM_Matrix))
    # Step 2: Interpolation 

    reduceZf = 2*np.floor( (ckSize - ckSize/np.array(PVM_EncZf))/2 )
    ckSize = ckSize - reduceZf
    
    # # index of central k-space point (+1 for 1-based indexing in MATLAB) 
    ckCenterIndex = np.floor(ckSize/2 + 0.25) + 1

    readStartIndex = int(ckSize[0]-scanSize + 1)
    
    

    # # Reshape & store
    # switch ACQ_dim
    if ACQ_dim == 1:
        
        frameData = np.reshape(frameData,(scanSize, 1, 1, 1, numSelectedReceivers, NI, NR) , order='F')
        data = np.zeros((ckSize[0], 1, 1, 1, numSelectedReceivers, NI, NR), dtype=complex)
        data[readStartIndex-1:,0,0,0,:,:,:] = frameData
    elif ACQ_dim == 2: 
        frameData=np.reshape(frameData,(scanSize, ACQ_size[1], 1, 1, numSelectedReceivers, NI, NR) , order='F')
        data=np.zeros([ckSize[0], ckSize[1], 1, 1, numSelectedReceivers, NI, NR], dtype=complex)
        encSteps1indices = PVM_EncSteps1 + ckCenterIndex[1] - 1
        data[readStartIndex-1:,encSteps1indices,0,0,:,:,:] = frameData               
    elif ACQ_dim == 3:
        #if NR == 1:
        #    frameData = np.reshape(frameData,(scanSize, int(ACQ_size[1]), int(ACQ_size[2]), 1, numSelectedReceivers, NI) , order='F')
        #else:
        frameData = np.reshape(frameData,(scanSize, int(ACQ_size[1]), int(ACQ_size[2]), 1, numSelectedReceivers, NI, NR) , order='F')
        
        data=np.zeros([int(ckSize[0]), int(ckSize[1]), int(ckSize[2]), 1, numSelectedReceivers, NI, NR], dtype=complex)
        encSteps1indices = (PVM_EncSteps1 + ckCenterIndex[1] - 1).astype(int)
        encSteps2indices = (PVM_EncSteps2 + ckCenterIndex[2] - 1).astype(int)
        
        data[readStartIndex-1:,list(encSteps1indices),:,:,:,:,:] = frameData[:,:,list(encSteps2indices),:,:,:,:]
    else:
        raise 'Unknown ACQ_dim with useMethod'
    
    return data

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

if 'quadrature' in recopart:
    for NR in range(N7):
        for NI in range(N6):
            for chan in range(N5):
                reco_qopts(kdata[:,:,:,:,chan,NI,NR], reco, NI*NR)
    
if 'phase_rotate' in recopart:
    for NR in range(N7):
        for NI in range(N6):
            for chan in range(N5):
                phase_rot(kdata[:,:,:,:,chan,NI,NR], reco, NI*NR)
    
# if 'zero_filling' in recopart:
#     zero_fill()

# if 'FT' in recopart:
#     ft()

# if 'phase_corr_pi' in recopart:
#     phase_corr_pi()
 
# if 'cutoff' in recopart:  
#     cutoff()
    
# if 'scale_phase_channels' in recopart: 
#     scale_phase_channels()
    
# if 'sumOfSquares' in recopart:
#     sumofsquares()
    
# if 'transposition' in recopart:
#     transpoition()

# from numpy.fft import fft2, ifft2,fftshift, ifftshift, fftn

# kdata = fftn(kdata[:,:,:,0,0,0,0])

# for i in range(N3):
    
#     img = fftshift(np.squeeze(kdata[:,:,i]))
#     #img = fft2(img)
#     plt.figure()
#     plt.imshow(np.abs(img))
#     plt.title(i) 