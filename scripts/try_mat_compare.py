# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:20:33 2023

@author: Timothy
"""

import sys
sys.path.append("..")
import os
import traceback

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
#from recoFunctions import *

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
    
    X = fid[::2] + 1j*fid[1::2] # Assume data is complex
    
    # [num_lines, channel, scan_size]
    X = np.reshape(X, [dim1, nRecs, int(jobScanSize/2)])
    #X = np.transpose(X, (1, 2, 0)).astype(np.complex64)
    return X


def convertRawToFrame(data, acqp, meth):
    results = data.copy()
     
    # Turn Raw into frame
    NI = get_value(acqp, 'NI')
    NR = get_value(acqp, 'NR')
    
    if 'rare' in get_value(meth,'Method').lower():
        ACQ_phase_factor    = get_value(meth,'PVM_RareFactor')    
    else:
        ACQ_phase_factor    = get_value(acqp,'ACQ_phase_factor')
    
    ACQ_obj_order       = get_value(acqp, 'ACQ_obj_order')
    if isinstance(ACQ_obj_order, int):
        ACQ_obj_order = [ACQ_obj_order]
    
    ACQ_dim             = get_value(acqp, 'ACQ_dim')
    numSelectedRecievers= results.shape[-2]
    
    acqSizes = np.zeros(ACQ_dim)
    #disp(get_value(acqp, 'ACQ_jobs'))
    scanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    
    isSpatialDim = [i == 'Spatial' for i in get_value(acqp, 'ACQ_dim_desc')]
    spatialDims = sum(isSpatialDim)
    
    if isSpatialDim[0]:
        if spatialDims == 3:
            acqSizes[1:] = get_value(meth, 'PVM_EncMatrix')[1:]
        elif spatialDims == 2:
            acqSizes[1]  = get_value(meth, 'PVM_EncMatrix')[1]
    
    numresultsHighDim=np.prod(acqSizes[1:])
    acqSizes[0] = scanSize   
    
    if np.iscomplexobj(results):
        scanSize = int(acqSizes[0]/2)
    else:  
        scanSize = acqSizes[0]
    
    
    # print("NR\tRem\t\tACQ_Phase\tNI\tReceiver\tScanSize\n"+
    #     f'{int(NR)}\t{int(numresultsHighDim/ACQ_phase_factor)}\t\t\t{int(ACQ_phase_factor)}\t{int(NI)}\t{int(numSelectedRecievers)}\t\t\t{int(scanSize)}'
    #     )
    
    # print(ACQ_obj_order)
    
    # Resort
    if ACQ_dim>1:
        # [..., num_lines, channel, scan_size]
        # Order is a guess, will need to adjust where int(ACQ_phase_factor) and NR go 
        results = results.transpose((1,2,0))
        
        results1 = results.reshape(
            int(numSelectedRecievers), 
            int(scanSize), 
            int(ACQ_phase_factor), 
            int(NI), 
            int(numresultsHighDim/ACQ_phase_factor), 
            int(NR), order='F').copy()
        
        
        results2 = np.transpose(results1, (1, 2, 4, 0, 3, 5)) # => scansize, ACQ_phase_factor, numDataHighDim/ACQ_phase_factor, numSelectedReceivers, NI, NR
    
        results3 =  results2.reshape( 
            int(scanSize), 
            int(numresultsHighDim), 
            int(numSelectedRecievers), 
            int(NI), 
            int(NR), order='F') 
        
        frame = np.zeros_like(results3)
        frame[:,:,:,ACQ_obj_order,:] = results3 
        
        # Commented out (IDK if theres any use)
        #if NI != len(ACQ_obj_order):
        #    print('Size of ACQ_obj_order is not equal to NI: Data from unsupported method?')
        
        # We have a bug, Fortran vs C indexing
        #if not np.log2(results.shape[0]).is_integer() or not np.log2(results.shape[1]).is_integer():
        #    print('Current bug with any matrix that is not 2^n ')
        
        
    else:
        # Havent encountered this situation yet
        results = np.reshape(results,(numSelectedRecievers, scanSize,1,NI,NR), order='F')
        return_out = np.zeros_like(results)
        return_out = np.transpose(results, (1, 2, 0, 3, 4))
        frame = None
    
    return frame
    

if __name__ == "__main__":
    #pass
    # Load Matlab Test
    i = 3
    i = 10
    #i = 50
    #i = 145 # T2 star 4
    #i = 143 # T1 Rare 4
    #i = 142 # localizer 4
    #i = 167 # T1 FLASH 4
    #i = 186 # T1 RARE 1
    #i = 188 # T1 FLASH 1
    
    i = 9
    
    mat_ds = loadmat(f'E:\\TimHo\\brkraw\\scripts\\Test_Sets\\{i}.mat')
    
    ExpNum          = mat_ds['study'][0][0]
    scan_name       = str(mat_ds['scan_name'])
    folder          = os.path.dirname(mat_ds['pathTestfolder'][0])
    test_raw        = mat_ds[ 'raw']
    test_frame      = mat_ds['frame']
    test_kdata      = mat_ds['kdata']
    test_image_c    = mat_ds['image_comp']
    test_image      = mat_ds['image_real']
    
    
    # Load my stuff
    datahandler = br.load(os.path.join(folder))
    
    # Get MetaData
    acqp = datahandler.get_acqp(ExpNum)
    meth = datahandler.get_method(ExpNum)
    with open(os.path.join(folder, str(ExpNum),'pdata','1','reco'),'r') as f:
        reco = Parameter(f.read().split('\n'))
    print(get_value(acqp, 'ACQ_protocol_name'), ExpNum)
    
    
    fid_binary = datahandler.get_fid(ExpNum)
    # test functions
    start_time = time.time()
    p_raw = readBrukerRaw(fid_binary, acqp, meth).astype(np.complex64)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(np.mean(np.equal(p_raw.transpose((1,2,0)),test_raw)))
    
    start_time = time.time()
    p_frame = convertRawToFrame(p_raw, acqp, meth)#, trans1, trans2, trans3)#.astype(np.complex64)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #print(trans1.shape, p_frame.shape)
    print(np.mean(np.equal(p_frame[:,:,:,:,0],test_frame)))

    start_time = time.time()
    p_kdata = convertFrameToCKData(p_frame, acqp, meth)
    print("--- %s seconds ---" % (time.time() - start_time))

    [N1,N2,N3,N4,N5,N6,N7] = p_kdata.shape
    for i in range(1000):
        x    = random.randint(0,N1-1)
        y    = random.randint(0,N2-1)
        z    = random.randint(0,N3-1)
        a    = random.randint(0,N4-1)
        b    = random.randint(0,N5-1)
        c    = random.randint(0,N6-1)
        #print(p_kdata[x,y,z,a,b,c,0], test_kdata[x,y,z,a,b,c])
        #assert((p_kdata[x,y,z,a,b,0],5) == round(test_image[x,y,z,a,b],5))
    
    start_time = time.time()
    #stuff = ['quadrature', 'phase_rotate', 'FT', 'phase_corr_pi']
    p_image = brkraw_Reco(p_kdata, reco, meth, recoparts = 'all') #stuff)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    [N1,N2,N3,N4,N5,N6,N7] = p_image.shape
    for i in range(1000):
        x    = random.randint(0,N1-1)
        y    = random.randint(0,N2-1)
        z    = random.randint(0,N3-1)
        a    = random.randint(0,N4-1)
        b    = random.randint(0,N5-1)
        c    = random.randint(0,N6-1)
        
        #assert(round(p_image[x,y,z,a,b,c,0],5) == round(test_image[x,y,z,a,b,c],5))
    