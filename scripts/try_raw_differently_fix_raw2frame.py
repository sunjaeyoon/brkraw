# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 23:48:37 2023

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

from raw2frame import convertFrameToCKData, brkraw_Reco

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

# def convertRawToFrame(data, acqp, meth):
    
#     results = data.copy()
    
    
#     # Turn Raw into frame
#     NI = get_value(acqp, 'NI')
#     NR = get_value(acqp, 'NR')
    
#     ACQ_phase_factor    = get_value(acqp,'ACQ_phase_factor')
#     ACQ_obj_order       = get_value(acqp, 'ACQ_obj_order')
#     if isinstance(ACQ_obj_order, int):
#         ACQ_obj_order = [ACQ_obj_order]
    
#     ACQ_dim             = get_value(acqp, 'ACQ_dim')
#     numSelectedRecievers= results.shape[-2]
    
#     acqSizes = np.zeros(ACQ_dim)
#     #disp(get_value(acqp, 'ACQ_jobs'))
#     scanSize = get_value(acqp, 'ACQ_jobs')[0][0]
    
#     isSpatialDim = [i == 'Spatial' for i in get_value(acqp, 'ACQ_dim_desc')]
#     spatialDims = sum(isSpatialDim)
    
#     if isSpatialDim[0]:
#         if spatialDims == 3:
#             acqSizes[1:] = get_value(meth, 'PVM_EncMatrix')[1:]
#         elif spatialDims == 2:
#             acqSizes[1]  = get_value(meth, 'PVM_EncMatrix')[1]
    
#     numresultsHighDim=np.prod(acqSizes[1:])
#     acqSizes[0] = scanSize   
    
#     if np.iscomplexobj(results):
#         scanSize = int(acqSizes[0]/2)
#     else:  
#         scanSize = acqSizes[0]
    
    
#     # print(int(numSelectedRecievers), 
#     #       int(scanSize), 
#     #       int(ACQ_phase_factor), 
#     #       int(NI), 
#     #       int(numresultsHighDim/ACQ_phase_factor),
#     #       int(NR))
    
    
#     #print(ACQ_obj_order)
    
#     # # Resort
#     if ACQ_dim>1:
#         # [NR, -1 ,ACQ_phase_factor, NI?, channel, scan_size]
#         # Order is a guess, will need to adjust where int(ACQ_phase_factor) and NR go 
#         results = results.reshape(int(NR),
#                                   int(numresultsHighDim/ACQ_phase_factor), 
#                                   int(ACQ_phase_factor), 
#                                   int(NI), 
#                                   int(numSelectedRecievers),
#                                   int(scanSize))
            
#         # for i in range(int(numSelectedReceivers)):
#         #     print(results[i].shape)
#         #     each_rec = results[i].reshape( int(scanSize), int(ACQ_phase_factor), 
#         #     int(NI), int(numresultsHighDim/ACQ_phase_factor),int(NR))  
#         #     t1[i,:,:,:,:] = each_rec
        
#         # => scansize, ACQ_phase_factor, numDataHighDim/ACQ_phase_factor, numSelectedReceivers, NI, NR
#         print(results.shape)
#         results = np.transpose(results, (5, 4, 3, 2, 1, 0)) 
#         print(results.shape)
        
#         results = results.reshape(int(NR),
#                                   int(numresultsHighDim),  
#                                   int(NI),
#                                   int(numSelectedRecievers),
#                                   int(scanSize))
        
#         # Commented out (IDK if theres any use)
#         #if NI != len(ACQ_obj_order):
#         #    print('Size of ACQ_obj_order is not equal to NI: Data from unsupported method?')
#         #print(np.sum(np.equal(results[:,:,0,1,0], results[:,:,0,2,0])))
#         #return_out = np.zeros_like(results)
#         #p_frame = results[:,:,:,[ACQ_obj_order],:,:]
        
#         frame = np.zeros_like(results)
#         #for i in range(len(ACQ_obj_order)):
#         frame[:,:,ACQ_obj_order,:,:] = results[:,:,:,:,:]#ACQ_obj_order[i],:,:]
        
#         # int(NR), int(numresultsHighDim), int(NI), int(numSelectedRecievers), int(scanSize)
#         # scansize, numDataHighDim/ACQ_phase_factor, numSelectedReceivers, NI, NR
#         frame = np.transpose(frame, (4, 1, 3, 2, 0))#[:,:,:,:,0]
        
#     else:
#         # Havent encountered this situation yet
#         results = np.reshape(results,(numSelectedRecievers, scanSize,1,NI,NR), order='F')
#         return_out = np.zeros_like(results)
#         return_out = np.transpose(results, (1, 2, 0, 3, 4))
#         frame = None
    
#     return frame


if __name__ == "__main__":
    i = 3
    #i = 10
    #i = 50
    #i = 145 # T2 star 4
    #i = 143 # T1 Rare 4
    #i = 142 # localizer 4
    #i = 167 # T1 FLASH 4
    i = 186 # T1 RARE 1
    #i = 188 # T1 FLASH 1
    i = 14
    mat_ds = loadmat(f'E:\\TimHo\\brkraw\\scripts\\Test_Sets\\{i}.mat')
    ExpNum          = mat_ds['study'][0][0]
    scan_name       = str(mat_ds['scan_name'][0])
    folder          = os.path.dirname(mat_ds['pathTestfolder'][0])
    test_raw        = mat_ds[ 'raw']
    test_frame      = mat_ds['frame']
    test_kdata      = mat_ds['kdata']
    #test_image_c    = mat_ds['image_comp']
    test_image      = mat_ds['image_real']
    
    frame = loadmat('E:\\TimHo\\brkraw\\scripts\\Test_Sets\\frame_test_T1_rare_2.mat')#frame_test208_288_1_19')
    
    trans1          = frame[ 'trans1']
    trans2          = frame[ 'trans2']
    trans3          = frame[ 'trans3']
    
    
    # Load my stuff
    datahandler = br.load(os.path.join(folder))
    
    # Get MetaData
    acqp = datahandler.get_acqp(ExpNum)
    meth = datahandler.get_method(ExpNum)
    with open(os.path.join(folder, str(ExpNum),'pdata','1','reco'),'r') as f:
        reco = Parameter(f.read().split('\n'))
    
    print(get_value(acqp, 'ACQ_protocol_name'), ExpNum)
    
    fid_binary = datahandler.get_fid(ExpNum)
    
    # FID 2 RAW SORTED
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
    
    p_X = fid[::2] + 1j*fid[1::2] # Assume data is complex
    
    p_raw = readBrukerRaw(fid_binary, acqp, meth).astype(np.complex64)
    N1, N2, N3 = p_raw.shape
    
    # Track Indexes
    # with open('indexes.txt','w') as f:
    #     for i in X:
    #         f.write(str(np.where(p_raw == i)) +'\n')
    """
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                assert(p_raw[i,j,k] == test_raw[i,j,k])
    """
    assert(np.mean(np.equal(p_raw.transpose(1,2,0), test_raw)) == 1.0) 
    
    
    ### RAW TO FRAME-----------------------------------------------------------
    #p_frame = convertRawToFrame(p_raw, acqp, meth).astype(np.complex64)[:,:,:,:,0]
    results = p_raw.copy()
    
    
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
    
    
    print("NR\tRem\t\tACQ_Phase\tNI\tReceiver\tScanSize\n"+
        f'{int(NR)}\t{int(numresultsHighDim/ACQ_phase_factor)}\t\t\t{int(ACQ_phase_factor)}\t{int(NI)}\t{int(numSelectedRecievers)}\t\t\t{int(scanSize)}'
        )
    
    print(ACQ_obj_order)
    
    # # Resort
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
            int(NR), order='F').copy() 
        
        p_frame = np.zeros_like(results3)
        p_frame[:,:,:,ACQ_obj_order,:] = results3 
        
        # Commented out (IDK if theres any use)
        #if NI != len(ACQ_obj_order):
        #    print('Size of ACQ_obj_order is not equal to NI: Data from unsupported method?')
        
        # We have a bug, Fortran vs C indexing
        if not np.log2(results.shape[0]).is_integer() or not np.log2(results.shape[1]).is_integer():
            print('Current bug with any matrix that is not 2^n ')
        

        
    
    
    # N1, N2, N3, N4 = p_frame.shape
    # for i in range(N1):
    #     for j in range(N2):
    #         for k in range(N3):
    #             for l in range(N4):
    #                 assert(p_frame[i,j,k,l] == test_frame[i,j,k,l])
    
    

    
    # Using Trans
    N1, N2, N3, N4, N5 = trans1.shape

    # print(np.where(results1 == trans1[0,0,0,0,1]))
    # print(np.where(results1 == trans1[0,0,0,1,0]))
    # print(np.where(results1 == trans1[0,0,1,0,0]))
    # print(np.where(results1 == trans1[0,1,0,0,0]))
    # print(np.where(results == trans1[0,2,1,3,4]))
    
    count = 0
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                for l in range(N4):
                    for m in range(N5):
                        if count > 10000:
                            break
                        if 'rare' in get_value(meth,'Method').lower():
                            print(i,j,k,l,m)
                            #print(np.where(results1 == trans1[i,j,k,l,m]))#ACQ_obj_order[l]]))
                            assert(i+j+k+l+m == np.sum(np.where(results1 == trans1[i,j,k,l,m])))
                            #pass
                           # print(np.where(results1 == trans1[i,j,k,l]))
                            #print(i+j+k+l == np.sum(np.where(results == test_frame[i,j,k,l])))
                            #print(np.where(p_frame == test_frame[i,j,k,l]))
                        #print()
                        count += 1
                   
    # print()
    print()                    
    N1, N2, N3, N4, N5 = trans2.shape
    count = 0
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                for l in range(N4):
                    for m in range(N5):
                        if count > 10000:
                            break
                        if 'rare' in get_value(meth,'Method').lower():
                            print(i,j,k,l,m)
                            assert(i+j+k+l+m == np.sum(np.where(results2 == trans2[i,j,k,l,m])))
                            #print(np.where(results2 == trans2[i,j,k,l,m]))#ACQ_obj_order[l]]))
                            #print(np.where(results2 == test_frame[i,j,k,l]))
                            #print(i+j+k+l == np.sum(np.where(results == test_frame[i,j,k,l])))
                            #print(np.where(p_frame == test_frame[i,j,k,l]))
                        #print()
                        count += 1
                        
    print()                    
    N1, N2, N3, N4 = trans3.shape
    count = 0
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                for l in range(N4):
                    if count > 10000:
                        break
                    if 'rare' in get_value(meth,'Method').lower():
                        print(i,j,k,l)
                        assert(i+j+k+l == np.sum(np.where(results3 == trans3[i,j,k,l])))
                        #print(np.where(results2 == trans2[i,j,k,l,m]))#ACQ_obj_order[l]]))
                        #print(np.where(results2 == test_frame[i,j,k,l]))
                        #print(i+j+k+l == np.sum(np.where(results == test_frame[i,j,k,l])))
                        #print(np.where(p_frame == test_frame[i,j,k,l]))
                    #print()
                    count += 1
    
    N1, N2, N3, N4 = test_frame.shape
    #assert(np.mean(np.equal(p_frame, test_frame)) == 1.0)
    
    #Using Frame
    count = 0
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                for l in range(N4):
                    if count > 10000:
                        break
                    if 'rare' in get_value(meth,'Method').lower():
                        print(i,j,k,l)
                        print(np.where(p_frame == test_frame[i,j,k,l]))#ACQ_obj_order[l]]))
                        #print(np.where(results1 == test_frame[i,j,k,l]))
                        #print(i+j+k+l == np.sum(np.where(results == test_frame[i,j,k,l])))
                        #print(np.where(p_frame == test_frame[i,j,k,l]))
                        assert(i+j+k+l == np.sum(np.where(p_frame == test_frame[i,j,k,l])))
                    #print()
                    count += 1   

             
    # assert(np.mean(np.equal(p_frame[:,:,:,:,0], test_frame)) == 1.0)
    
    p_kdata = convertFrameToCKData(p_frame, acqp, meth)
    
    # # assert(np.mean(np.equal(p_kdata[:,:,:,:,0],test_kdata)) == 1.0)

    # # stuff = ['quadrature', 'phase_rotate', 'FT', 'phase_corr_pi']
    p_image = brkraw_Reco(p_kdata, reco, meth, recoparts = 'all') #stuff)
    # # for j in range(63,65):
    # #     for i in range(8):
    # #         plt.figure()
    # #         plt.subplot(1,2,1)
    # #         plt.imshow(np.squeeze(p_image[:,:,j,0,0,i,0]))
    # #         plt.subplot(1,2,2)
    # #         plt.imshow(np.squeeze(test_image[:,:,j,0,0,i]))
    # #         plt.title(f'Slcie {j} TE {i}')
            
    # #         print(np.mean(np.equal(np.squeeze(p_image[:,:,j,0,0,i,0]), test_image[:,:,j,0,0,i])))
    
    
    # [N1,N2,N3,N4,N5,N6,N7] = p_image.shape
    # for i in range(1000):
    #     x    = random.randint(0,N1-1)
    #     y    = random.randint(0,N2-1)
    #     z    = random.randint(0,N3-1)
    #     a    = random.randint(0,N4-1)
    #     b    = random.randint(0,N5-1)
    #     c    = random.randint(0,N6-1)
        
    #     assert(round(p_image[x,y,z,a,b,c,0],5) == round(test_image[x,y,z,a,b,c],5))