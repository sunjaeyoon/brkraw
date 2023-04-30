# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:32:14 2023
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

def test(mat_ds):
    
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
    print(folder)

    fid_binary = datahandler.get_fid(ExpNum)
    # test functions
    start_time = time.time()
    p_raw = readBrukerRaw(fid_binary, acqp, meth).astype(np.complex64)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(p_raw.shape)
    assert(np.mean(np.equal(p_raw.transpose((1,2,0)), test_raw)) == 1.0)   
    
    start_time = time.time()
    p_frame = convertRawToFrame(p_raw, acqp, meth).astype(np.complex64)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(p_frame.shape, test_frame.shape) 
    assert(np.mean(np.equal(np.squeeze(p_frame), np.squeeze(test_frame))) == 1.0)
    
    start_time = time.time()
    p_kdata = convertFrameToCKData(p_frame, acqp, meth)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(p_kdata.shape, test_kdata.shape) 
    
    start_time = time.time()
    #stuff = ['quadrature', 'phase_rotate', 'FT', 'phase_corr_pi']
    p_image = brkraw_Reco(p_kdata, reco, meth, recoparts = 'all') #stuff)
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(len(p_image.shape))#, test_image.shape)
    
    # try:
    #     [N1,N2,N3,N4,N5,N6,N7] = p_kdata.shape
    #     for i in range(1000):
    #         x    = random.randint(0,N1-1)
    #         y    = random.randint(0,N2-1)
    #         z    = random.randint(0,N3-1)
    #         a    = random.randint(0,N4-1)
    #         b    = random.randint(0,N5-1)
    #         c    = random.randint(0,N6-1)
    #         #print(round(p_kdata[x,y,z,a,b,c,0],5) == round(test_kdata[x,y,z,a,b,c],5))
    #         #print(p_kdata[x,y,z,a,b,c,0], test_kdata[x,y,z,a,b,c])
        
        
    #     [N1,N2,N3,N4,N5,N6,N7] = p_image.shape
    #     for i in range(10):
    #         x    = random.randint(0,N1-1)
    #         y    = random.randint(0,N2-1)
    #         z    = random.randint(0,N3-1)
    #         a    = random.randint(0,N4-1)
    #         b    = random.randint(0,N5-1)
    #         c    = random.randint(0,N6-1)    
    #         print(p_image[x,y,z,a,b,c,0], test_image[x,y,z,a,b,c])
    #         assert(round(p_image[x,y,z,a,b,c,0],5) == round(test_image[x,y,z,a,b,c],5))
    #     print('Passed')
    # except Exception as e:
    #     if 'KeyError' not in traceback.format_exc():
    #         print(traceback.format_exc())
    


for i in range(3,300):
    mat_ds = loadmat(f'E:\\TimHo\\brkraw\\scripts\\Test_Sets\\{i}.mat')
    
    #try:
    print(i)
    test(mat_ds)
    print()
    #except Exception as e:        
    #    if 'KeyError' not in traceback.format_exc():
    #        print(traceback.format_exc())