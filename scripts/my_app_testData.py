# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:45:38 2023

@author: Timothy
"""

import sys
sys.path.append("..")
import os

import time

import numpy as np
import matplotlib.pyplot as plt

from raw2frame import *
import brkraw as br

import streamlit as st


@st.cache_data
def test(i): 
    from scipy.io import loadmat
    from brkraw.lib.parser import Parameter
    from brkraw.lib.pvobj import PvDatasetDir
    from brkraw.lib.utils import get_value, set_value
    
    mat_ds = loadmat(f'E:\\TimHo\\brkraw\\scripts\\Test_Sets\\{i}.mat')
    ExpNum          = mat_ds['study'][0][0]
    scan_name       = str(mat_ds['scan_name'][0])
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
    st.title(f"{scan_name}")
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
    
    return (p_image, test_image)



#try:
test_num = st.slider('Test Num', 1, 300, 1)
images, test_img = test(test_num)

col1, col2 = st.columns(2)
view = st.sidebar.radio(
        "Views",
        ('1', '2', '3'))

with col1:
    
    # Main Section-------------------------------
    N1,N2,N3,N4,N5,N6,N7 = images.shape

    if N1 > 1 and view == '3':
        n1 = st.slider('N1', 0, N1-1, int(N1/2))
    else:
        n1 = 0
        
    if N2 > 1 and view == '2':
        n2 = st.slider('N2', 0, N2-1, int(N2/2))
    else:
        n2 = 0

    if N3 > 1 and view == '1':
        n3 = st.slider('N3', 0, N3-1, int(N3/2))
    else:
        n3 = 0
        
    if N4 > 1:
        n4 = st.slider('N4', 0, N4-1, 0)
    else:
        n4 = 0
        
    if N5 > 1:
        n5 = st.slider('N5', 0, N5-1, 0)
    else:
        n5 = 0

    if N6 > 1:
        n6 = st.slider('N6', 0, N6-1, 0)
    else:
        n6 = 0
        
    if N7 > 1:
        n7 = st.slider('N7', 0, N7-1, 0)
    else:
        n7 = 0

    st.header(f'[:,:,{n3},{n4},{n5},{n6},{n7}]')
    if view == '1':
        st.image(np.squeeze(images[:,:,n3,n4,n5,n6,n7])/np.max(images), use_column_width=True)
        
    elif view == '2':
        st.image(np.squeeze(images[:,n2,:,n4,n5,n6,n7])/np.max(images), use_column_width=True)

    elif view == '3':
        st.image(np.squeeze(images[n1,:,:,n4,n5,n6,n7])/np.max(images), use_column_width=True)
        

with col2:

    # Main Section-------------------------------
    
    #st.header(f'{test_img.shape}')
    new_shape = np.ones(shape=7)
    new_shape[:len(test_img.shape)] = test_img.shape
    test_img = test_img.reshape(new_shape.astype(int))
    M1,M2,M3,M4,M5,M6,M7 = test_img.shape
    
    if M1 > 1 and view == '3':
        m1 = st.slider('m1', 0, M1-1, int(M1/2))
    else:
        m1 = 0
        
    if M2 > 1 and view == '2':
        m2 = st.slider('M2', 0, M2-1, int(M2/2))
    else:
        m2 = 0

    if M3 > 1 and view == '1':
        m3 = st.slider('M3', 0, M3-1, int(M3/2))
    else:
        m3 = 0
        
    if M4 > 1:
        m4 = st.slider('M4', 0, M4-1, 0)
    else:
        m4 = 0
        
    if M5 > 1:
        m5 = st.slider('M5', 0, M5-1, 0)
    else:
        m5 = 0

    if M6 > 1:
        m6 = st.slider('M6', 0, M6-1, 0)
    else:
        m6 = 0
        
    if N7 > 1:
        m7 = st.slider('M7', 0, M7-1, 0)
    else:
        M7 = 0

    st.header(f'[:,:,{n3},{n4},{n5},{n6},{n7}]')
    
    if view == '1':
        st.image(np.squeeze(test_img[:,:,n3,n4,n5,n6,n7])/np.max(test_img), use_column_width=True)
        
    elif view == '2':
        st.image(np.squeeze(test_img[:,n2,:,n4,n5,n6,n7])/np.max(test_img), use_column_width=True)

    elif view == '3':
        st.image(np.squeeze(test_img[n1,:,:,n4,n5,n6,n7])/np.max(test_img), use_column_width=True)
        

    # if view == '1':
    #     st.image(np.squeeze(test_img[:,:,m3,m4,m5,m6,n7])/np.max(test_img), use_column_width=True)
        
    # elif view == '2':
    #     st.image(np.squeeze(test_img[:,m2,:,m4,m5,m6,m7])/np.max(test_img), use_column_width=True)

    # elif view == '3':
    #     st.image(np.squeeze(test_img[m1,:,:,m4,m5,m6,m7])/np.max(test_img), use_column_width=True)