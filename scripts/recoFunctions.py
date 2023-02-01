# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:57:02 2023

@author: Timothy
"""
import sys
sys.path.append("..")

from brkraw.lib.utils import get_value
import numpy as np


def reco_qopts(frame, Reco, actual_framenumber):
    Reco_qopts = get_value(Reco, 'RECO_qopts') 
    
    dims = frame.shape[:4]
    use_qneg = False
    if 'QUAD_NEGATION' in Reco_qopts or 'CONJ_AND_QNEG' in Reco_qopts:
        use_qneg = True
        qneg = np.ones(frame.shape)
    
    
    #print('break') 
    for i in range(len(Reco_qopts)):
        #print(i)
        if Reco_qopts[i] == 'COMPLEX_CONJUGATE':
            pass
        
        elif Reco_qopts[i] == 'QUAD_NEGATION':
            if i == 0:
                pass
            elif i == 1:
                pass
            elif i == 2:
                pass
            elif i == 3:
                pass
            
        elif Reco_qopts[i] == 'CONJ_AND_QNEG':
            if i == 0:
                pass
            elif i == 1:
                pass
            elif i == 2:
                pass
            elif i == 3:
                pass
    
    if use_qneg:
        # odd dimension size (ceil() does affekt the size)
        if len(qneg) != len(frame):
            pass    
            #qneg=qneg(1:dims(1), 1:dims(2), 1:dims(3), 1:dims(4));
        #frame = frame*qneg
    
    return frame
    
# if 'phase_rotate' in recopart:
def phase_rot(frame, Reco, actual_framenumber):
    
    RECO_ft_mode = get_value(Reco, 'RECO_ft_mode')
    RECO_rotate = get_value(Reco, 'RECO_rotate')[:,actual_framenumber]
    
    #print(get_value(Reco, 'RECO_rotate')[:,actual_framenumber])
    #print( get_value(Reco, 'RECO_ft_mode') )
    if type(RECO_ft_mode) == list:
        RECO_ft_mode=RECO_ft_mode[0]
    
    
    dims = frame.shape
    phase_matrix = np.ones(dims)
    for i in range(1,len(dims)):
        #pass
    
    
    return frame
# if 'zero_filling' in recopart:
def zero_fill():
    pass
# if 'FT' in recopart:
def ft():
    pass

# if 'phase_corr_pi' in recopart:
def phase_corr_pi():
    pass
# if 'cutoff' in recopart:  
def cutoff():
    pass
# if 'scale_phase_channels' in recopart: 
def scale_phase_channels():
    pass
# if 'sumOfSquares' in recopart:
def sumofsquares():
    pass
# if 'transposition' in recopart:
def transpoition():
    pass