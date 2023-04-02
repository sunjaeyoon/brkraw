# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:58:06 2023

@author: Timothy
"""
import numpy as np

def reco_qopts(frame, Reco, actual_framenumber):
    # check input
    if not ('RECO_qopts' in Reco and isinstance(Reco['RECO_qopts'], list)):
        raise ValueError('RECO_qopts is missing or not a list')

    # import variables
    RECO_qopts = Reco['RECO_qopts']

    # claculate additional parameters
    dims = [frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]]

    # check if the qneg-Matrix is necessary:
    use_qneg = False
    if (RECO_qopts.count('QUAD_NEGATION') + RECO_qopts.count('CONJ_AND_QNEG')) >= 1:
        use_qneg = True
        qneg = np.ones(frame.shape)  # Matrix containing QUAD_NEGATION multiplication matrix

    # start preocess
    for i in range(len(RECO_qopts)):
        if RECO_qopts[i] == 'COMPLEX_CONJUGATE':
            frame = np.conj(frame)
        elif RECO_qopts[i] == 'QUAD_NEGATION':
            if i == 0:
                qneg[:, :, :, :] *= np.tile(np.array([1, -1]).reshape((2, 1, 1, 1)), (np.ceil(dims[0] / 2).astype(int), dims[1], dims[2], dims[3]))
            elif i == 1:
                qneg[:, :, :, :] *= np.tile(np.array([1, -1]).reshape((1, 2, 1, 1)), (dims[0], np.ceil(dims[1] / 2).astype(int), dims[2], dims[3]))
            elif i == 2:
                tmp = np.zeros((1, 1, dims[2], 2))
                tmp[0, 0, :, :] = np.array([1, -1]).reshape((1, 1, 1, 2))
                qneg[:, :, :, :] *= np.tile(tmp, (dims[0], dims[1], np.ceil(dims[2] / 2).astype(int), dims[3]))
            elif i == 3:
                tmp = np.zeros((1, 1, 1, dims[3], 2))
                tmp[0, 0, 0, :, :] = np.array([1, -1]).reshape((1, 1, 1, 1, 2))
                qneg[:, :, :, :] *= np.tile(tmp, (dims[0], dims[1], dims[2], np.ceil(dims[3] / 2).astype(int)))

        elif RECO_qopts[i] == 'CONJ_AND_QNEG':
            frame = np.conj(frame)
            if i == 0:
                qneg[:, :, :, :] *= np.tile(np.array([1, -1]).reshape((2, 1, 1, 1)), (np.ceil(dims[0] / 2).astype(int), dims[1], dims[2], dims[3]))
            elif i == 1:
                qneg[:, :, :, :] *= np.tile(np.array([1, -1]).reshape((1, 2, 1, 1)), (dims[0], np.ceil(dims[1] / 2).astype(int), dims[2], dims[3]))
            elif i == 2:
                tmp = np.zeros((1, 1, dims[2], 2))




def reco_phase_rotate(frame, Reco, actual_framenumber):
    # check input
    if not ('RECO_rotate' in Reco and 'RECO_ft_mode' in Reco):
        raise ValueError('RECO_rotate or RECO_ft_mode is missing')
        
    # import variables
    RECO_rotate = Reco['RECO_rotate'][:, actual_framenumber]
    if isinstance(Reco['RECO_ft_mode'], list):
        if any(x != Reco['RECO_ft_mode'][0] for x in Reco['RECO_ft_mode']):
            raise ValueError('It''s not allowed to use different transfomations on different Dimensions: ' + Reco['RECO_ft_mode'])
        RECO_ft_mode = Reco['RECO_ft_mode'][0]
    else:
        RECO_ft_mode = Reco['RECO_ft_mode']

    # calculate additional variables
    dims = [frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]]

    # start process
    phase_matrix = np.ones_like(frame)
    for index in range(len(frame.shape)):
        f = np.arange(dims[index])
        if RECO_ft_mode in ['COMPLEX_FT', 'COMPLEX_FFT']:
            phase_vector = np.exp(1j*2*np.pi*RECO_rotate[index]*f/dims[index])
        elif RECO_ft_mode in ['NO_FT', 'NO_FFT']:
            phase_vector = np.ones_like(f)
        elif RECO_ft_mode in ['COMPLEX_IFT', 'COMPLEX_IFFT']:
            phase_vector = np.exp(1j*2*np.pi*(1-RECO_rotate[index])*f/dims[index])
        else:
            raise ValueError('Your RECO_ft_mode is not supported')

        if index == 0:
            phase_matrix *= np.tile(phase_vector[:,np.newaxis,np.newaxis,np.newaxis], [1, dims[1], dims[2], dims[3]])
        elif index == 1:
            phase_matrix *= np.tile(phase_vector[np.newaxis,:,np.newaxis,np.newaxis], [dims[0], 1, dims[2], dims[3]])
        elif index == 2:
            tmp = np.zeros((1,1,dims[2],1), dtype=np.complex)
            tmp[0,0,:,0] = phase_vector
            phase_matrix *= np.tile(tmp, [dims[0], dims[1], 1, dims[3]])
        elif index == 3:
            tmp = np.zeros((1,1,1,dims[3]), dtype=np.complex)
            tmp[0,0,0,:] = phase_vector
            phase_matrix *= np.tile(tmp, [dims[0], dims[1], dims[2], 1])

    frame *= phase_matrix
    return frame


import numpy as np

def reco_zero_filling(frame, Reco, actual_framenumber, signal_position):
    # check input
    if not ('RECO_ft_size' in Reco and isinstance(Reco['RECO_ft_size'], tuple)):
        raise ValueError('RECO_ft_size is missing or has an invalid format')

    # use function only if: Reco.RECO_ft_size is not equal to size(frame)
    if frame.shape != Reco['RECO_ft_size']:
        if any(signal_position > 1) or any(signal_position < 0):
            raise ValueError('signal_position has to be a vector between 0 and 1')

        # import variables:
        RECO_ft_size = Reco['RECO_ft_size']

        # check if ft_size is correct:
        for i in range(len(RECO_ft_size)):
            if not np.log2(RECO_ft_size[i]).is_integer():
                raise ValueError('RECO_ft_size has to be only of 2^n ')
            if RECO_ft_size[i] < frame.shape[i]:
                raise ValueError('RECO_ft_size has to be bigger than the size of your data-matrix')

        # calculate additional variables
        dims = (frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3])

        # start process

        # Dimensions of frame and RECO_ft_size doesn't match? -> zero filling
        if not all(frame.shape == RECO_ft_size):
            newframe = np.zeros(RECO_ft_size)
            startpos = np.zeros(len(RECO_ft_size), dtype=int)
            pos_ges = [None] * 4

            for i in range(len(RECO_ft_size)):
                diff = RECO_ft_size[i] - frame.shape[i] + 1
                startpos[i] = int(np.floor(diff * signal_position[i] + 1))
                if startpos[i] > RECO_ft_size[i]:
                    startpos[i] = RECO_ft_size[i]
                pos_ges[i] = slice(startpos[i] - 1, startpos[i] - 1 + dims[i])
                
            newframe[pos_ges[0], pos_ges[1], pos_ges[2], pos_ges[3]] = frame
        else:
            newframe = frame

        del startpos, pos_ges

    else:
        newframe = frame

    return newframe


import numpy as np

def reco_FT(frame, Reco, actual_framenumber):
    """
    Perform Fourier Transform on the input frame according to the specified RECO_ft_mode in the Reco dictionary.
    
    Args:
    frame: ndarray
        Input frame to perform Fourier Transform on
    Reco: dict
        Dictionary containing the specified Fourier Transform mode (RECO_ft_mode)
    actual_framenumber: int
        Index of the current frame
    
    Returns:
    frame: ndarray
        Output frame after Fourier Transform has been applied
    """
    # Check input
    if ('RECO_ft_mode' not in Reco.keys()) or (not isinstance(Reco['RECO_ft_mode'], str)):
        raise ValueError('RECO_ft_mode is missing or not a string')
    
    # Import variables
    dimnumber = np.where(np.array(frame.shape) != 1)[0][0] + 1
    RECO_ft_mode = Reco['RECO_ft_mode']
    
    # Start process
    if RECO_ft_mode in ['COMPLEX_FT', 'COMPLEX_FFT']:
        frame = np.fft.fftn(frame)
    elif RECO_ft_mode in ['NO_FT', 'NO_FFT']:
        pass
    elif RECO_ft_mode in ['COMPLEX_IFT', 'COMPLEX_IFFT']:
        frame = np.fft.ifftn(frame)
    else:
        raise ValueError('Your RECO_ft_mode is not supported')
        
    return frame


def reco_cutoff(frame, Reco, actual_framenumber):
    """
    Crops the input frame according to the specified RECO_size in the Reco dictionary.
    
    Args:
    frame: ndarray
        Input frame to crop
    Reco: dict
        Dictionary containing the specified crop size (RECO_size) and offset (RECO_offset)
    actual_framenumber: int
        Index of the current frame
    
    Returns:
    newframe: ndarray
        Cropped output frame
    """
    # Check input
    if ('RECO_size' not in Reco.keys()) or (not isinstance(Reco['RECO_size'], tuple)):
        raise ValueError('RECO_size is missing or not a tuple')
    
    # Use function only if Reco.RECO_size is not equal to size(frame)
    if not (Reco['RECO_size'] == frame.shape):
        if ('RECO_offset' not in Reco.keys()) or (not isinstance(Reco['RECO_offset'], np.ndarray)):
            raise ValueError('RECO_offset is missing or not an ndarray')
        
        # Import variables
        RECO_offset = Reco['RECO_offset'][:, actual_framenumber]
        RECO_size = Reco['RECO_size']
        
        # Cut the new part with RECO_size and RECO_offset
        pos_ges = []
        for i in range(len(RECO_size)):
            # +1 because RECO_offset starts with 0
            pos_ges.append(slice(RECO_offset[i], RECO_offset[i] + RECO_size[i]))
        newframe = frame[tuple(pos_ges)]
    else:
        newframe = frame
    
    return newframe


def reco_image_rotate(frame, Reco, actual_framenumber):
    # check input
    if not ('RECO_rotate' in Reco and isinstance(Reco['RECO_rotate'], np.ndarray)):
        raise ValueError('RECO_rotate is missing or not a numpy array')
        
    # import variables
    RECO_rotate = Reco['RECO_rotate'][:, actual_framenumber]

    # start process
    for i in range(len(frame.shape)):
        pixelshift = np.zeros(4)
        pixelshift[i] = round(RECO_rotate[i]*frame.shape[i])
        frame = np.roll(frame, int(pixelshift[i]), axis=i)
        
    return frame


import numpy as np

def reco_phase_corr_pi(frame, Reco, actual_framenumber):
    # start process
    checkerboard = np.ones_like(frame)
    nDim = len(frame.shape)
    for dim in range(nDim):
        line = np.ones((frame.shape[dim],))
        location = tuple([1 if i == dim else frame.shape[i] for i in range(nDim)])
        line[1::2] = -1
        checkerboard *= np.repeat(line.reshape(location), frame.shape[dim], axis=dim)
    frame = frame * checkerboard
    return frame


def reco_scale_phase_channels(frame, Reco, channel):
    # check input
    if 'Reco' in locals() and isinstance(Reco, dict) and 'RecoScaleChan' in Reco and \
       channel <= len(Reco['RecoScaleChan']):
        scale = Reco['RecoScaleChan'][channel]
    else:
        scale = 1.0
        warnings.warn('RecoScaleChan is missing or too short. Using scaling 1.0.', 
                      Warning('MATLAB:bruker_RecoScaleChan'))
    if 'Reco' in locals() and isinstance(Reco, dict) and 'RecoPhaseChan' in Reco and \
       channel <= len(Reco['RecoPhaseChan']):
        phase = Reco['RecoPhaseChan'][channel]
    else:
        phase = 0.0
        warnings.warn('RecoPhaseChan is missing or too short. Using 0 degree phase.',
                      Warning('MATLAB:bruker_RecoPhaseChan'))
    spFactor = scale * np.exp(1j * phase * np.pi / 180.0)
    # multiply each pixel by common scale and phase factor
    frame = spFactor * frame
    return frame


import numpy as np

def reco_channel_sumOfSquares(frame, Reco, actual_framenumber):
    dims = [frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]]
    newframe = np.zeros(dims)
    
    for i in range(frame.shape[4]):
        newframe += np.square(np.abs(frame[..., i]))
    
    newframe = np.sqrt(newframe)
    return newframe


def reco_transposition(frame, Reco, actual_framenumber):
    # Check input
    if not ('RECO_transposition' in Reco and Reco['RECO_transposition']):
        raise ValueError('RECO_transposition is missing')
    
    # Import variables
    RECO_transposition = Reco['RECO_transposition'][actual_framenumber - 1]
    
    # Calculate additional variables
    dims = [frame.shape[i] for i in range(4)]
    
    # Start process
    if RECO_transposition > 0:
        ch_dim1 = (RECO_transposition % 4) + 1
        ch_dim2 = RECO_transposition - 1 + 1
        new_order = list(range(4))
        new_order[ch_dim1] = ch_dim2
        new_order[ch_dim2] = ch_dim1
        frame = np.transpose(frame, new_order)
        frame = np.reshape(frame, dims)
    
    return frame
