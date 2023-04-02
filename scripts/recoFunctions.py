# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:57:02 2023

@author: 
"""

from brkraw.lib.utils import get_value
import numpy as np

def reco_qopts(frame, Reco, actual_framenumber):

    # import variables
    RECO_qopts = get_value(Reco, 'RECO_qopts') # RECO_qopts = Reco['RECO_qopts']

    # claculate additional parameters
    dims = [frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]]

    # check if the qneg-Matrix is necessary:
    use_qneg = False
    if (RECO_qopts.count('QUAD_NEGATION') + RECO_qopts.count('CONJ_AND_QNEG')) >= 1:
        use_qneg = True
        qneg = np.ones(frame.shape)  # Matrix containing QUAD_NEGATION multiplication matrix

    # start process
    for i in range(len(RECO_qopts)):
        if RECO_qopts[i] == 'COMPLEX_CONJUGATE':
            frame = np.conj(frame)
        elif RECO_qopts[i] == 'QUAD_NEGATION':
            if i == 0:
                qneg = qneg * np.tile([[1, -1]], [np.ceil(dims[0]/2), dims[1], dims[2], dims[3]])
            elif i == 1:
                qneg = qneg * np.tile([[1], [-1]], [dims[0], np.ceil(dims[1]/2), dims[2], dims[3]])
            elif i == 2:
                tmp = np.zeros([1, 1, dims[2], 2])
                tmp[0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], np.ceil(dims[2]/2), dims[3]])
            elif i == 3:
                tmp = np.zeros([1, 1, 1, dims[3], 2])
                tmp[0, 0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], dims[2], np.ceil(dims[3]/2)])
        elif RECO_qopts[i] == 'CONJ_AND_QNEG':
            frame = np.conj(frame)
            if i == 0:
                qneg = qneg * np.tile([[1, -1]], [np.ceil(dims[0]/2), dims[1], dims[2], dims[3]])
            elif i == 1:
                qneg = qneg * np.tile([[1], [-1]], [dims[0], np.ceil(dims[1]/2), dims[2], dims[3]])
            elif i == 2:
                tmp = np.zeros([1, 1, dims[2], 2])
                tmp[0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], np.ceil(dims[2]/2), dims[3]])
            elif i == 3:
                tmp = np.zeros([1, 1, 1, dims[3], 2])
                tmp[0, 0, 0, :, :] = [[1, -1]]
                qneg = qneg * np.tile(tmp, [dims[0], dims[1], dims[2], np.ceil(dims[3]/2)])
    
    if use_qneg:
        if qneg.shape != frame.shape:
            qneg = qneg[0:dims[0], 0:dims[1], 0:dims[2], 0:dims[3]]
        frame = frame * qneg
    
    return frame

    
# if 'phase_rotate' in recopart:
def reco_phase_rotate(frame, Reco, actual_framenumber):
    
    # import variables
    #print(actual_framenumber)
    RECO_rotate =  get_value(Reco,'RECO_rotate')[:, actual_framenumber]
    if isinstance( get_value(Reco,'RECO_ft_mode'), list):
        if any(x != get_value(Reco,'RECO_ft_mode')[0] for x in get_value(Reco,'RECO_ft_mode')):
            raise ValueError('It''s not allowed to use different transfomations on different Dimensions: ' + Reco['RECO_ft_mode'])
        RECO_ft_mode = get_value(Reco,'RECO_ft_mode')[0]
    else:
        RECO_ft_mode = get_value(Reco,'RECO_ft_mode')

    # calculate additional variables
    dims = [frame.shape[0], frame.shape[1], frame.shape[2], frame.shape[3]]

    # start process
    phase_matrix = np.ones_like(frame)
    for index in range(len(frame.shape)-1):
        f = np.arange(dims[index])
        #print(f)
        if RECO_ft_mode in ['COMPLEX_FT', 'COMPLEX_FFT']:
            phase_vector = np.exp(1j*2*np.pi*RECO_rotate[index]*f)
        elif RECO_ft_mode in ['NO_FT', 'NO_FFT']:
            phase_vector = np.ones_like(f)
        elif RECO_ft_mode in ['COMPLEX_IFT', 'COMPLEX_IFFT']:
            #print(RECO_rotate[index])
            phase_vector = np.exp(1j*2*np.pi*(1-RECO_rotate[index])*f)
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

# if 'zero_fill' in recopart:
def reco_zero_filling(frame, Reco, actual_framenumber, signal_position):
    # check input
    """
    if not ('RECO_ft_size' in Reco and isinstance(Reco['RECO_ft_size'], tuple)):
        raise ValueError('RECO_ft_size is missing or has an invalid format')
    """
    RECO_ft_mode = get_value(Reco,'RECO_ft_mode')
    
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

# if 'FT' in recopart:
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
    
    # Import variables
    RECO_ft_mode = get_value(Reco,'RECO_ft_mode')[0]
    
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



# if 'phase_corr_pi' in recopart: Something wrong
def reco_phase_corr_pi(frame, Reco, actual_framenumber):
    # start process
    checkerboard = np.ones(shape=frame.shape)

    # Use NumPy broadcasting to alternate the signs
    checkerboard[::2,::2,::2,0] = -1
    checkerboard[1::2,1::2,::2,0] = -1
    
    checkerboard[::2,1::2,1::2,0] = -1
    checkerboard[1::2,::2,1::2,0] = -1
    
    frame = frame * checkerboard * -1
    
    return frame


# if 'cutoff' in recopart:  
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
    
    # Use function only if Reco.RECO_size is not equal to size(frame)
    #print(get_value(Reco,'RECO_size'), frame.shape)
    dim_equal = True
    for i,j in zip(get_value(Reco,'RECO_size'), frame.shape):
        dim_equal = (i==j)
   
    if not dim_equal:
        
        # Import variables
        RECO_offset = get_value(Reco,'RECO_offset')[:, actual_framenumber]
        RECO_size = get_value(Reco, 'RECO_size')
        
        # Cut the new part with RECO_size and RECO_offset
        pos_ges = []
        for i in range(len(RECO_size)):
            # +1 because RECO_offset starts with 0
            pos_ges.append(slice(RECO_offset[i], RECO_offset[i] + RECO_size[i]))
        newframe = frame[tuple(pos_ges)]
    else:
        newframe = frame
    
    return newframe


# if 'scale_phase_channels' in recopart: 
def reco_scale_phase_channels(frame, Reco, channel):
    # check input
    reco_scale = get_value(Reco,'RecoScaleChan')
    if channel <= len(reco_scale) and reco_scale != None:
        scale = reco_scale[channel]
    else:
        scale = 1.0
    
    reco_phase = get_value(Reco,'RecoPhaseChan')
    if channel <= len(reco_phase) and reco_phase != None:
        phase = reco_phase[channel]
    else:
        phase = 0.0
        
    spFactor = scale * np.exp(1j * phase * np.pi / 180.0)
    # multiply each pixel by common scale and phase factor
    frame = spFactor * frame
    return frame

# if 'sumOfSquares' in recopart:
def reco_sumofsquares(frame, Reco): 
    out = np.sqrt( np.sum(np.square(np.abs(frame)), axis=4, keepdims=True) )
    #print(out[1])
    return out

# if 'transposition' in recopart:
def reco_transposition(frame, Reco, actual_framenumber):
    # Check input
    
    # Import variables
    RECO_transposition = get_value(Reco,'RECO_transposition')[actual_framenumber - 1]
    
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



"""
## reco_qopts
function frame=reco_qopts(frame, Reco, actual_framenumber)
# check input
if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_qopts') )
    error('RECO_qopts is missing')
end

# import variables
RECO_qopts=Reco.RECO_qopts;

# claculate additional parameters
dims=[size(frame,1), size(frame,2), size(frame,3), size(frame,4)];

# check if the qneg-Matrix is necessary:
use_qneg=false;
if ( sum(strcmp(RECO_qopts, 'QUAD_NEGATION')) + sum(strcmp(RECO_qopts, 'CONJ_AND_QNEG')) )>=1
    use_qneg=true;
    qneg=ones(size(frame)); # Matrix containing QUAD_NEGATION multiplication matrix
end

# start preocess
#disp('break')
for i=1:length(RECO_qopts)
   
   #disp(RECO_qopts{i})
   switch(RECO_qopts{i})
       case 'COMPLEX_CONJUGATE'
           frame=conj(frame);
       case 'QUAD_NEGATION'
           switch i
               case 1
                   qneg=qneg.*repmat([1, -1]', [ceil(dims(1)/2), dims(2), dims(3), dims(4)]);
               case 2
                   qneg=qneg.*repmat([1, -1], [dims(1), ceil(dims(2)/2), dims(3), dims(4)]);
               case 3
                   tmp(1,1,:)=[1,-1];
                   qneg=qneg.*repmat(tmp, [dims(1), dims(2), ceil(dims(3)/2), dims(4)]);
               case 4
                   tmp(1,1,1,:)=[1,-1];
                   qneg=qneg.*repmat(tmp, [dims(1), dims(2), dims(3), ceil(dims(4)/2)]);
           end                   
                   
       case 'CONJ_AND_QNEG'
           frame=conj(frame);
           switch i
               case 1
                   qneg=qneg.*repmat([1, -1]', [ceil(dims(1)/2), dims(2), dims(3), dims(4)]);
               case 2
                   qneg=qneg.*repmat([1, -1], [dims(1), ceil(dims(2)/2), dims(3), dims(4)]);
               case 3
                   tmp(1,1,:)=[1,-1];
                   qneg=qneg.*repmat(tmp, [dims(1), dims(2), ceil(dims(3)/2), dims(4)]);
               case 4
                   tmp(1,1,1,:)=[1,-1];
                   qneg=qneg.*repmat(tmp, [dims(1), dims(2), dims(3), ceil(dims(4)/2)]);
           end 
   end   
end

if use_qneg
    # odd dimension size (ceil() does affekt the size)
    
    if ~( size(qneg)==size(frame))
        qneg=qneg(1:dims(1), 1:dims(2), 1:dims(3), 1:dims(4));
    end
    frame=frame.*qneg;
end
end

## reco_phase_rotate
function frame=reco_phase_rotate(frame, Reco, actual_framenumber)
# check input
if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_rotate') ... 
        && isfield(Reco, 'RECO_ft_mode'))
    error('RECO_rotate or RECO_ft_mode is missing')
end
# import variables:
#disp(Reco.RECO_rotate(:, actual_framenumber))
#disp(Reco.RECO_ft_mode)
RECO_rotate=Reco.RECO_rotate(:, actual_framenumber);
if iscell(Reco.RECO_ft_mode)
    for i=1:length(Reco.RECO_ft_mode)
        #disp(Reco.RECO_ft_mode{i})
        #disp(strrep(Reco.RECO_ft_mode{i},'FFT','FT'))
        #disp(strrep(Reco.RECO_ft_mode{1},'FFT','FT'))
        if ~strcmp(strrep(Reco.RECO_ft_mode{i},'FFT','FT'), strrep(Reco.RECO_ft_mode{1},'FFT','FT'))
           error(['It''s not allowed to use different transfomations on different Dimensions: ' Reco.RECO_ft_mode{:}]);
        end
    end
    RECO_ft_mode=Reco.RECO_ft_mode{1};
else
    RECO_ft_mode=Reco.RECO_ft_mode;
end

# calculate additional variables
dims=[size(frame,1), size(frame,2), size(frame,3), size(frame,4)];

# start process
phase_matrix=ones(size(frame));
for index=1:length(size(frame))
    # written complete it would be:
    # 0:1/dims(index):(1-1/dims(index));
    # phase_vector=exp(1i*2*pi*RECO_rotate(index)*dims(index)*f);
    f=0:(dims(index)-1);
    #RECO_ft_mode
    switch RECO_ft_mode
    case {'COMPLEX_FT', 'COMPLEX_FFT'}
        phase_vector=exp(1i*2*pi*RECO_rotate(index)*f);
    case {'NO_FT', 'NO_FFT'}
        # no phase ramp
        phase_vector=ones(size(f));
    case {'COMPLEX_IFT', 'COMPLEX_IFFT'}
        phase_vector=exp(1i*2*pi*(1-RECO_rotate(index))*f);
    otherwise
        error('Your RECO_ft_mode is not supported');
    end
    
    # index
    switch index
       case 1
           phase_matrix=phase_matrix.*repmat(phase_vector.', [1, dims(2), dims(3), dims(4)]);
       case 2
           phase_matrix=phase_matrix.*repmat(phase_vector, [dims(1), 1, dims(3), dims(4)]);
       case 3
           tmp(1,1,:)=phase_vector;
           phase_matrix=phase_matrix.*repmat(tmp, [dims(1), dims(2), 1, dims(4)]);
       case 4
           tmp(1,1,1,:)=phase_vector;
           phase_matrix=phase_matrix.*repmat(tmp, [dims(1), dims(2), dims(3), 1]);
    end        
    clear phase_vector f tmp
end
frame=frame.*phase_matrix;


end

## reco_zero_filling
function newframe=reco_zero_filling(frame, Reco, actual_framenumber, signal_position)
# check input
if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_ft_size') )
    error('RECO_ft_size is missing')
end


# use function only if: Reco.RECO_ft_size is not equal to size(frame)
if ~( isequal(size(frame), Reco.RECO_ft_size ) )
    if sum(signal_position >1)>=1 || sum(signal_position < 0)>=1
        error('signal_position has to be a vektor between 0 and 1');
    end

    # import variables:
    RECO_ft_size=Reco.RECO_ft_size;

    # check if ft_size is correct:
    for i=1:length(RECO_ft_size)
#         if ~( log2(RECO_ft_size(i))-floor(log2(RECO_ft_size(i))) == 0 )
#             error('RECO_ft_size has to be only of 2^n ');
#         end
        if RECO_ft_size(i) < size(frame,i)
            error('RECO_ft_size has to be bigger than the size of your data-matrix')
        end
    end

    # calculate additional variables
    dims=[size(frame,1), size(frame,2), size(frame,3), size(frame,4)];

    # start process

    #Dimensions of frame and RECO_ft_size doesn't match? -> zerofilling
    if ~( sum( size(frame)==RECO_ft_size )==length(RECO_ft_size) )
        newframe=zeros(RECO_ft_size);
        startpos=zeros(length(RECO_ft_size),1);
        pos_ges={1, 1, 1, 1};
        for i=1:length(RECO_ft_size)
           diff=RECO_ft_size(i)-size(frame,i)+1;
           startpos(i)=fix(diff*signal_position(i)+1);
           if startpos(i)>RECO_ft_size(i)
               startpos(i)=RECO_ft_size(i);
           end
           pos_ges{i}=startpos(i):startpos(i)+dims(i)-1;
        end
        newframe(pos_ges{1}, pos_ges{2}, pos_ges{3}, pos_ges{4})=frame;
    else
        newframe=frame;
    end
    clear startpos pos_ges diff;
else
    newframe=frame;
end
end

## reco_FT
function frame=reco_FT(frame, Reco, actual_framenumber)
# check input
if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_ft_mode') )
    error('RECO_ft_mode is missing')
end

# import variables:
for i=1:4
    if size(frame,i)>1
        dimnumber=i;
    end
end

if iscell(Reco.RECO_ft_mode)
    for i=1:length(Reco.RECO_ft_mode)
        if ~strcmp(strrep(Reco.RECO_ft_mode{i},'FFT','FT'), strrep(Reco.RECO_ft_mode{1},'FFT','FT'))
           error(['It''s not allowed to use different transfomations on different Dimensions: ' Reco.RECO_ft_mode{:}]);
        end
    end
    RECO_ft_mode=Reco.RECO_ft_mode{1};
else
    RECO_ft_mode=Reco.RECO_ft_mode;
end

# start process
switch RECO_ft_mode    
    case {'COMPLEX_FT', 'COMPLEX_FFT'}
        frame=fftn(frame);
    case {'NO_FT', 'NO_FFT'}
        # do nothing
    case {'COMPLEX_IFT', 'COMPLEX_IFFT'}
        frame=ifftn(frame);
    otherwise
        error('Your RECO_ft_mode is not supported');
end
end

## reco_cutoff
function newframe=reco_cutoff(frame, Reco, actual_framenumber)
# check input
if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_size') )
    error('RECO_size is missing')
end

# use function only if: Reco.RECO_ft_size is not equal to size(frame)
if ~( isequal(size(frame), Reco.RECO_size ) )
    if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_offset') )
        error('RECO_offset is missing')
    end
    if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_size') )
        error('RECO_size is missing')
    end

    # import variables:
    RECO_offset=Reco.RECO_offset(:, actual_framenumber);
    RECO_size=Reco.RECO_size;

    # cut the new part with RECO_size and RECO_offset
    pos_ges={1, 1, 1, 1};
    for i=1:length(RECO_size)
        # +1 because RECO_offset starts with 0
       pos_ges{i}=RECO_offset(i)+1:RECO_offset(i)+RECO_size(i);
    end
    newframe=frame(pos_ges{1}, pos_ges{2}, pos_ges{3}, pos_ges{4});
else
    newframe=frame;
end
end

## reco_image_rotate
function frame=reco_image_rotate(frame, Reco, actual_framenumber)
# check input
if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_rotate') )
    error('RECO_rotate is missing')
end
# import variables:
RECO_rotate=Reco.RECO_rotate(:,actual_framenumber);

# start process
for i=1:length(size(frame))
   pixelshift=zeros(4,1); 
   pixelshift(i)=round(RECO_rotate(i)*size(frame,i));
   frame=circshift(frame, pixelshift);
end
end

## reco_phase_corr_pi
function frame=reco_phase_corr_pi(frame, Reco, actual_framenumber)
# start process
checkerboard = ones(size(frame));
nDim = length(size(frame));
for dim=1:nDim
    line = ones(size(frame, dim), 1);
    location = size(frame);
    location(dim) = 1;
    line(2:2:end) = -1;
    checkerboard = checkerboard .* repmat(shiftdim(line, 1-dim), location);
end
frame = frame.*checkerboard;
end

## reco_scale_phase_channels
function frame=reco_scale_phase_channels(frame, Reco, channel)
# check input
if exist('Reco', 'var') && isstruct(Reco) && isfield(Reco, 'RecoScaleChan') ...
        && channel <= length(Reco.RecoScaleChan)
    scale = Reco.RecoScaleChan(channel);
else
    scale = 1.0;
    warning('MATLAB:bruker_RecoScaleChan','RecoScaleChan is missing or too short. Using scaling 1.0.');
    warning('off','MATLAB:bruker_RecoScaleChan');
end
if exist('Reco', 'var') && isstruct(Reco) && isfield(Reco, 'RecoPhaseChan') ...
        && channel <= length(Reco.RecoPhaseChan)
    phase = Reco.RecoPhaseChan(channel);
else
    phase = 0.0;
    warning('MATLAB:bruker_RecoPhaseChan','RecoPhaseChan is missing or too short. Using 0 degree phase.');
    warning('off','MATLAB:bruker_RecoPhaseChan');
end
spFactor = scale*exp(1i*phase*pi/180.0);
# multiply each pixel by common scale and phase factor
frame = spFactor*frame;
end

## channel_sumOfSquares
function newframe=reco_channel_sumOfSquares(frame, Reco, actual_framenumber)
dims=[size(frame,1), size(frame,2), size(frame,3), size(frame,4)];
newframe=zeros(dims);
for i=1:size(frame,5)
    newframe=newframe+(abs(frame(:,:,:,:,i))).^2;
end
newframe=sqrt(newframe);
end

## reco_transposition
function frame=reco_transposition(frame, Reco, actual_framenumber)
# check input
if ~(exist('Reco','var') && isstruct(Reco) && isfield(Reco, 'RECO_transposition') )
    error('RECO_transposition is missing')
end
# import variables:
RECO_transposition=Reco.RECO_transposition(actual_framenumber);

# calculate additional variables:
dims=[size(frame,1), size(frame,2), size(frame,3), size(frame,4)];

# start process
if RECO_transposition > 0
        ch_dim1=mod(RECO_transposition, length(size(frame)) )+1;
        ch_dim2=RECO_transposition-1+1;
        new_order=1:4;
        new_order(ch_dim1)=ch_dim2;
        new_order(ch_dim2)=ch_dim1;
        frame=permute(frame, new_order);
        frame=reshape(frame, dims);
end
end
"""