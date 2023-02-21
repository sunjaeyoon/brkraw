# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:57:02 2023

@author: 
"""

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
def reco_phase_rot(frame, Reco, actual_framenumber):
    
    RECO_ft_mode = get_value(Reco, 'RECO_ft_mode')
    RECO_rotate = get_value(Reco, 'RECO_rotate')[:,actual_framenumber]
    #print(RECO_rotate)
    #print(get_value(Reco, 'RECO_rotate')[:,actual_framenumber])
    #print( get_value(Reco, 'RECO_ft_mode') )
    if type(RECO_ft_mode) == list:
        assert len( set( RECO_ft_mode ) ) == 1
        RECO_ft_mode=RECO_ft_mode[0]
    
    
    dims = frame.shape
    #print(dims)
    phase_matrix = np.ones(dims)
    
    for i in range(1,len(dims)):
        f = np.arange(dims[i-1])
        #print(f)
        if RECO_ft_mode in ['COMPLEX_FT', 'COMPLEX_FFT']:
            phase_vector = np.exp(1j*2*np.pi*RECO_rotate[i-1]*f)
        
        elif RECO_ft_mode in ['NO_FT', 'NO_FFT']:
            phase_vector = np.ones(f.shape)
        
        elif RECO_ft_mode in ['COMPLEX_IFT', 'COMPLEX_IFFT']:
            phase_vector = np.exp(1j*2*np.pi*(1-RECO_rotate[i-1])*f)
            
        else:
            print('Not supported thing in phase rot')
        
        if i == 1:
            phase_matrix = phase_matrix * np.tile(phase_vector, [1, dims[1], dims[2], dims[3]]);
        elif i == 2:
            phase_matrix = phase_matrix * np.tile(phase_vector, [dims[0], 1, dims[1], dims[3]]);
        elif i == 3:
            pass
        elif i == 4:
            pass
    
    return frame
# if 'zero_filling' in recopart:
def reco_zero_fill():
    pass
# if 'FT' in recopart:
def reco_ft(frame, Reco):
    
    RECO_ft_mode = get_value(Reco, 'RECO_ft_mode')
    if type(RECO_ft_mode) == list:
        assert len( set( RECO_ft_mode ) ) == 1
        RECO_ft_mode=RECO_ft_mode[0]
    
    
    if RECO_ft_mode in ['COMPLEX_FT', 'COMPLEX_FFT']:
        #print('ft')
        frame = np.fft.fftn(frame)
    elif RECO_ft_mode in ['COMPLEX_IFT', 'COMPLEX_IFFT']:
        #print('ift')
        frame = np.fft.ifftn(frame)
    elif RECO_ft_mode in ['NO_FT', 'NO_FFT']:
        pass
    else:
        print('Not Supported', RECO_ft_mode)
        
    return frame

# if 'phase_corr_pi' in recopart:
def reco_phase_corr_pi():
    pass
# if 'cutoff' in recopart:  
def reco_cutoff():
    pass
# if 'scale_phase_channels' in recopart: 
def reco_scale_phase_channels():
    pass
# if 'sumOfSquares' in recopart:
def reco_sumofsquares(frame, Reco): 
    
    out = np.sqrt( np.sum(np.square(np.abs(frame)), axis=4, keepdims=True) )
    #print(out[1])
    return out
# if 'transposition' in recopart:
def reco_transposition():
    pass



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