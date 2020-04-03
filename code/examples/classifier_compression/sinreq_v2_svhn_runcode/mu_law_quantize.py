from __future__ import division
import numpy as np
import tensorflow as tf
import os
import pickle
import math 
from math import exp, expm1


# FUN = performs uniform quantization 
def uniform_quant(x,x_max,x_min,nbits,type=0):
    L = 2**nbits
    range = x_max-x_min
    q = range/L  # step size 
    if (type):
        beta = np.floor((x - x_min)/q) # index of the quantization level 
        xq = beta*q + q/2 + x_min
    else:
        beta = np.round((x - x_min)/q) # index of the quantization level 
        xq = beta*q + x_min
    return xq


'''===================================================================
# FUNCTION VERSION IN MATLAB
% function [xq,q,beta]=uniform_quant(x,x_max,x_min,nbits,type);
%     L = 2^nbits;
%     range = x_max - x_min;
%     q = range/L; % step size 
%     if (type)
%         beta = floor((x - x_min)/q); % index of the quantization level 
%         xq = beta*q + q/2 + x_min;
%     else 
%         beta = round((x - x_min)/q); % index of the quantization level 
%         xq = beta*q + x_min;
%     end 
% end 
==================================================================='''

# FUN = performs mu_law quantization 
def mu_law_quantize(x,nbits,mu,Xmax):

    # load weights as numpy array / or list?!
    y=Xmax*(np.log(1+mu*np.absolute(x)/Xmax)/np.log(1+mu))*np.sign(x)
    # uniform quantization using nbits quantizer
    yq=uniform_quant(y,Xmax,-Xmax,nbits,1)

    # mu law expand yq[n]
    xq=(np.exp(np.absolute(yq)*np.log(1+mu)/Xmax)-1)*Xmax/mu
    xq=xq*np.sign(yq);

    # compute error signal
    eq=np.subtract(xq,x)
    return xq

# FUN = calculates the SNR and ERROR between orginal and quantized signals 
def snr(xh,x):
    # compute snr using standard methods
    s_n_r=10*np.log10(np.sum(np.square(x))/np.sum(np.square(np.subtract(x,xh))))
    
    # compute error signal as difference of unquantized and quantized signals
    e=np.subtract(xh,x)
    mse = np.mean(np.square(e))
    return s_n_r, mse


