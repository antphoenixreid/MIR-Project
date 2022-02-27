import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import get_window
import math

tol = 1e-14

# Class DFT
class MusicProcessor:
    def __init__(self,x,w='hamming',M=511,N=1024):
        self.x = x
        self.w = get_window(w,M)
        self.M = M
        self.N = N
        
    def DFT(self,isLog=True):
        hN = (self.N//2) + 1
        hM1 = (self.M + 1)//2
        hM2 = self.M//2
        
        w = self.w/sum(self.w)
        x_win = self.x*w
        fftbuffer = np.zeros(self.N)
        fftbuffer[:hM1] = x_win[hM2:]
        fftbuffer[-hM2:] = x_win[:hM2]
        
        X = fft(fftbuffer)
        absX = abs(X[:hN])
        absX[absX<np.finfo(float).eps] = np.finfo(float).eps
        
        if isLog:
            mX = 20*np.log10(absX)
        else:
            mX = absX
            
        X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0
        X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0
        pX = np.unwrap(np.angle(X[:hN]))
        
        return mX,pX
    
    def IDFT(self,mX,pX):
        hN = mX.size
        
        hM1 = int(math.floor((self.M + 1)/2))
        hM2 = int(math.floor(self.M/2))
        
        Y = np.zeros(self.N,dtype=complex)
        Y[:hN] = 10**(mX/20)*np.exp(1j*pX)
        Y[hN:] = 10**(mX[-2:0:-1]/20)*np.exp(-1j*pX[-2:0:-1])
        
        fftbuffer = np.zeros(self.N)
        fftbuffer = np.real(ifft(Y))
        
        y = np.zeros(self.M)
        y[:hM2] = fftbuffer[-hM2:]
        y[hM2:] = fftbuffer[:hM1]
        
        return y