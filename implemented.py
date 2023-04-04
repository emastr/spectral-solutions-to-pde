import numpy as np
from numpy.fft import fft, ifft, fftfreq
from typing import Callable, Tuple

MAT = np.ndarray

def discretize(f: Callable[[MAT, MAT], MAT], n: int) -> MAT:
    """Evaluate the function f. f is a function of two variables,
     and is evaluated at n x n points uniformly spaced on [0,1]^2."""
    x = np.linspace(0, 1, n + 1)[:-1]
    y = np.linspace(0, 1, n + 1)[:-1]
    X, Y = np.meshgrid(x, y)
    return f(X, Y)

def fourier2D(f: MAT) -> MAT:
    """Compute the 2D Fourier transform of f."""
    return fft(fft(f, axis=0), axis=1) / f.shape[0] ** 2
    
    
def invFourier2D(coef: MAT) -> MAT:
    """Evaluate Fourier series with coefficients coef at points
    uniformly spaced on [0,1]^2. Result has same shape as coef."""
    return np.real(ifft(ifft(coef, axis=0), axis=1)) * coef.shape[0] ** 2


def get_freq(n: int) -> Tuple[MAT, MAT]:
    """Returns two n x n matrices Kx and Ky, with the frequencies
    for each fourier coefficient in a grid of size n x n."""
    freq = 2j * np.pi * fftfreq(n) * n
    return np.meshgrid(freq, freq)
    
