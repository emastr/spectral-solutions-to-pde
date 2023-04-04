import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple

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
    


def euler_solve(u0: MAT, L: MAT, b: Callable[[MAT], MAT], 
                N: int, T: float, callback: Callable=None) -> np.ndarray:
    """Solve the Matrix ODE u' - L*u = b(u) with initial condition u(0) = u0.
    u is a matrix, L is a matrix that is multiplied elementwise with u, b is a matrix valued function of u, 
    N is the number of times steps, T is the final time. Solution is returned as a matrix.
    """
    u = u0
    dt = T / N
    t = np.linspace(0, T, N)
    for n in range(1, N):
        u = euler_step(u, L, b, dt)
        if callback is not None:
            callback(u, n*dt, n)
    return u


def euler_step(u: MAT, L: MAT, b: Callable[[MAT], MAT], dt: float) -> MAT:
    """One step of the Euler Backwards method. u is a matrix,
    L is a matrix, b is a matrix valued function of u, dt is the step size."""
    return (u + dt * b(u)) / (1 - dt * L)


def get_L(Kx: MAT, Ky: MAT, lam: float) -> MAT:
    """Return the Laplace operator L*u = lam * (du/dx^2 + du/dy^2),
    represented as a matrix (frequency domain) with same dimensions as the fourier coefficients Kx and Ky.
    To evaluate the Laplace operator in the real domain, use invFourier2D(L * fourier2D(u))."""
    return lam * (Kx**2 + Ky**2)


def get_b(u_four: MAT, f_four: MAT, bx_real: MAT, by_real: MAT, Kx: MAT, Ky: MAT) -> MAT:
    """Return the fourier coefficients of f - (bx, by) dot grad(u),
     for a grid of size n x n. f_four is the fourier coefficients of f,
    bx_real and by_real are the real space values of bx and by, and 
    Kx and Ky are the fourier frequencies."""
    dxu_real = invFourier2D(Kx * u_four)
    dyu_real = invFourier2D(Ky * u_four)
    return f_four - fourier2D(bx_real * dxu_real + by_real * dyu_real)
    
    
def solve(u0: Callable[[MAT, MAT], MAT], f: Callable[[MAT, MAT], MAT], 
          bx: Callable[[MAT, MAT], MAT], by: Callable[[MAT, MAT], MAT], 
          lam: float, N: int, T: float, K: int, callback=None) -> MAT:
    """Solve the ODE u' - lam * laplace(u) + (bx, by) dot grad(u) = f
    with initial condition u(0) = u0. f is a function of u, T is the final time.
    N is the number of time steps and K is the number of Fourier bases.
    """
    
    bx_real = discretize(bx, K)
    by_real = discretize(by, K)
    Kx, Ky = get_freq(K)
    u0_four = fourier2D(discretize(u0, K))
    f_four = fourier2D(discretize(f, K))
    L_four = get_L(Kx, Ky, lam)
    b_four = lambda u: get_b(u, f_four, bx_real, by_real, Kx, Ky)
    
    u_freq  = euler_solve(u0_four, L_four, b_four, N, T, callback=callback)
    return invFourier2D(u_freq)



def plot_2d(f_list: List[MAT], **kwargs):
    """Plot a list of 2D functions. f_list is a list of 2D arrays.
     kwargs are passed to plt.imshow (plt.imshow([], **kwargs)). The functions are plotted in a grid."""
    n = int(np.ceil(np.sqrt(len(f_list))))
    for i,f in enumerate(f_list):
        plt.subplot(n, n, i + 1)
        plt.imshow(f, **kwargs)


def stream_plot(bx, by):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)
    bx = bx(x, y)
    by = by(x, y)
    plt.streamplot(x, y, bx, by, color=np.sqrt(bx**2 + by**2), cmap='Blues')
        
