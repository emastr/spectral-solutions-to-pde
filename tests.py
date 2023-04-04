from implemented import *
from notimplemented import *

def test_get_L():
    """Test the derivative of the Fourier transform."""
    ns = [2, 4, 8, 16, 32, 64, 128]
    er = []
    for n in ns:
        k = 2 * np.pi
        g = lambda x,y: np.exp(np.sin(k*x) + np.sin(k*y))
        Lg = lambda x,y: g(x,y) * k**2 * (np.cos(k*x)**2 - np.sin(k*x) + np.cos(k*y)**2 - np.sin(k*y))
        
        lam = 1.
        L = get_L(*get_freq(n), lam)
        g_disc = discretize(g, n)
        Lg_apx = invFourier2D(L * fourier2D(g_disc))
        
        er.append(np.linalg.norm(Lg_apx - discretize(Lg, n)) / n)
    
    plt.semilogy(ns, er, 'o-')
    plt.title('L2 error of Laplace operator')
    plt.xlabel('Number of basis functions')
    plt.ylabel('L2 error')
    plt.show()
    

def test_get_b():
    k = 2 * np.pi
    f = lambda x,y: np.exp(np.sin(k*x) + np.sin(k*y))
    bx = lambda x,y: np.sin(k*x)
    by = lambda x,y: np.sin(k*y)
    u = lambda x,y: np.exp(np.sin(k*x) * np.sin(k*y))
    dxu = lambda x,y: k * np.cos(k*x) * np.sin(k*y) * u(x,y)	
    dyu = lambda x,y: k * np.sin(k*x) * np.cos(k*y) * u(x,y)
    b = lambda x,y: f(x,y) - (bx(x,y) * dxu(x,y) + by(x,y) * dyu(x,y))
    
    ns = [8, 16, 32, 64, 128]
    er = []
    
    for n in ns:    
        b_disc = discretize(b, n)
        b_apx = invFourier2D(get_b(fourier2D(discretize(u, n)), fourier2D(discretize(f, n)), 
                            discretize(bx, n), discretize(by, n), *get_freq(n)))
    
        er.append(np.linalg.norm(b_disc - b_apx) / n)
    plt.semilogy(ns, er, 'o-')
    plt.title('L2 error of b(u)')
    plt.xlabel('Number of basis functions')
    plt.ylabel('L2 error')
    plt.show()
    

def test_euler():
    # Test the accuracy of the Euler method
    # By solving a simple ODE u' - (-u) = 0, u(0) = 1
    # Which has the solution u(t) = exp(-t).
    
    Ns = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512]).astype(float)
    er = []
    T = 1.
    for N in Ns:
        u = euler_solve(np.array([1.]), np.array([-1.]), lambda u: np.array([0.]), int(N), T)
        er.append(np.abs(u[0] - np.exp(-T)))
        
    plt.title("Test of Euler method for u' - (-u) = 0, u(0) = 1")
    plt.loglog(Ns, er, 'o-', label='error at time T=1')
    plt.plot(Ns, Ns**(-1), '--', label='O(N^-1)')
    plt.legend()
    plt.show()
    

def test_solve():
    t_vec = []
    u_vec = []
    N_plot = 16
    N = N_plot * 100 + 1

    def save_callback(u_four, t, n):
        """Save the solution at time t, and the time t itself."""
        if n % (N // N_plot) == 0:
            t_vec.append(t)
            u_vec.append(invFourier2D(u_four))

    g = lambda x,y: np.exp(np.sin(11 * np.pi * x) + np.sin(11 * np.pi * y))
    bx = lambda x, y: np.sin(2 * np.pi * y)  # 3 is nice setting for the test
    by = lambda x, y: -np.sin(2 * np.pi * x)
    f  = lambda x, y: np.sin(np.pi * 2 * x * y)

    #test_derivative(20)
    K = 200
    lam = 0.0005
    print(f"CFL condition: {N / K**2}")
    solve(g, f, bx, by, lam, N, 1, K, callback=save_callback)
    plt.figure(figsize=(20,20))
    plot_2d([discretize(g, K)] + u_vec, vmin=0., vmax=7., cmap="inferno")
    

def test_plot_2d():
    Kx, Ky = get_freq(10)
    plot_2d([Kx, Ky])



