import numba as nb
import numpy as np
#!pip install numba icc_rt rocket-fft


@nb.njit(fastmath=True)
def cumulative_trapezoid(y, x):
    """
    Compute the cumulative trapezoidal integral of y with respect to x.

    Parameters:
    y : numpy.ndarray
        Array of y values.
    x : numpy.ndarray
        Array of x values (must have the same shape as y).

    Returns:
    numpy.ndarray
        Array of cumulative trapezoidal integrals.
    """
    integral = np.zeros_like(y)
    for i in range(1, len(x)):
        integral[i] = integral[i-1] + 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return integral


@nb.njit
def jit_fft(x,f):
    return np.fft.fft(x,f)

@nb.njit
def jit_fftfreq(x,d):
    return np.fft.fftfreq(x,d)

@nb.njit
def jit_ifft(x):
    return np.fft.ifft(x)

@nb.njit(fastmath=True)
def corr(x,y,nmax,dt=False):
    assert len(x)==len(y)

    n=len(x)
    fsize=2**int(np.ceil(np.log2(2*n-1)))

    xp=x-np.mean(x)
    yp=y-np.mean(y)

    cfx=jit_fft(xp,fsize)
    cfy=jit_fft(yp,fsize)
    
    if dt != False:
        freq = jit_fftfreq(n, d=dt)
        idx = np.where((freq<-1/(2*dt))+(freq>1/(2*dt)))[0]
        cfx[idx]=0
        cfy[idx]=0
        
    sf=cfx.conjugate()*cfy
    corr = jit_ifft(sf).real / n

    return corr[:nmax]
   
@nb.njit(fastmath=True)
def D(X, D1, D2, dt):
    #g = np.random.normal(0, dt**0.5) #random uguale per posizioni e forze scale=dt**0.5
    g = np.random.normal(loc=0.0, scale=dt**0.5, size=(len(X), 2)) 
    result = np.empty((len(X), 2))
    for i in range(len(X)):
        result[i, 0] = np.sqrt(2*D1)*g[i,0]
        result[i, 1] = np.sqrt(2*D2)*g[i,1]
    return result
    
@nb.njit(fastmath=True)
def force(X, d, μ, k, τ):
    result = np.empty((len(X), 2))
    for i in range(len(X)):
        x = X[i]
        result[i, 0] = μ * (-k * x[0] - d * x[0]**3+ x[1]) #x update
        result[i, 1] = -x[1] / τ #fx update
    return result

@nb.njit(fastmath=True)
def dx(state, dt, d, μ, k, τ, D1, D2):
    Fs =force(state, d, μ, k, τ)
    Ds=D(state, D1, D2, dt)
    """
    Ds=np.empty((Fs.shape[0], 2))
    for i in range(Fs.shape[0]):
        Ds[i,0]= np.sqrt(2*D1) * random_values[i] 
        Ds[i,1]=np.sqrt(2*D2) * random_values[i]
            """

    return dt * Fs + Ds

@nb.njit(fastmath=True)
def simulate(tlist, initial_position, oversampling, prerun, d, μ, k, τ, D1, D2, dt):
    state = initial_position.copy()
    dto = dt / oversampling
    
    #random_values =  np.random.normal(0, dt**0.5, len(tlist) * oversampling) 

    # Pre-equilibration
    for j in range(prerun * oversampling):
        state += dx(state, dt, d, μ, k, τ, D1, D2)
    
    # Start recording
    data = np.zeros((len(tlist),state.shape[0],state.shape[1])) 
    forces = np.zeros((len(tlist),state.shape[0],state.shape[1]))
    for i in range(len(tlist)):
        data[i, :, :] = state
        forces[i, :, :] = force(state, d, μ, k, τ)
        for j in range(oversampling):
            state += dx(state, dto, d, μ, k, τ, D1, D2)
        data[i, :, :] = state

    #data[-1, :, :] = state
    #forces[-1, :, :] = force(state, d, μ, k, τ) 
    return data, forces

@nb.njit(fastmath=True)
def entropy(x, f, tlist, dt, D1):
    dx = x[1:]-x[:-1]
    #print(dx)
    #print(x[:-1])
    Fs=(f[1:]+f[:-1])/2
    #print(Fs)
    S=np.sum(Fs*dx)/(D1*(tlist[-1]-1)*dt)
    return S


@nb.njit(fastmath=True)
def simulator_sbi_entropy(pars, dt, oversampling, prerun, Npts, prefix = None, d=0, k=1): 

    μ = 1
    kBT = 1
    D1 = kBT * μ
    ϵ = np.float64(pars[0])
    τ = np.float64(pars[1])
    D2 = ϵ**2 / τ
    # Force field parameters (stochastic Lorenz process)
    d = np.float64(pars[2])
    tau = dt * Npts
    tlist = np.linspace(0. + dt/2., tau, Npts)
    initial_position = np.zeros(shape=(1,2), dtype=np.float64) #(x,f) pairs

    data, forces = simulate(tlist, initial_position=initial_position, oversampling=oversampling, 
                            prerun=prerun, d=d, μ=μ, k=k, τ=τ, D1=D1, D2=D2, dt=dt)
    x = data[:, 0, 0]
    f_int = (forces[:, 0, 0] / μ + k * x)
    
    #dt = tlist[1] - tlist[0]
    Cxx = corr(x, x, len(tlist), dt=dt)
    Cfx = corr(f_int, x, len(tlist), dt=dt)
     
    S2 = cumulative_trapezoid(Cfx - Cfx[0], x=tlist)
    S3 = cumulative_trapezoid(Cfx, x=tlist)
    S3 = -μ*k*cumulative_trapezoid(S3, x=tlist)
    
    t_corr = 10
    idx_corr = np.where(tlist < t_corr)[0]
    S_red = 1 - (S2[idx_corr] + S3[idx_corr]) / (kBT * tlist[idx_corr])
   
    npcombo = np.concatenate((Cxx[idx_corr], S_red[idx_corr]))
    
    S = entropy(x, forces[:, 0, 0], tlist, dt, D1)
    
    if prefix == 'corr' : 
        return Cxx[idx_corr] , S
    if prefix == 'svr' :
        return S_red[idx_corr] , S
    if prefix == 'full' : 
        return npcombo, S


@nb.njit(fastmath=True)
def simulator_sbi(pars, dt, oversampling, prerun, Npts, d=0, k=1): 

    μ = 1
    kBT = 1
    D1 = kBT * μ
    ϵ = np.float64(pars[0])
    τ = np.float64(pars[1])
    D2 = ϵ**2 / τ
    # Force field parameters (stochastic Lorenz process)
    d = np.float64(pars[2])
    tau = dt * Npts
    tlist = np.linspace(0. + dt/2., tau, Npts)
    initial_position = np.zeros(shape=(1,2), dtype=np.float64) #(x,f) pairs

    data, forces = simulate(tlist, initial_position=initial_position, oversampling=oversampling, 
                            prerun=prerun, d=d, μ=μ, k=k, τ=τ, D1=D1, D2=D2, dt=dt)
    x = data[:, 0, 0]
    f_int = (forces[:, 0, 0] / μ + k * x)
    
    Cxx = corr(x, x, len(tlist), dt=dt)
    Cfx = corr(f_int, x, len(tlist), dt=dt)
    
    S2 = cumulative_trapezoid(Cfx - Cfx[0], x=tlist)
    S3 = cumulative_trapezoid(Cfx, x=tlist)
    S3 = -μ*k*cumulative_trapezoid(S3, x=tlist)
    
    t_corr = 10
    idx_corr = np.where(tlist < t_corr)[0]
    S_red = 1 - (S2[idx_corr] + S3[idx_corr]) / (kBT * tlist[idx_corr])
    
    npcombo = np.stack((Cxx[idx_corr], S_red[idx_corr]))
    
    return npcombo