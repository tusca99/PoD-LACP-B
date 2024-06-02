import numba as nb
import numpy as np


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
    g = np.random.normal(loc=0.0, scale=dt**0.5, size=(len(X), 3)) 
    result = np.empty((len(X), 3))
    for i in range(len(X)):
        result[i, 0] = np.sqrt(2*D1[0]) * g[i,0]
        result[i, 1] = np.sqrt(2*D1[1]) * g[i,1]
        result[i, 2] = np.sqrt(2*D2) * g[i,2]
        #print('g1:',result[i, 0])
        #print('g2:',result[i, 1])
        #print('g3:',result[i, 2])
    return result

@nb.njit(fastmath=True)
def force(X, d, μ, k, τ):
    result = np.empty((len(X), 3))
    for i in range(len(X)):
        x = X[i]
        result[i, 0] = μ[0] * (-k[0] * x[0] + d*x[1]) #x update
        result[i, 1] = μ[1] * (-k[1] * x[1] + d*x[0] + x[2])  #y update
        #result[i, 1] = μ[1] * (-k[1] * x[1] + d*x[0])  #y update
        result[i, 2] = -x[2] / τ #fy update
        #print('x:',result[i, 0])
        #print('y:',result[i, 1])
        #print('f:',x[2])
    return result

@nb.njit(fastmath=True)
def dx(state, dt, d, μ, k, τ, D1, D2):
    Fs = force(state, d, μ, k, τ)
    Ds = D(state, D1, D2, dt)
    #print('Ds:',Ds)
    #print('Fs:', Fs)
    return dt * Fs + Ds


@nb.njit(fastmath=True)
def simulate(tlist, initial_position, oversampling, prerun, d, μ, k, τ, D1, D2, dt):
    state = initial_position.copy()
    dto = dt / oversampling
    
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
            #print(dx)
        data[i, :, :] = state
        forces[i, :, :] = force(state, d, μ, k, τ) #!!!
    return data, forces

@nb.njit(fastmath=True)
def entropy(x, y, fx, fy, tlist, dt, D1):
    dx = x[1:]-x[:-1]
    dy = y[1:]-y[:-1]
    Fsx=(fx[1:]+fx[:-1])/2
    Fsy=(fy[1:]+fy[:-1])/2
    S=np.sum(Fsx*dx)/(D1[0]*(tlist[-1]-1)*dt)+np.sum(Fsy*dy)/(D1[1]*(tlist[-1]-1)*dt)
    return S

@nb.njit(fastmath=True)
def simulator_sbi(pars, dt, oversampling, prerun, Npts, prefix = None, d=0, k=1): 
    dim = 3
    k = np.array([1,1])
    μ =np.array([1,1])
    kBT = 1
    D1 = kBT * μ
    ϵ = pars[0]
    τ = pars[1]
    D2 = ϵ**2 / τ
    # Force field parameters (stochastic Lorenz process)
    d = np.float64(pars[2])
    tau = dt * Npts
    tlist = np.linspace(0. + dt/2., tau, Npts)
    initial_position = np.zeros(shape=(1,dim), dtype=np.float64) #(x,f) pairs

    data, forces = simulate(tlist, initial_position=initial_position, oversampling=oversampling, 
                            prerun=prerun, d=d, μ=μ, k=k, τ=τ, D1=D1, D2=D2, dt=dt)
    x = data[:, 0, 0]
    y = data[:, 0, 1]
    f_int_x = forces[:, 0, 0] / μ[0] + k[0] * x
    f_int_y = forces[:, 0, 1] / μ[1] + k[1] * y

    #dt = tlist[1] - tlist[0]
    Cyy = corr(y, y, len(tlist), dt=dt)
    Cxx = corr(x, x, len(tlist), dt=dt)
    Cff = corr(f_int_x, f_int_y, len(tlist), dt=dt)
    Cfx = corr(f_int_x, x, len(tlist), dt=dt)
    Cfy = corr(f_int_y, y, len(tlist), dt=dt)

    S2 = cumulative_trapezoid(Cfx - Cfx[0], x=tlist)
    S3 = cumulative_trapezoid(Cfx, x=tlist)
    S3 = -μ[1]*k[1]*cumulative_trapezoid(S3, x=tlist)
     
    S2_y = cumulative_trapezoid(Cfy - Cfy[0], x=tlist)
    S3_y = cumulative_trapezoid(Cfy, x=tlist)
    S3_y = -μ[1]*k[1]*cumulative_trapezoid(S3, x=tlist)
    
    t_corr = 10
    idx_corr = np.where(tlist < t_corr)[0]
    S_red_x = 1 - (S2[idx_corr] + S3[idx_corr]) / (kBT * tlist[idx_corr])
    S_red_y = 1 - (S2_y[idx_corr] + S3_y[idx_corr]) / (kBT * tlist[idx_corr])

    npcombo = np.stack((Cxx[idx_corr],Cyy[idx_corr], S_red_x[idx_corr], S_red_y[idx_corr]))
    return npcombo


@nb.njit(fastmath=True)
def simulator_entropy(pars, dt, oversampling, prerun, Npts, prefix = None): 
    dim = 3
    k = np.array([1,1])
    μ =np.array([1,1])
    kBT = 1
    D1 = kBT * μ
    ϵ = pars[0]
    τ = pars[1]
    D2 = ϵ**2 / τ
    # Force field parameters (stochastic Lorenz process)
    d = np.float64(pars[2])
    tau = dt * Npts
    tlist = np.linspace(0. + dt/2., tau, Npts)
    initial_position = np.zeros(shape=(1,dim), dtype=np.float64)
    #initial_position =[[200.,-200.,0.]] #(x,y,f) pairs
    print(initial_position)
    data, forces = simulate(tlist, initial_position=initial_position, oversampling=oversampling, 
                            prerun=prerun, d=d, μ=μ, k=k, τ=τ, D1=D1, D2=D2, dt=dt)
    
    x = data[:, 0, 0]
    y = data[:, 0, 1]
    #print(d)
    f_int = forces[:, 0, 1] / μ[1] + k[1] * y
    #print(x)

    
    S = entropy(x, y, forces[:, 0, 0]/μ[0], forces[:, 0, 1]/μ[1], tlist, dt, D1)
    #S=entropy2(data, tlist, dt, d, μ, k, τ, D1, D2)
    return S

@nb.njit(fastmath=True)
def simulator_sbi_entropy(pars, dt, oversampling, prerun, Npts, features = None, d=0, k=1): 
    dim = 3
    k = np.array([1,1])
    μ =np.array([1,1])
    kBT = 1
    D1 = kBT * μ
    ϵ = pars[0]
    τ = pars[1]
    D2 = ϵ**2 / τ
    # Force field parameters (stochastic Lorenz process)
    d = np.float64(pars[2])
    tau = dt * Npts
    tlist = np.linspace(0. + dt/2., tau, Npts)
    initial_position = np.zeros(shape=(1,dim), dtype=np.float64) #(x,f) pairs

    data, forces = simulate(tlist, initial_position=initial_position, oversampling=oversampling, 
                            prerun=prerun, d=d, μ=μ, k=k, τ=τ, D1=D1, D2=D2, dt=dt)
    x = data[:, 0, 0]
    y = data[:, 0, 1]
    f_int_x = forces[:, 0, 0] / μ[0] + k[0] * x
    f_int_y = forces[:, 0, 1] / μ[1] + k[1] * y

    #dt = tlist[1] - tlist[0]
    Cyy = corr(y, y, len(tlist), dt=dt)
    Cxx = corr(x, x, len(tlist), dt=dt)
    Cff = corr(f_int_x, f_int_y, len(tlist), dt=dt)
    Cfx = corr(f_int_x, x, len(tlist), dt=dt)
    Cfy = corr(f_int_y, y, len(tlist), dt=dt)

    S2 = cumulative_trapezoid(Cfx - Cfx[0], x=tlist)
    S3 = cumulative_trapezoid(Cfx, x=tlist)
    S3 = -μ[1]*k[1]*cumulative_trapezoid(S3, x=tlist)
     
    S2_y = cumulative_trapezoid(Cfy - Cfy[0], x=tlist)
    S3_y = cumulative_trapezoid(Cfy, x=tlist)
    S3_y = -μ[1]*k[1]*cumulative_trapezoid(S3, x=tlist)
    
    t_corr = 10
    idx_corr = np.where(tlist < t_corr)[0]
    S_red_x = 1 - (S2[idx_corr] + S3[idx_corr]) / (kBT * tlist[idx_corr])
    S_red_y = 1 - (S2_y[idx_corr] + S3_y[idx_corr]) / (kBT * tlist[idx_corr])

    S = entropy(x, y, forces[:, 0, 0]/μ[0], forces[:, 0, 1]/μ[1], tlist, dt, D1) 
    
    # Determine the total length needed for the concatenated array
    total_length = 0
    if features:
        for feature in features:
            if feature == "Cxx":
                total_length += len(Cxx[idx_corr])
            elif feature == "Cyy":
                total_length += len(Cyy[idx_corr])
            elif feature == "S_red_x":
                total_length += len(S_red_x)
            elif feature == "S_red_y":
                total_length += len(S_red_y)

    # Pre-allocate the concatenated array
    npcombo = np.zeros(total_length, dtype=np.float64)

    # Copy selected features into the pre-allocated array
    current_index = 0
    if features:
        for feature in features:
            if feature == "Cxx":
                npcombo[current_index:current_index + len(Cxx[idx_corr])] = Cxx[idx_corr]
                current_index += len(Cxx[idx_corr])
            elif feature == "Cyy":
                npcombo[current_index:current_index + len(Cyy[idx_corr])] = Cyy[idx_corr]
                current_index += len(Cyy[idx_corr])
            elif feature == "S_red_x":
                npcombo[current_index:current_index + len(S_red_x)] = S_red_x
                current_index += len(S_red_x)
            elif feature == "S_red_y":
                npcombo[current_index:current_index + len(S_red_y)] = S_red_y
                current_index += len(S_red_y)

    S_analytic = (μ[1]*ϵ**2) / ((1 + k[1]*μ[1]*τ) - (d**2*μ[0]*μ[1]*τ**2) / (1 + k[0]*μ[0]*τ))
    
    return npcombo, S, S_analytic