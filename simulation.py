import numpy as np
from fbm import FBM

T = 1.0

def binomial_cascade(n, m):
    N = 2**n
    measure = np.ones(N)
    for level in range(n):
        step = 2**(n - level - 1)
        for i in range(0, N, 2*step):
            multiplier = np.random.choice([m, 2-m], size=step)
            measure[i:i+step] *= multiplier
            measure[i+step:i+2*step] *= (2 - multiplier)
    theta = np.cumsum(measure)
    theta = theta / theta[-1] * T
    return theta

def generate_fbm(n, H):
    N = 2**n
    f = FBM(n=N, hurst=H, length=T, method='hosking')
    b = f.fbm()
    return b

def multifractal_returns(n, H, m):
    theta = binomial_cascade(n, m)
    b = generate_fbm(n, H)
    returns = np.interp(theta, np.linspace(0, T, len(b)), b)
    return theta, returns
