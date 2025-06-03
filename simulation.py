import numpy as np
from fbm import FBM

T = 1.0

def binomial_cascade(n, m):
    """
    n is the depth of the cascade
    m is like a weight of the randomness in the model

    Cascading unevenly splits data at each level over n times.
    Divide the measure into blocks of 2*step size.
    At each block split the level in half, either assign m or 2-m to one half, then the other choice to the other half.
    Multiply the sections to get the values.

    Normalize the time.
    """
    N = 2**n
    measure = np.ones(N)
    for block in range(n):
        step = 2**(n - block - 1)
        for i in range(0, N, 2*step):
            multiplier = np.random.choice([m, 2-m], size=step)
            measure[i:i+step] *= multiplier
            measure[i+step:i+2*step] *= (2 - multiplier)
    theta = np.cumsum(measure)
    theta = theta / theta[-1] * T
    return theta

def generate_fbm(n, H):
    """
    Retrive 2**n data points of Fractional Brownian Motion with Hurst Exponent H
    """
    N = 2**n
    f = FBM(n=N, hurst=H, length=T, method='hosking')
    b = f.fbm()
    return b

def multifractal_returns(n, H, m):
    """
    Interpolate the fBm values over distored time.
    """
    theta = binomial_cascade(n, m)
    b = generate_fbm(n, H)
    returns = np.interp(theta, np.linspace(0, T, len(b)), b)
    return theta, returns
