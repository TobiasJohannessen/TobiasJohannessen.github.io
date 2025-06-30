import numpy as np
import scipy.integrate as spi



def gaussian(x):
    return np.exp(-x**2)

def cauchy(x):
    return 1 /  (1 + x**2)

def exponential(x):
    return np.exp(-x)



g_value, g_error, info = spi.quad(gaussian, -np.inf, np.inf, epsabs=1e-4, epsrel=1e-4, full_output=True)
value, error, gN_evals = g_value, g_error, info['neval']

c_value, c_error, info = spi.quad(cauchy, -np.inf, 0, epsabs=1e-4, epsrel=1e-4, full_output=True)
cN_evals = info['neval']

e_value, e_error, info = spi.quad(exponential, 0, np.inf, epsabs=1e-4, epsrel=1e-4, full_output=True)
eN_evals = info['neval']



with open('out.txt', 'a') as f:
    f.write(f'\n\n\n -----------PYTHON SCRIPT OUTPUT-----------\n\n')
    f.write(f'Using scipy.integrate.quad:\n\n')
    f.write(f'Gaussian integral: {g_value} with error {g_error} and {gN_evals} evaluations\n')
    f.write(f'The actual error is {np.abs(g_value - np.sqrt(np.pi))}\n\n')
    f.write(f'Cauchy integral: {c_value} with error {c_error} and {cN_evals} evaluations\n')
    f.write(f'The actual error is {np.abs(c_value - np.pi/2.)}\n\n')
    f.write(f'Exponential integral: {e_value} with error {e_error} and {eN_evals} evaluations\n')
    f.write(f'The actual error is {np.abs(e_value - 1)}\n\n')
    f.write(f'\n\n')

    f.write(f' It seems that the estimated error is overestimated for all three integrals as the actual error seems to be significantly smaller.\n')
    f.write(f' The number of evaluations is also 3-5 times bigger than the solver made in C++.\n')
    f.write
    
