# %%
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import xarray as xr
from scipy.io import loadmat
from tqdm import tqdm
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import pandas as pd
from multiprocessing import Pool
import sys

# %%
#HERE I DEFINE THE FUNCTION FOR THE gA 

def gA (pres,sol_zen):
    #pressure in Pa. solar zenith angle in degreea
    d2r=np.pi/180
    logp85corr=np.array([-7,
                        -5.51677832e+00, -5.47384432e+00, -5.43035414e+00, -5.38628065e+00,
                           -5.34159539e+00, -5.29626675e+00, -5.25026155e+00, -5.20354361e+00,
                           -5.15607393e+00, -5.10780999e+00, -5.05870658e+00, -5.00871335e+00,
                           -4.95777577e+00, -4.90583479e+00, -4.85282478e+00, -4.79867481e+00,
                           -4.74330466e+00, -4.68662766e+00, -4.62854591e+00, -4.56895088e+00,
                           -4.50771994e+00, -4.44471559e+00, -4.37978159e+00, -4.31273936e+00,
                           -4.24338432e+00, -4.17132701e+00, -4.09649172e+00, -4.01787775e+00,
                           -3.93488221e+00, -3.84698185e+00, -3.75373381e+00, -3.65477858e+00,
                           -3.54984201e+00, -3.43873830e+00, -3.32137346e+00, -3.19774914e+00,
                           -3.06796702e+00, -2.93223248e+00, -2.79085945e+00, -2.64426997e+00,
                           -2.49295172e+00, -2.33741388e+00, -2.17818109e+00, -2.01579135e+00,
                           -1.85079156e+00, -1.68373602e+00, -1.51517955e+00, -1.34567416e+00,
                           -1.17576692e+00, -1.00599335e+00, -8.36835949e-01, -6.68690824e-01,
                           -5.01857914e-01, -3.36553552e-01, -1.72910030e-01, -1.09823693e-02,
                            1.49248401e-01,  3.07869289e-01,  4.65037639e-01,  6.20978983e-01,
                            7.75935063e-01,  9.30116637e-01,  1.08370375e+00,  1.23684995e+00,
                            1.38969194e+00,  1.54234146e+00,  1.69489018e+00,  1.84741160e+00,
                            1.99995226e+00,  2.15253138e+00,  2.30514089e+00,  2.45774140e+00,
                            2.61026531e+00,  2.76260989e+00,  2.91464127e+00,  3.06619577e+00,
                            3.21681565e+00,  3.36683997e+00,  3.51585920e+00,  3.66374439e+00,
                            3.81038797e+00,  3.95570377e+00,  4.09962606e+00,  4.24211127e+00,
                            4.38313591e+00,  4.52269886e+00,  4.66077268e+00,  4.79739872e+00,
                            4.93268741e+00,  5.06672307e+00,  5.19961223e+00,  5.33148235e+00,
                            5.46248364e+00,  5.59278767e+00,  5.72258652e+00,  5.85207645e+00,
                            5.98144222e+00,  6.11085417e+00,  6.24046914e+00,  6.37042997e+00,
                            6.50086777e+00,  6.63189673e+00,  6.76361868e+00,  6.89612274e+00,
                            7.02948333e+00,  7.16377947e+00,  7.29910627e+00,  7.43558281e+00,
                            7.57334199e+00,  7.71254137e+00,  7.85335655e+00,  7.99598192e+00,
                            8.14063480e+00,  8.28755019e+00,  8.43698424e+00,  8.58921219e+00,
                            8.74451770e+00,  8.90288700e+00,  9.06399760e+00,  9.22751350e+00,
                            9.39309728e+00,  9.56041150e+00,  9.72911829e+00,  9.89887894e+00,
                            1.00693539e+01,  1.02402039e+01,  1.04110887e+01,  1.05816670e+01,
                            1.07515973e+01,  1.09205436e+01,  1.10882395e+01,  1.12545533e+01,
                            1.14194944e+01,  1.15832126e+01,  1.17459852e+01,  1.19080569e+01,
                            1.20694783e+01,  1.22300895e+01,  1.23895218e+01,  1.25472100e+01,
                            1.27025577e+01,  1.28551028e+01,  1.30045306e+01,  1.31506769e+01,
                            1.32935274e+01,  1.34332177e+01,  1.35700338e+01,  1.37044140e+01,
                            1.38369482e+01])
    gA85=np.array([6.01966695e-09,
                      6.01966695e-09, 6.02587341e-09, 6.03218597e-09, 6.03859833e-09,
                       6.04510289e-09, 6.05168540e-09, 6.05834330e-09, 6.06506815e-09,
                       6.07185745e-09, 6.07869647e-09, 6.08555677e-09, 6.09239934e-09,
                       6.09919568e-09, 6.10595726e-09, 6.11270653e-09, 6.11940893e-09,
                       6.12599011e-09, 6.13234353e-09, 6.13849983e-09, 6.14447565e-09,
                       6.15022153e-09, 6.15566047e-09, 6.16073693e-09, 6.16539715e-09,
                       6.16960128e-09, 6.17321804e-09, 6.17695239e-09, 6.17986846e-09,
                       6.18178900e-09, 6.18268123e-09, 6.18263075e-09, 6.18181491e-09,
                       6.18044779e-09, 6.17870756e-09, 6.17668143e-09, 6.17449626e-09,
                       6.17228934e-09, 6.17013865e-09, 6.16804327e-09, 6.16602142e-09,
                       6.16406432e-09, 6.16215159e-09, 6.16025809e-09, 6.15835221e-09,
                       6.15639631e-09, 6.15434861e-09, 6.15216000e-09, 6.14977859e-09,
                       6.14714626e-09, 6.14420110e-09, 6.14085091e-09, 6.13699440e-09,
                       6.13252618e-09, 6.12732723e-09, 6.12127318e-09, 6.11421388e-09,
                       6.10598205e-09, 6.09638051e-09, 6.08515375e-09, 6.07201408e-09,
                       6.05665884e-09, 6.03874578e-09, 6.01789083e-09, 5.99361562e-09,
                       5.96540859e-09, 5.93266855e-09, 5.89472001e-09, 5.85079398e-09,
                       5.80007682e-09, 5.74164332e-09, 5.67455148e-09, 5.59781436e-09,
                       5.51046272e-09, 5.41158287e-09, 5.30039017e-09, 5.17632952e-09,
                       5.03909803e-09, 4.88847600e-09, 4.72441570e-09, 4.54723454e-09,
                       4.35763107e-09, 4.15659391e-09, 3.94539467e-09, 3.72591842e-09,
                       3.49997101e-09, 3.26985614e-09, 3.03790245e-09, 2.80664703e-09,
                       2.57843685e-09, 2.35560644e-09, 2.14019809e-09, 1.93391038e-09,
                       1.73819559e-09, 1.55412510e-09, 1.38243550e-09, 1.22386922e-09,
                       1.07889518e-09, 9.47556249e-10, 8.29514941e-10, 7.24348205e-10,
                       6.31192667e-10, 5.49258359e-10, 4.77617578e-10, 4.15295219e-10,
                       3.61346281e-10, 3.14734252e-10, 2.74629039e-10, 2.40185833e-10,
                       2.10747943e-10, 1.85758850e-10, 1.64740457e-10, 1.47285856e-10,
                       1.33077259e-10, 1.21871725e-10, 1.13393656e-10, 1.07459137e-10,
                       1.04010969e-10, 1.03088945e-10, 1.04181492e-10, 1.06761555e-10,
                       1.10283829e-10, 1.14163747e-10, 1.17858983e-10, 1.20840878e-10,
                       1.22552573e-10, 1.22901409e-10, 1.21936774e-10, 1.20066944e-10,
                       1.17083445e-10, 1.13360779e-10, 1.08783137e-10, 1.03570401e-10,
                       9.75917826e-11, 9.11088954e-11, 8.39765842e-11, 7.61178745e-11,
                       6.79935241e-11, 6.17932461e-11, 6.17966997e-11, 6.21686621e-11,
                       6.17810163e-11, 5.90059283e-11, 5.41402204e-11, 4.69897160e-11,
                       3.96540264e-11, 3.32432305e-11, 2.83111959e-11, 2.46175330e-11,
                       2.17711145e-11])
    coeff=np.array([-3.90433229e-19,  1.99584747e-16, -4.47281751e-14,  5.77512092e-12,
           -4.74851914e-10,  2.59465839e-08, -9.54424055e-07,  2.34369889e-05,
           -3.73174981e-04,  3.64483013e-03, -1.98135679e-02,  5.00956221e-02,
           -3.95132483e-02])
    presscale=np.exp(np.polyval(coeff,sol_zen))
    g=np.interp(np.log(pres/np.cos(sol_zen*d2r)*presscale),logp85corr,gA85)
    if sol_zen > 90:
        g = 0.0
    return g


# %%
# CROSS SECTIONS
path = '/home/anqil/Documents/Python/Photochemistry/Pablo/'
data_sigma = loadmat(path+'sigma.mat')
irrad = data_sigma['irrad']            
sn2 = data_sigma['sN2']            
so = data_sigma['sO']        
so2 = data_sigma['sO2']    
so3 = data_sigma['sO3']
wave = data_sigma['wave']
# ABSORPTION CROSS SECTION FOR CO2
sco2 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
                 [10e-19],[5e-19],[4e-19],[3e-19],[2e-19],[1e-19],[1.2e-19],[2e-19],[1.4e-19],
                 [1.6e-19],[1.8e-19],[2e-19],[2.6e-19],[3.2e-19],[3.8e-19],[4.4e-19],[5e-19],
                 [5.6e-19],[6.2e-19],[7e-19],[8e-19],[8e-19],[8e-19],[8e-19],[8e-19],[8e-19],
                 [7.25e-19],[6.5e-19],[5.75e-19],[5e-19],[5.33e-19],[5.66e-19],[6e-19],[6e-19],
                 [6e-19],[4.75e-19],[3.5e-19],[2.5e-19],[1.5e-19],[1.125e-19],[0.75e-19],
                 [0.5e-19],[0.25e-19],[0.175e-19],[0.1e-19],[0.05e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],[0.0e-19],
                 [0.0e-19]])

# %%
# FUNCTION FOR THE SCHUMANN-RUNGE BANDS O2 ABSORPTION CROSS SECTION 

def effxsect(Temp, O2col):
    '''' Calculates the effective absorption Cross section 
        in the 17 spectral regions including the Schumann Runge Bands. 
        Input : is an array of temperatures (K) at the position of interest and
        the O2 column densities to the points mol cm^-2.
        Output: array  of 17 cross sections for given points
    '''
    Cheb_A=np.array([[ 1.033e-02,  5.132e-03, -3.009e-03, -2.003e-03, -8.843e-04,
        -3.824e-04, -3.312e-04, -1.675e-04,  5.279e-05,  5.043e-04,
         4.491e-04, -7.836e-06, -7.537e-05,  6.519e-05,  2.912e-05,
        -1.536e-04, -1.536e-04,  7.366e-05,  5.936e-05, -1.171e-04],
       [ 1.246e-02,  6.695e-03, -3.621e-03, -4.540e-03, -3.080e-03,
        -4.744e-04, -1.237e-04, -4.086e-04, -1.671e-04,  1.738e-04,
         4.295e-04,  3.581e-04, -1.913e-04, -2.113e-04,  2.907e-04,
         2.178e-05, -4.010e-04, -3.487e-05,  2.115e-04,  5.386e-06],
       [ 1.175e-02,  6.751e-03, -1.718e-03, -2.431e-03, -1.695e-03,
         1.556e-04,  5.667e-04,  3.655e-04, -2.062e-04, -5.546e-04,
         4.676e-05,  4.878e-04, -2.704e-05, -2.990e-04,  1.422e-04,
         1.008e-04, -2.067e-04,  7.060e-05,  1.314e-04, -1.855e-04],
       [ 6.001e-03,  3.042e-03, -1.346e-03, -3.106e-05,  5.446e-04,
         2.609e-06, -7.980e-05,  3.703e-04,  7.089e-05, -1.953e-04,
        -1.455e-04, -9.353e-05,  8.317e-05,  1.319e-04,  3.010e-06,
        -5.573e-05, -6.400e-05, -1.809e-05,  7.139e-05,  7.834e-05],
       [ 3.832e-03,  9.256e-04, -2.416e-03, -7.508e-04,  1.218e-04,
        -4.561e-05, -2.622e-04,  1.699e-05, -4.781e-05, -1.209e-04,
        -1.078e-05,  2.298e-05,  4.999e-05,  8.861e-05,  5.189e-05,
        -1.061e-04, -1.506e-04,  6.299e-05,  1.269e-04, -2.209e-05],
       [ 4.195e-03, -6.875e-05, -1.425e-03, -8.608e-04,  2.085e-04,
        -2.037e-04, -4.586e-04,  9.239e-05,  1.486e-04, -9.623e-05,
        -1.651e-04, -5.442e-05,  1.235e-04,  1.695e-04,  4.293e-05,
        -1.325e-04, -1.658e-04,  1.561e-05,  1.402e-04,  6.840e-05],
       [ 8.683e-03,  3.659e-03,  8.173e-04,  6.057e-04,  1.356e-03,
         5.945e-04, -3.003e-04, -4.622e-04, -4.735e-04, -2.480e-04,
        -8.935e-05, -1.078e-04,  1.236e-04,  3.665e-04,  2.077e-04,
        -1.372e-04, -3.188e-04, -1.725e-04,  1.986e-04,  3.317e-04],
       [ 6.617e-03, -2.154e-04,  1.405e-04, -1.179e-03,  6.122e-04,
        -6.014e-05, -6.300e-04, -6.661e-05, -5.149e-05, -1.009e-04,
        -1.795e-04, -2.051e-04,  1.201e-04,  3.396e-04,  1.421e-04,
        -9.847e-05, -2.104e-04, -1.657e-04,  1.180e-04,  2.606e-04],
       [ 5.455e-03, -3.623e-03, -1.365e-03, -3.610e-03, -6.176e-04,
        -4.358e-04, -8.037e-04, -2.747e-04, -1.566e-04,  6.109e-05,
         2.625e-05, -6.445e-05,  1.369e-04,  2.588e-04,  8.223e-05,
        -1.661e-04, -2.861e-04, -1.169e-04,  2.269e-04,  2.723e-04],
       [ 9.378e-03, -1.003e-03, -3.170e-04, -6.087e-04,  7.737e-04,
         3.806e-04,  5.263e-04,  3.687e-04, -5.599e-04, -5.481e-04,
        -1.703e-04, -1.498e-05,  1.344e-04,  2.162e-04,  1.604e-04,
        -4.670e-05, -2.666e-04, -1.295e-04,  2.006e-04,  2.186e-04],
       [ 3.731e-03, -2.148e-03, -1.250e-03, -4.274e-03, -1.693e-03,
        -7.912e-04, -5.688e-04, -1.375e-04, -1.717e-04,  2.300e-05,
        -3.813e-05, -1.837e-04,  4.132e-05,  2.408e-04,  1.046e-04,
        -1.125e-04, -2.054e-04, -8.391e-05,  1.875e-04,  2.339e-04],
       [ 4.250e-03, -4.023e-03,  2.523e-04, -4.041e-03, -6.125e-04,
        -5.337e-05, -2.020e-04,  1.908e-05, -4.512e-04, -1.253e-04,
         8.872e-05, -2.686e-05,  5.806e-05,  1.301e-04,  3.399e-05,
        -3.946e-05, -1.015e-04, -8.093e-05,  9.077e-05,  1.077e-04],
       [ 1.118e-02, -8.154e-03,  5.057e-03, -5.704e-03,  1.007e-03,
         8.534e-04, -5.180e-06,  5.501e-04, -7.834e-04, -2.877e-04,
         1.952e-04, -1.685e-05,  2.848e-05,  4.998e-05, -5.108e-05,
         2.857e-05, -3.484e-07, -6.914e-05,  8.822e-05,  2.924e-05],
       [ 1.222e-02, -8.487e-03,  4.967e-03, -4.590e-03,  9.812e-04,
         7.116e-04, -2.353e-04,  5.489e-04, -4.133e-04, -2.342e-04,
         4.735e-05, -8.489e-05,  1.307e-05,  8.056e-05, -1.496e-05,
         2.425e-05,  1.301e-05, -6.106e-05,  4.749e-05,  1.982e-05],
       [ 6.587e-03, -4.333e-03,  2.636e-03, -2.217e-03,  6.892e-04,
         2.247e-04, -1.253e-04,  3.642e-04, -2.662e-04, -1.300e-04,
         4.913e-05, -3.771e-05,  1.910e-05,  2.632e-05, -2.793e-05,
         3.069e-05,  1.637e-05, -4.171e-05,  2.419e-05,  5.690e-06],
       [ 2.150e-03, -1.660e-03,  1.185e-03, -9.134e-04,  2.519e-04,
         8.031e-05, -7.813e-05,  1.303e-04, -1.024e-04, -4.303e-05,
         2.585e-05, -1.565e-05,  2.653e-06,  9.374e-06, -9.905e-06,
         1.393e-05,  9.953e-06, -1.683e-05,  5.019e-06, -1.065e-06],
       [ 7.193e-04, -4.139e-04,  4.263e-04, -2.087e-04,  1.193e-04,
         6.330e-05, -9.131e-07,  4.958e-05, -2.952e-05, -2.036e-05,
         4.815e-06, -5.640e-06, -2.665e-06,  1.915e-06, -1.019e-06,
         5.658e-06,  4.322e-06, -4.622e-06,  4.123e-07, -4.071e-07]])
    
    Cheb_B=np.array([[-9.034e+01, -2.043e+00,  5.543e-01,  2.635e-01, -1.908e-01,
        -8.945e-02,  1.031e-01,  8.685e-03, -1.973e-02, -2.748e-02,
         1.732e-02,  2.110e-02, -1.711e-02, -3.315e-03,  1.131e-02,
        -1.716e-03, -5.029e-03,  2.513e-04,  9.395e-04, -1.109e-03],
       [-9.323e+01, -3.166e+00,  6.366e-01,  3.918e-01, -1.438e-01,
        -9.340e-02,  1.797e-02, -1.055e-02,  2.962e-02, -1.900e-02,
        -7.333e-03,  1.382e-02, -4.432e-03,  4.100e-04,  1.295e-03,
        -4.573e-03,  2.852e-03,  8.983e-03, -2.693e-03, -1.595e-02],
       [-9.386e+01, -2.947e+00,  5.784e-01,  3.878e-01, -1.458e-01,
        -7.076e-02,  4.514e-02, -5.122e-02,  2.623e-02,  1.820e-02,
        -1.198e-02,  5.901e-03, -1.095e-02, -7.768e-03,  1.531e-02,
         4.066e-03, -1.058e-02,  5.817e-04,  5.587e-03, -5.306e-03],
       [-9.410e+01, -2.906e+00,  3.536e-01,  5.538e-01, -1.171e-01,
        -1.412e-01,  4.824e-02, -2.360e-02,  2.029e-02,  2.826e-02,
        -1.718e-02, -1.347e-02,  3.619e-03,  3.826e-03,  8.445e-04,
         2.207e-05, -9.542e-04,  4.961e-04,  7.493e-04, -2.483e-03],
       [-9.534e+01, -3.088e+00,  2.757e-01,  5.764e-01, -7.624e-02,
        -1.421e-01,  3.233e-02, -2.099e-02,  1.737e-02,  1.591e-02,
        -1.513e-02,  1.962e-03,  5.233e-03, -8.281e-03, -2.361e-03,
         5.310e-03,  2.310e-03,  2.390e-03, -1.331e-03, -8.785e-03],
       [-9.642e+01, -2.858e+00,  5.829e-02,  6.375e-01, -5.312e-02,
        -1.482e-01,  2.887e-02, -1.076e-02, -1.116e-02,  2.979e-02,
         8.237e-03, -1.832e-02, -2.452e-03,  1.801e-03, -3.948e-03,
         6.072e-03,  6.778e-03, -3.745e-03, -4.993e-03, -6.660e-04],
       [-9.703e+01, -2.594e+00, -7.024e-02,  6.001e-01,  1.512e-02,
        -1.638e-01,  7.434e-03,  1.099e-02, -2.190e-02,  1.758e-02,
         2.665e-02, -1.295e-02, -1.879e-02, -1.824e-03,  1.829e-03,
         7.205e-03,  1.038e-02, -6.363e-04, -9.602e-03, -6.678e-03],
       [-9.864e+01, -2.536e+00, -3.294e-01,  6.505e-01,  5.220e-02,
        -1.058e-01, -2.734e-02,  1.492e-02, -1.366e-03,  1.298e-04,
         1.760e-02, -1.015e-02, -9.069e-03,  1.654e-03, -2.437e-03,
         4.078e-03,  8.884e-03,  5.609e-04, -8.298e-03, -5.016e-03],
       [-1.000e+02, -2.252e+00, -5.156e-01,  6.248e-01,  6.029e-02,
        -8.307e-02, -6.397e-02,  1.694e-02, -7.283e-03, -2.217e-02,
         3.612e-02,  1.071e-02, -1.768e-02, -1.305e-02, -1.182e-04,
         9.896e-03,  8.328e-03,  3.503e-03, -8.986e-03, -1.197e-02],
       [-1.011e+02, -2.083e+00, -4.853e-01,  4.803e-01,  1.351e-01,
        -6.864e-02, -5.558e-02,  3.124e-02,  4.599e-03, -1.786e-02,
         2.274e-02,  5.957e-03, -1.761e-02, -1.137e-02,  4.972e-03,
         1.112e-02,  3.272e-03, -7.705e-04, -4.973e-03, -6.572e-03],
       [-1.014e+02, -2.066e+00, -7.504e-01,  4.745e-01,  2.168e-01,
        -1.937e-03, -1.364e-01, -2.072e-02,  3.356e-02, -9.511e-03,
         5.074e-03,  8.147e-03,  2.561e-03, -1.267e-02, -1.241e-02,
         9.669e-03,  1.462e-02,  2.407e-03, -1.067e-02, -9.105e-03],
       [-1.030e+02, -1.338e+00, -7.495e-01,  4.210e-01,  9.038e-02,
         5.266e-02, -1.285e-01, -2.158e-02,  5.222e-02, -5.759e-03,
        -1.919e-03, -1.161e-02,  6.151e-03,  5.287e-03, -8.353e-03,
         1.882e-04,  5.076e-03,  4.631e-03, -4.606e-03, -5.628e-03],
       [-1.052e+02, -2.053e-01, -9.184e-01,  6.424e-01, -2.013e-01,
         1.741e-01, -1.402e-01, -1.998e-02,  7.445e-02, -2.876e-02,
         1.849e-02, -2.493e-02,  1.185e-03,  1.271e-02, -6.457e-03,
         1.923e-03, -2.604e-03,  3.955e-03, -1.771e-03, -8.120e-04],
       [-1.065e+02,  3.918e-01, -6.771e-01,  4.524e-01, -2.580e-01,
         1.869e-01, -7.475e-02, -2.651e-02,  3.941e-02, -2.945e-02,
         2.945e-02, -1.244e-02, -5.676e-03,  4.094e-03, -3.998e-03,
         5.606e-03, -1.500e-03,  9.896e-04, -2.084e-03,  8.332e-04],
       [-1.064e+02,  1.969e-01, -3.810e-01,  2.417e-01, -1.386e-01,
         1.041e-01, -3.950e-02, -1.237e-02,  1.871e-02, -1.770e-02,
         1.796e-02, -6.459e-03, -2.906e-03,  2.009e-03, -2.747e-03,
         3.347e-03, -7.951e-04,  3.932e-04, -1.041e-03,  6.420e-04],
       [-1.066e+02,  1.143e-01, -1.429e-01,  9.719e-02, -6.323e-02,
         4.084e-02, -1.395e-02, -2.951e-03,  7.982e-03, -8.596e-03,
         6.086e-03, -2.523e-03, -3.647e-04,  1.411e-03, -1.198e-03,
         8.609e-04, -6.351e-04,  2.287e-04, -1.511e-04,  5.175e-04],
       [-1.066e+02,  4.236e-02, -3.672e-02,  2.941e-02, -1.935e-02,
         1.066e-02, -4.261e-03, -5.068e-04,  2.929e-03, -2.263e-03,
         1.671e-03, -8.325e-04, -3.006e-04,  4.262e-04, -2.424e-04,
         2.416e-04, -1.981e-04,  4.404e-05, -2.645e-05,  1.734e-04]])
    
    def chebev(min,max,c,x):
        ncoeff=c.shape[0]
        d=np.zeros(x.shape[0])
        dd=np.zeros(x.shape[0])
        y=(2*x-min-max)/(max-min)
        y2=2*y
        for j in range(ncoeff-1,0,-1):
            sv=d
            d=y2*d-dd+c[j]
            dd=sv
        return y*d-dd+0.5*c[0]

    Xsect=[]
    lnO2col=np.log(O2col); #log is natural logaritm
    bound=np.ones_like (lnO2col)
    Temp=(Temp * np.ones_like(lnO2col))
    lnO2col=np.max((lnO2col,38*bound),axis=0)
    lnO2col=np.min((lnO2col,56*bound),axis=0)
    for region in range(17):    
        A=chebev(38,56,Cheb_A[region,:],lnO2col)
        B=chebev(38,56,Cheb_B[region,:],lnO2col)
        #print (A,B)
        Xsect.append(np.exp(A*(Temp-220)+B))
    return np.array(Xsect)


# %%
#ALL INITIAL NUMBERDENSITY (AT NOON) AND BACKGROUND ATMOSPHERE

#number density O family
n_o    = np.array([1.8e9, 2.6e9, 2.9e9, 2.6e9, 2.1e9, 1.9e9, 5.2e9, 5.0e10, 2.0e11, 3.4e11, 4.0e11, 4.9e11, 4.6e11])
n_o_1D = np.array([1.9e2, 1.5e2, 9.8e1, 5.2e1, 2.4e1, 1.3e1, 1.5e1, 3.3e1,  7.0e1,  8.9e1,  1.4e2, 3.2e2, 7.8e2])
n_o2        = np.array([4.8e15, 2.4e15, 1.2e15, 6.5e14, 3.4e14, 1.7e14, 8.6e13, 4.2e13, 2.0e13, 1.0e13, 5.0e12, 2.2e12, 7.7e11])
n_o2_1Delta = np.zeros_like(n_o2)
n_o2_1Sigma = np.zeros_like(n_o2)
n_o3 = np.array([5.7e10, 1.9e10, 7.6e9, 3.5e9, 1.5e9, 4e8, 6.9e7, 9.9e7, 1.6e8, 1.3e8, 8.3e7, 3.8e7, 6.5e6])

#number density N family
n_n2 = 4 * n_o2

#number density CO2
n_co2 = n_o2 / 565.0

#number density H family
n_oh = np.array([6.8e6,4.6e6,3.2e6,2.8e6,3.2e6,3.4e6,3.1e6,4.9e5,5.9e4,1.3e4,3.4e3, 6.0e2, 6.5e1])
n_ho2 = np.array([4.0e6,2.7e6,1.9e6,1.5e6,1.5e6,1.5e6,1.2e6,1.3e5,4.0e3,3.3e2,4.0e1, 2.9e0, 2.1e-1])
n_h = np.array([9.9e4,3.6e5,9.7e5,2.5e6,6.8e6,2.1e7,7.9e7,2.4e8,2.0e8,1.2e8,6.3e7, 3.2e7, 1.6e7])

#background atmosphere
T = np.array([253.7,247.2, 235.4, 220.6, 207.3, 198.0, 195.3, 195.4, 192.5, 185.2, 176.3, 177.6, 192.5])
heights = np.arange(50, 115, 5)

# %%
# EXTRAPOLATION OF O, O2, O3 FOR 120KM AND 130KM FOR PHOTOLYSIS CALCULATIONS
heights2 = np.array(heights.tolist() + np.arange(heights[-1]+10, 130+10, 10).tolist())
T_extrapolated = interp1d(heights, T, fill_value='extrapolate')(heights2)

n_o_extrapolated = np.exp(interp1d(heights, np.log(n_o), fill_value='extrapolate')(heights2))
n_o2_extrapolated = np.exp(interp1d(heights, np.log(n_o2), fill_value='extrapolate')(heights2))
n_o3_extrapolated = np.exp(interp1d(heights, np.log(n_o3), fill_value='extrapolate')(heights2))
n_n2_extrapolated = 4 * n_o2_extrapolated
n_co2_extrapolated = n_o2_extrapolated / 565.0

# %%
# OBTAINING THE PRESSURE AT EACH ALTITUDE BASED ON MY TEMPERATURES FOR gA
Kb = 1.380649e-23 # J/K Boltzmans constant
N = n_n2 + n_o2 + n_o
pres = N*Kb*T*1e6

# %%
# FITTING O2 TO AN EXPONENTIAL CURVE TO EASE THE O2 column density CALCULATION 
# FOR THE SCHUMANN RUNGE BANDS PROGRAM 

def no2_fit(x, a, b):
    return a * np.exp(-x/b) 

# %%
# FIRST FUNTION IS THE O2 COLUMN FOR THE SCHUMANN RUNGE BANDS PROGRAM
#
# SECOND FUNTION IS THE PHOTOLYSIS PROGRAM FOR THE PHOTOLYSIS RATES FOR EACH REGION 
# AS WELL AS FOR THE O2 AND O3 PHOTOLYSIS RATES

def O2_column(alt, alt_top): 
    #alt in km
    p0 = [10**17,3]
    popt, pcov = curve_fit(no2_fit, heights, n_o2, p0 = p0)
    n0, H = popt
    return H * 1e5 * n0 * (np.exp(-alt/H)-np.exp(-alt_top/H))

def photolysis(alt, T, sol_zen, o, o2, o3, n2, co2):
    #alt in m
    #o2, o3 in cm-3
    # if sol_zen == 0 or sol_zen == 180:
        # sol_zen += 0.001
    def path_z(alt_top, z_t, sol_zen, nsteps):
        if sol_zen == 0:
            z_step = np.linspace(alt_top, z_t, nsteps)
            step = (alt_top - z_t)/nsteps
        elif sol_zen == 180:
            z_step = np.zeros(nsteps)
            step = 0
        else:
            Re = 6375e3 #m
            sol_zen /= 180/np.pi #deg to rad
            B = np.arcsin((Re+z_t) * np.sin(np.pi-sol_zen)/(Re+alt_top))
            S_top = np.sin(sol_zen-B)*(Re+alt_top)/np.sin(np.pi-sol_zen)

            Ret2 = (Re+z_t)**2
            step = S_top/nsteps
            S_top_half = S_top - step/2
            z_step = [np.sqrt(Ret2 + (S_top_half - i*step)**2 - 
                                2*(Re + z_t)*(S_top_half - 
                                i*step)*np.cos(np.pi-sol_zen))-Re for i in range(nsteps)]
            z_step = np.array(z_step)
            #check the Earth's shadow
            if (z_step<0).any():
                    z_step = np.zeros(nsteps)
        return (z_step, step)
    
    Jsrb, Jsrc, Jlya, Jherz, Jhart = [], [], [], [], []
    J1, J2, J3, J4 = [], [], [], [] 
    for iz, z_t in enumerate(alt):
        so2[65:82] = effxsect(T[iz], [O2_column(z_t*1e-3, alt[-1]*1e-3)])
        z_paths, path_step = path_z(alt[-1],z_t,sol_zen, 500)
        tau = (so2*(np.exp(np.interp(z_paths,alt,np.log(o2)))).sum()+
               so*(np.interp(z_paths,alt,o)).sum()+
               sn2*(np.interp(z_paths,alt,n2)).sum()+
               sco2*(np.interp(z_paths,alt,co2)).sum()+
               so3*(np.interp(z_paths,alt,o3)).sum()) * path_step * 1e2 #m->cm
        
        jo2, jo3 = (irrad * (so2, so3) * np.exp(-tau))
        
        Jhart.append(jo3[np.transpose(np.logical_and(wave>210,wave<310))].sum())
        Jsrc.append(jo2[np.transpose(np.logical_and(wave>122,wave<175))].sum())
        Jsrb.append(jo2[np.transpose(np.logical_and(wave>175,wave<200))].sum())
        Jherz.append(jo2[np.transpose(np.logical_and(wave>194,wave<240))].sum())
        Jlya.append(jo2[np.transpose(wave==121.567)].sum())
        J1.append(jo2[np.transpose(np.logical_and(wave>177.5,wave<256))].sum()) #SRB+Herzberg?
        J2.append(jo2[np.transpose(np.logical_and(wave<177.5,wave>0))].sum()) #SRC+Lya?
        J3.append(jo3[np.transpose(np.logical_and(wave>200.0,wave<320.0))].sum()*0.1+jo3[np.transpose(np.logical_and(wave>320.0,wave<730.0))].sum())
        J4.append(jo3[np.transpose(np.logical_and(wave>167.5,wave<200.0))].sum()+jo3[np.transpose(np.logical_and(wave>200.0,wave<320.0))].sum()*0.9)

    return (np.array(Jhart), np.array(Jherz), np.array(Jsrc), np.array(Jsrb), np.array(Jlya), 
            np.array(J1), np.array(J2), np.array(J3), np.array(J4))


# %%
# CREATION OF THE INTERPOLATED gA
# AND ALSO THE INTERPOLATED O2 & O3 PHOTOLYSIS RATES
# FOR EACH ALTITUDE FOR THE SIMULATION PROGRAM

def create_G(h,t):
    def function():
        string = 'G_factor_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def create_J1(h,t):
    def function():
        string = 'J1_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def create_J2(h,t):
    def function():
        string = 'J2_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def create_J4(h,t):
    def function():
        string = 'J4_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def norm_t(t):
        return np.mod(t,24*60*60) 


# %%
# SAME AS BEFORE BUT THIS IS FOR THE SECOND ITERATION 
# WITH THE PHOTOLYSIS RATES BEING RE-EVALUATED 
# AFTER THE FIRST SIMULATION

def create_new_G(h,t):
    def function():
        string = 'G_factor_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def create_new_J1(h,t):
    def function():
        string = 'new_J1_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def create_new_J2(h,t):
    def function():
        string = 'new_J2_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def create_new_J4(h,t):
    def function():
        string = 'new_J4_day' + "[" + str(h) + ",:]"
        return interp1d(t_day,eval(string + ".tolist()"))(t)
    return function

def norm_t(t):
        return np.mod(t,24*60*60) 


# %%
# CREATING A FUNCTION THAT CREATES A STRING 
# THAT IS GOING TO BE COMPLETED WITH THE PROPER 
# ALTITUDE IN THE SIMULATION PROGRAM

def rates():
    string = ['create_J1', 'create_J2', 'create_J4', 'create_G']
    return string

def new_rates():
    string = ['create_new_J1', 'create_new_J2', 'create_new_J4', 'create_new_G']
    return string


# %%
# SIMULATION PROGRAM THAT CONSIST IN AN ODEINT INTEGRATION
# FOR EACH OF THE ALTITUDES 
# STARTING AT MIDDAY (12:00) OF THE 1ST DAY AND 
# FINISHING AT MIDNITH (12:00) OF THE 4RD DAY 
# FOR A TOTAL OF 4 DAYS OF INTEGRATION
# WITH A TIME STEP OF 5 MINUTES
#
#
# RATES TO BE UPDATED TO ANQI'S
def simulation(z, rates, tout, x0):
    print('z={}km'.format(heights[z]))
    photolysis_rates = rates()
    
    def system(values, t):
        #First the concentrations
        
        O3        = values[0]     
        O2        = values[1]     
        O2_1Delta = values[2]
        O2_1Sigma = values[3]    
        O         = values[4] 
        O_1D      = values[5]       
        OH        = values[6]      
        HO2       = values[7]      
        H         = values[8]         
        N2        = 4.0*O2
        M         = O2 + N2
        
        #O3 reactions & rates
        
        J4 = eval(photolysis_rates[2] + "(z,norm_t(t))()")
        k2 = 1.0e-10*np.exp(-516/T[z])
        k3 = 1.5e-12*np.exp(-1000/T[z])
        
        r1 = J4 * O3 
        r2 = k2 * O3 * H 
        r3 = k3 * O3 * OH
        
        #O2 reactions

        f = 2 #solar variability on wavelength below 256nm
        k4 = 1.07e-34*np.exp(510/T[z])
        k5 = 2.08e-32*np.exp(290/T[z])
        J1 = eval(photolysis_rates[0] + "(z,norm_t(t))()") * f
        J2 = eval(photolysis_rates[1] + "(z,norm_t(t))()") * f
        G  = eval(photolysis_rates[3] + "(z,norm_t(t))()")
        
        r4 = k4 * O2 * O * M
        r5 = k5 * O2 * H * M
        r6 = J1 * O2
        r7 = J2 * O2
        r20 = G * O2
        
        #O2_1Delta reactions
        
        k8  = 3.6e-18*np.exp(-220/T[z])
        k9  = 1.0e-20
        k10 = 1.3e-16
        A1 = 2.23e-4
        
        r8  =  k8 * O2_1Delta * O2
        r9  =  k9 * O2_1Delta * N2
        r10 = k10 * O2_1Delta * O
        r21 = A1  * O2_1Delta
        
        #O2_1Sigma reactions
        
        k11 = 2.1e-15
        k12 = 2.2e-11
        k13 = 8.0e-14
        k14 = 3.9e-17
        A2  = 0.0758
        
        r11 = k11 * O2_1Sigma * N2
        r12 = k12 * O2_1Sigma * O3
        r13 = k13 * O2_1Sigma * O
        r14 = k14 * O2_1Sigma * O2
        r22 = A2  * O2_1Sigma
        
        #O reactions
        
        k15 = 4.2e-11
        k16 = 3.5e-11
        
        r15 = k15 * O * OH 
        r16 = k16 * O * HO2 
        
        #O_1D reactions
        
        k17 = 1.8e-11*np.exp(110/T[z])
        k18 = 3.2e-11*np.exp(70/T[z]) * 0.77
        k19 = 3.2e-11*np.exp(70/T[z]) * 0.23
        
        r17 = k17 * O_1D * N2
        r18 = k18 * O_1D * O2
        r19 = k19 * O_1D * O2
        
        
        #The system of equations
        dO3dt        = - r1 - r2 - r3 + r4
        dO2dt        = r2 + r3 - r4 - r5 - r6 - r7 + r8 + r9 + r10 + r15 + r16 - r18 - r20 + r21 + r22
        dO2_1Deltadt = r1 - r8 - r9 - r10 + r11 + r12 + r13 + r14 - r21
        dO2_1Sigmadt = - r11 - r12 - r13 - r14 + r18 + r20 - r22
        dOdt         = - r4 + 2*r6 + r7 - r15 - r16 + r17 + r18 + r19
        dO_1Ddt      = r1 + r7 - r17 - r18 - r19
        dOHdt        = r2 - r3 - r15 + r16 
        dHO2dt       = r3 - r16 + r5
        dHdt         = - r2 - r5 + r15
    
        return [dO3dt,
                dO2dt,
                dO2_1Deltadt,
                dO2_1Sigmadt,
                dOdt,
                dO_1Ddt,
                dOHdt,
                dHO2dt,
                dHdt]

    yout = odeint(system, x0, tout, mxstep=5000000)
    return yout

# %%
# INTERPOLATION OF O2 AND O3 PHOTOLYSIS RATES WITH 
# CONSTANT VALUES FOR O,O2,O3,N2 & CO2
# FOR A SET OF SOLAR ZENITH ANGLES EVENLY SPACED
# IN TIME THROUGHT THE DAY

dt = 60*15 #s (15min interval)
t_day = np.arange(0, 24*60*60+dt, dt) #s 
month = int(sys.argv[1])
lat = int(sys.argv[2])
time = Time(pd.date_range(
    "2001-{}-15".format(str(month).zfill(2)), 
    freq="{}min".format(int(dt/60)), 
    periods=len(t_day)))
loc = coord.EarthLocation(lon=0 * u.deg,
                        lat=lat * u.deg)
altaz = coord.AltAz(location=loc, obstime=time)
sun = coord.get_sun(time)
Xi = sun.transform_to(altaz).zen.deg

# np.seterr(divide='ignore', invalid='ignore')

# #EMPTY J's & G
# J1_day = np.zeros((len(heights),len(Xi)))
# J2_day = np.zeros((len(heights),len(Xi)))
# J3_day = np.zeros((len(heights),len(Xi)))
# J4_day = np.zeros((len(heights),len(Xi)))
# G_factor_day = np.zeros((len(heights),len(Xi)))    

# j = 0
# for sol_zen in tqdm(Xi):
#     _,_,_,_,_, J1, J2, J3, J4 = photolysis(
#         heights2*1e3, 
#         T_extrapolated, 
#         sol_zen,
#         n_o_extrapolated,
#         n_o2_extrapolated,
#         n_o3_extrapolated,
#         n_n2_extrapolated,
#         n_co2_extrapolated)
#     for z in range(len(heights)):        
#         G_factor_day[z,j] = gA(pres[z],np.abs(sol_zen))
#         J1_day[z,j] = eval("J1[z]")
#         J2_day[z,j] = eval("J2[z]")
#         J3_day[z,j] = eval("J3[z]")
#         J4_day[z,j] = eval("J4[z]")
        
#     j += 1

# da_Js = xr.DataArray(
#     [J1_day, J2_day, J3_day, J4_day, G_factor_day], 
#     dims=('species', 'z', 't'), 
#     coords=(
#         ('species', 'J1_day, J2_day, J3_day, J4_day, G_factor_day'.split(', ')),
#         ('z', heights, dict(units='km')),
#         ('t', t_day, dict(units='s'))
#         ),
#     )
# da_Js.to_dataset('species').assign(
#     sza=xr.DataArray(Xi, dims=('t',)), 
#     month=month,
#     lat=lat
#     ).to_netcdf(path+'results_nc/J2_factor2/Js_{}_{}.nc'.format(month, lat), mode='w')

# %%
#running all simulations
# t_ini = 12*60*60
# t_end = (24*4)*60*60
# dt = 10*60 #10 minute inteval
# tout = np.arange(t_ini, t_end+dt, dt) #s (5min inteval)
# # sim1 = []
# # for zi, alt in enumerate(heights):
# #     yout = simulation(zi, rates, tout)
# #     sim1.append(yout)
# def fun(zi):
#     x0 = [n_o3[zi],
#           n_o2[zi],
#           n_o2_1Delta[zi],
#           n_o2_1Sigma[zi],
#           n_o[zi],
#           n_o_1D[zi],
#           n_oh[zi],
#           n_ho2[zi],
#           n_h[zi]]
#     return simulation(zi, rates, tout, x0)
# with Pool(processes=8) as p:
#     sim1 = p.map(fun, range(len(heights)))

# species = 'O3,O2,O2_1Delta,O2_1Sigma,O,O_1D,OH,HO2,H'.split(',')
# sim1 = xr.DataArray(
#     sim1, 
#     dims=('z', 't', 'species'), 
#     coords=(
#         ('z', heights, dict(units='km')),
#         ('t', tout, dict(units='s')), 
#         ('species', species)
#         ),
#     )
# sim1.to_dataset('species').assign(
#     month=month, lat=lat
#     ).to_netcdf(path+'results_nc/J2_factor2/sim1_{}_{}.nc'.format(month,lat))
#%% load sim1 results
with xr.open_dataset(
    path+'results_nc/J2_factor2/sim1_{}_{}.nc'format(month, lat)
    ) as sim1:
    sim1 = sim1.drop_vars(['month', 'lat']).to_array('species')

# %%
# GETTING THE NEW O & O3 AS FUNCTION OF TIME IN A DAY
# DURING THE LAST INTEGRATION DAY
new_o3 = sim1.sel(
    species='O3', 
    ).interp(
        z=heights2, t=t_day+3*24*3600,
        kwargs=dict(fill_value='extrapolate')
        ).values
new_o = sim1.sel(
    species='O', 
    ).interp(
        z=heights2, t=t_day+3*24*3600,
        kwargs=dict(fill_value='extrapolate')
        ).values

# %%
# CALCULATE THE NEW PHOTOLYSIS RATES BASED ON THE NEW O & O3 CONCENTRATIONS DURING THE DAY

np.seterr(divide='ignore', invalid='ignore',over='ignore')

#EMPTY J's & G
new_J1_day = np.zeros((len(heights),len(Xi)))
new_J2_day = np.zeros((len(heights),len(Xi)))
new_J3_day = np.zeros((len(heights),len(Xi)))
new_J4_day = np.zeros((len(heights),len(Xi)))    

j = 0
for sol_zen in tqdm(Xi):
    _, _, _, _, _, J1, J2, J3, J4 = photolysis(
        heights2*1e3, T_extrapolated, sol_zen,
        new_o[:,j],
        n_o2_extrapolated,
        new_o3[:,j],
        n_n2_extrapolated,
        n_co2_extrapolated)
    for z in range(len(heights)):        
        new_J1_day[z,j] = eval("J1[z]")
        new_J2_day[z,j] = eval("J2[z]")
        new_J3_day[z,j] = eval("J3[z]")
        new_J4_day[z,j] = eval("J4[z]")
    j += 1

da_Js = xr.DataArray(
    [new_J1_day, new_J2_day, new_J3_day, new_J4_day], 
    dims=('species', 'z', 't'), 
    coords=(
        ('species', 'new_J1_day, new_J2_day, new_J3_day, new_J4_day'.split(', ')),
        ('z', heights, dict(units='km')),
        ('t', t_day, dict(units='s'))
        ),
    )
da_Js.to_dataset('species').to_netcdf(
    path+'results_nc/J2_factor2/Js_factor2_{}_{}.nc'.format(month, lat), mode='w')

# %%
#running all simulations again

# sim2 = []
# for zi, alt in enumerate(heights):
#     yout = simulation(zi, rates, tout)
#     sim2.append(yout)
# dt = 10*60 # 10min
tout = np.arange(0, 3*24*3600+dt, dt) + sim1.t[-1].values #s
def fun(zi):
    x0 = sim1.isel(t=-1, z=zi).values
    return simulation(zi, new_rates, tout, x0)

with Pool(processes=8) as p:
    sim2 = p.map(fun, range(len(heights)))

species = 'O3,O2,O2_1Delta,O2_1Sigma,O,O_1D,OH,HO2,H'.split(',')
sim2 = xr.DataArray(
    sim2, 
    dims=('z', 't', 'species'), 
    coords=(
        ('z', heights, dict(units='km')),
        ('t', tout, dict(units='s')), 
        ('species', species)
        ),
    )
xr.concat(
    [sim1,sim2], dim='t'
    ).to_dataset('species').assign(
        month=month, lat=lat
        ).to_netcdf(path+'results_nc/J2_factor2/sim2_factor2_{}_{}.nc'.format(month,lat))

# %%
