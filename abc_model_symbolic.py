# %%
from chem_classes import *
from sympy_tutorial import *
import numpy as np
import xarray as xr
# %%
t, T = sym.symbols('t T')
a = Species(name='A')
b = Species(name='B')
c = Species(name='C')
species_lst = [a,b,c]
reaction_lst = [
    Reaction(reactants=(a,), products=(b,), name='R1'),
    Reaction(reactants=(b, c), products=(a, c), name='R2'),
    Reaction(reactants=(b,b), products=(b, c), name='R3')]
rate_const_lst = [
    RateConstant(value=0.04*T, name='k1', reaction_name='R1'),
    RateConstant(value=1e4*T, name='k2', reaction_name='R2'),
    RateConstant(value=3e7*T, name='k3', reaction_name='R3')]

#set available rate constant to each reaction
reaction_lst = [r.set_rate_constant(next(k for k in rate_const_lst if k.reaction_name==r.name)) for r in reaction_lst]

# %% Generate a list of (coeff, r_stoich, net_stoich)
coeff_lst = [r.rate_constant.name for r in reaction_lst]
r_stoich_lst = [r.get_r_stoich() for r in reaction_lst]
net_stoich_lst = [r.get_net_stoich() for r in reaction_lst]
reactions = list(zip(coeff_lst, r_stoich_lst, net_stoich_lst))
species_names = [s.name for s in species_lst]

# %% set up all y and ydot equations
sym.init_printing()
ydot, y = mk_exprs_symbs(reactions, species_names) 
k = tuple(sym.S(coeff_lst))

t = sym.symbols('t')  # not used in this case.
f_ode = sym.lambdify((y, t) + k, ydot)
J = sym.Matrix(ydot).jacobian(y)  # EXERCISE: jacobian
f_jcb = sym.lambdify((y, t) + k, J)  # EXERCISE: (y, t) + k

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from scipy.integrate import odeint

tout = np.logspace(-6, 6)
z = np.arange(1,5) 
dz = np.gradient(z) 
func_ks = tuple(sym.lambdify(T, r.rate_constant.value) for r in reaction_lst)
y0 = np.ravel([1*z, 0*z, 0*z], order='F')
D = [0, 0, 0] #diffusion coefficient for each species 
def f_ode_flat_abc(y_flat, t, *func_ks):
    f_out = np.empty(len(y)*len(z))
    for i in range(len(z)):
        slc = slice(i*len(y), (i+1)*len(y))
        f_out[slc] = second_derivatives_spatial(i, y_flat, f_out[slc]) # diffusion contributions
        f_out[slc] *= D
        k_vals = [func_k(T=t*z[i]) for func_k in func_ks]
        f_out[slc] += f_ode(y_flat[slc], t, *tuple(k_vals)) # chemical contributions
    return f_out

def second_derivatives_spatial(i, state_flat, out):
    k = np.clip(i, 1, len(y) - 2)  #not sure what it means
    out = np.empty(len(y))
    for j in range(len(y)):
        left = state_flat[(k-1)*len(y) + j]
        cent = state_flat[(k  )*len(y) + j]
        rght = state_flat[(k+1)*len(y) + j]
        out[j] = (left - 2*cent + rght)/dz[i]**2
    return out

def f_jcb_flat_abc(y_flat, t, *func_ks):
    j_out = np.zeros((len(y_flat), len(y_flat)))  # dense matrix
    for i in range(len(z)):
        slc = slice(i*len(y), (i+1)*len(y))
        k_vals = [func_k(T=t*z[i]) for func_k in func_ks]
        j_out[slc, slc] = f_jcb(y_flat[slc], t, *tuple(k_vals)) #chemical contributions
        k = np.clip(i, 1, len(y) - 2)  #not sure what it means
        for j in range(len(y)): #diffusion contributions
            j_out[i*len(y) + j, (k-1)*len(y) + j] +=    D[j]/dz[i]**2
            j_out[i*len(y) + j, (k  )*len(y) + j] += -2*D[j]/dz[i]**2
            j_out[i*len(y) + j, (k+1)*len(y) + j] +=    D[j]/dz[i]**2
    return j_out

yout, info = odeint(f_ode_flat_abc, y0, tout, func_ks, Dfun=f_jcb_flat_abc,
                     full_output=True)
yout = xr.DataArray(yout.T.reshape(len(y), len(z), len(tout), order='F'),
                    dims=('species', 'z', 't'), coords=(species_names, z, tout))
line_plot_args = dict(x='t', hue='z',yscale='log', xscale='log')
yout.sel(species='A').plot.line(**line_plot_args, color='C0')
yout.sel(species='B').plot.line(**line_plot_args, color='C1')
yout.sel(species='C').plot.line(**line_plot_args, color='C2')
print("The Jacobian was evaluated %d times." % info['nje'][-1])
print("The function was evaluated %d times." % info['nfe'][-1])

# %%
#%%
# def reorder(test, nx, ny):
#     test_result = np.zeros(test.shape)
#     for i in range(nx):
#         for j in range(nx):
#             for k in range(ny):
#                 for l in range(ny):
#                     test_result[k * nx + i,l * nx + j] = test[i * ny + k, j * ny + l]
#     return test_result

# def f_jcb_flat_abc(y_flat, t, *func_ks):
#     k_vals = np.array([func_k(T=t*z) for func_k in func_ks]) #matrix ny*nz
#     y_vals = y_flat.reshape(len(y), len(z))
#     j_out = np.zeros((len(y_flat), len(y_flat)))  # dense matrix
#     for i in range(len(z)):
#         slc = slice(i*len(y), (i+1)*len(y))
#         j_out[slc, slc] = f_jcb(y_vals[:,i], t, *tuple(k_vals[:,i])) #chemical contributions
#         k = np.clip(i, 1, len(y) - 2)  #not sure what it means
#         for j in range(len(y)): #diffusion contributions
#             j_out[i*len(y) + j, (k-1)*len(y) + j] +=    D[j]/dz**2
#             j_out[i*len(y) + j, (k  )*len(y) + j] += -2*D[j]/dz**2
#             j_out[i*len(y) + j, (k+1)*len(y) + j] +=    D[j]/dz**2
#     j_out = reorder(j_out, len(z), len(y))
#     return j_out

# def f_ode_flat_abc(y_flat, t, *func_ks):
#     k_vals = tuple([func_k(T=t*z) for func_k in func_ks])
#     f_out = np.ravel(f_ode(y_flat.reshape(len(y), len(z)), t, *k_vals))
#     return f_out

# y0 = np.ravel([1*z, 0*z, 0*z], order='C')
# yout, info = odeint(f_ode_flat_abc, y0, tout, func_ks, Dfun=f_jcb_flat_abc,
#                      full_output=True)

# yout = yout.reshape(len(tout), len(y), len(z))
# yout = xr.DataArray(yout, dims=('t', 'species','z'), coords=(tout,species_names, z))
# line_plot_args = dict(x='t', hue='z',yscale='log', xscale='log')
# yout.sel(species='A').plot.line(**line_plot_args, color='C0')
# yout.sel(species='B').plot.line(**line_plot_args, color='C1')
# yout.sel(species='C').plot.line(**line_plot_args, color='C2')

# print("The Jacobian was evaluated %d times." % info['nje'][-1])
# print("The function was evaluated %d times." % info['nfe'][-1])