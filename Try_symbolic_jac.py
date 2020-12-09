# %%
from o2delta_model import gA 
from chem_classes import *
from sympy_tutorial import *
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %%
T = sym.symbols('T')
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

# %%
odesys = ODEsys(ydot, y, params=coeff_lst)
tout = np.logspace(-6, 6)
y0 = [1, 0, 0]
func_params = tuple(sym.lambdify(T, r.rate_constant.value) for r in reaction_lst)
param_vals = [func_param(T=1) for func_param in func_params]
yout, info = odesys.integrate_odeint(tout, y0, param_vals, atol=1e-9, rtol=1e-9)
yout = xr.DataArray(yout, dims=('t', 'species'), 
                    coords=(tout, [s.name for s in odesys.state]))

fig, axes = plt.subplots(1,2, figsize=(14,4))
yout.plot.line(x='t', ax=axes[0])
yout.plot.line(x='t', ax=axes[1], xscale='log', yscale='log')
plt.show()
odesys.print_info(info)

# %%
x = np.linspace(0, 0.01, 50)
molsys = mk_rsys(MOLsys, reactions, species_names, params=coeff_lst, 
                x=x, D=np.zeros(len(species_names)))
y0 = [np.ones_like(x), np.zeros_like(x), np.zeros_like(x)]
func_params = tuple(sym.lambdify(T, r.rate_constant.value) for r in reaction_lst)
param_vals = [func_param(T=np.ones_like(x)) for func_param in func_params]
yout, info = molsys.integrate(tout, y0, param_vals)
yout = xr.DataArray(yout, dims=('t', 'species', 'x'), 
                    coords=(tout, [s.name for s in molsys.state], x))

yout.isel(species=0).plot.line(x='t', hue='x', xscale='log', yscale='log', add_legend=False)
plt.show()
molsys.print_info(info)
# %%


# %%
