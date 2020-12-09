# %%
from chem_classes import *
from chem_sympy import *
import numpy as np
import xarray as xr

# %%
t, T = sym.symbols('t T')
n2 = Species(name='N2')
o2 = Species(name='O2')
o3 = Species(name='O3')
o = Species(name='O')
o1d = Species(name='O_1D')
o2sig = Species(name='O2_Sigma')
o2del = Species(name='O2_Delta')
m = Species(name='M')

species_lst = [n2, o2, o3, o, o1d, o2sig, o2del, m]

reaction_lst = [
    Reaction(reactants=(o3,), products=(o1d, o2del), name='J_o3'),
    Reaction(reactants=(o1d, o2), products=(o, o2sig), name='Q_1do2'),
    Reaction(reactants=(o1d, n2), products=(o, n2), name='Q_1dn2'),
    Reaction(reactants=(o2sig, m), products=(o2del, m), name='Q_Sigma_m'),
    Reaction(reactants=(o2del, o2), products=(o2, o2), name='Q_Delta_o2'),
    Reaction(reactants=(o2sig,), products=(o2,), name='A_Sigma'),
    Reaction(reactants=(o2del,), products=(o2,), name='A_Delta'),
    Reaction(reactants=(o2,), products=(o2sig,), name='g_A')]
rate_const_lst = [
    RateConstant(name='J_o3', reaction_name='J_o3', unit='s-1'), 
    RateConstant(value=2.9e-11*sym.exp(-67/T), name='k_1do2', reaction_name='Q_1do2', unit='cm3s-1'), 
    RateConstant(value=1.0e-11*sym.exp(-107/T), name='k_1dn2', reaction_name='Q_1dn2', unit='cm3s-1'), 
    RateConstant(value=1.0e-15, name='k_Sigma_m', reaction_name='Q_Sigma_m', unit='cm3s-1'), 
    RateConstant(value=2.22e-18*sym.exp(T/300)**0.78, name='k_Delta_o2', reaction_name='Q_Delta_o2', unit='cm3s-1'), 
    RateConstant(value=0.085, name='A_Sigma', reaction_name='A_Sigma', unit='s-1'), 
    RateConstant(value=2.58e-4, name='A_Delta', reaction_name='A_Delta', unit='s-1'),
    RateConstant(name='g_A', reaction_name='g_A', unit='s-1')]

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
k = sym.S(tuple(coeff_lst))

#%% O2_Delta model
# set fixed species and their concentrations
species_fix = [n2, o2, m, o3]
y_fixed = tuple([y[species_lst.index(s)] for s in species_fix])

# set equilibrium species and equations
species_eq = [o1d, o2sig]
y_solve_eq = sym.solve([sym.Eq(ydot[species_lst.index(s)])  for s in species_eq], 
                 [y[species_lst.index(s)] for s in species_eq])

# set non-equilibrium species and ode equations
species_non_eq = [o2del]
ydot_non_eq = [ydot[species_lst.index(s)].subs(y_solve_eq) for s in species_non_eq]
y_non_eq = [y[species_lst.index(s)] for s in species_non_eq]
jcb = sym.Matrix(ydot_non_eq).jacobian(y_non_eq)  
f_ode = sym.lambdify((y_non_eq, t) + k + y_fixed, ydot_non_eq)
f_jcb = sym.lambdify((y_non_eq, t) + k + y_fixed, jcb)  

# set values
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from scipy.integrate import odeint
from o2delta_model import gA

js_ds = xr.open_dataset('./Js_table_w0.nc')
ds = xr.open_dataset('./s&b_tables_anqi.nc')
z = np.arange(50, 105, 5) #km
dz = np.gradient(z)
P_val = ds.p.interp(z=z).drop('index') *1e2 #Pa
T_val = ds.T.interp(z=z).drop('index') #K
m = m.set_density(ds.m.interp(z=z).drop('index')) #cm-3
o2 = o2.set_density(m.density.rename('o2') * 0.2) #cm-3
n2 = n2.set_density(m.density.rename('n2') * 0.8) #cm-3
o3 = o3.set_density(ds.o3.interp(z=z).drop('index')) #cm-3
D = np.zeros(len(species_non_eq)) #diffusion coefficient for each non_eq species

y_fixed_vals = tuple([s.density for s in species_fix])
func_ks = tuple(sym.lambdify(T, r.rate_constant.value) for r in reaction_lst[1:-1])

def func_sza(t=np.linspace(0, 3600*24), elv_max=90):
    #t in s
    elv = elv_max - 90*((np.cos(2*np.pi/(3600*24) * t)) + 1)
    return 90-elv

def get_js(t):
    Jo3_val = js_ds.J4.interp(t=t,kwargs=dict(fill_value=0))
    sza = func_sza(t)
    if sza>=90:
        gA_val = np.zeros(z.shape)
    else:
        gA_val = gA(P_val, func_sza(t)) 
    return gA_val, Jo3_val

def f_ode_flat(y_flat, t, *func_ks):
    gA_val, Jo3_val = get_js(t)    
    f_out = np.empty(len(y_non_eq)*len(z))
    for i in range(len(z)):
        slc = slice(i*len(y_non_eq), (i+1)*len(y_non_eq))
        #diffusion contributions
        f_out[slc] = second_derivatives_spatial(i, y_flat, f_out[slc]) 
        f_out[slc] *= D
        # chemical contributions
        k_vals_z = [Jo3_val[i]] + [func_k(T_val[i]) for func_k in func_ks] + [gA_val[i]]
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        f_out[slc] += f_ode(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
    return f_out
    # k_vals = tuple([Jo3_val] + [func_k(T_val) for func_k in func_ks] + [gA_val])
    # ydot_vals_flat = np.ravel(f_ode(y_flat.reshape(len(y_non_eq), len(z)), t, *k_vals, *y_fixed_vals))
    # return ydot_vals_flat

def second_derivatives_spatial(i, state_flat, out):
    k = np.clip(i, 1, len(y_non_eq) - 2)  #not sure what it means
    out = np.empty(len(y_non_eq))
    for j in range(len(y_non_eq)):
        left = state_flat[(k-1)*len(y_non_eq) + j]
        cent = state_flat[(k  )*len(y_non_eq) + j]
        rght = state_flat[(k+1)*len(y_non_eq) + j]
        out[j] = (left - 2*cent + rght)/dz[i]**2
    return out

def f_jcb_flat(y_flat, t, *func_ks):
    j_out = np.zeros((len(y_flat), len(y_flat)))  # dense matrix
    for i in range(len(z)):
        #chemical contributions
        slc = slice(i*len(y_non_eq), (i+1)*len(y_non_eq))
        k_vals_z = [Jo3_val[i]] + [func_k(T_val[i]) for func_k in func_ks] + [gA_val[i]]
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        j_out[slc, slc] = f_jcb(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
        #diffusion contributions
        k = np.clip(i, 1, len(y_non_eq) - 2)  #not sure what it means
        for j in range(len(y_non_eq)): 
            j_out[i*len(y_non_eq) + j, (k-1)*len(y_non_eq) + j] +=    D[j]/dz[i]**2
            j_out[i*len(y_non_eq) + j, (k  )*len(y_non_eq) + j] += -2*D[j]/dz[i]**2
            j_out[i*len(y_non_eq) + j, (k+1)*len(y_non_eq) + j] +=    D[j]/dz[i]**2
    return j_out

# test on equlibrium profiles
gA_val, Jo3_val = get_js(12*3600)
k_vals = tuple([Jo3_val] + [func_k(T_val) for func_k in func_ks] + [gA_val])
eq_funcs = [sym.lambdify(k+y_fixed, y_solve_eq[y[species_lst.index(s)]]) for s in species_eq]
eq_vals = [f(*k_vals, *y_fixed_vals).values for f in eq_funcs]
fig, ax = plt.subplots(1,2)
ax[0].semilogx(eq_vals[0], z, label=species_eq[0].name)
ax[0].semilogx(eq_vals[1], z, label=species_eq[1].name)
# ax[0].semilogx(eq_vals[2], z, label=species_eq[2].name)
ax[0].legend()
ax[1].plot(f_ode_flat(eq_vals[1], 12*3600, *func_ks), z, 
            label='d{}/dt'.format(species_eq[1].name))
ax[1].legend()

#%%
%%time
# integration
tout = np.arange(8*3600, 9*3600, 60) #s
y0 = np.ravel([np.zeros(len(z))], order='F') 
yout, info = odeint(f_ode_flat, y0, tout, args=func_ks, #Dfun=f_jcb_flat,
                        full_output=True,
                        tcrit=np.array([6, 18])*3600)
yout = yout.reshape(len(tout),len(y_non_eq), len(z))
yout = xr.DataArray(yout, dims=('t', 'species', 'z'), 
                    coords=(tout/3600, [s.name for s in species_non_eq], z))
# plot results
fig, ax = plt.subplots(1,2)
line_plot_args = dict(x='t', hue='z',yscale='log', xscale='linear')
yout.sel(species=o2del.name).plot.line(**line_plot_args, ax=ax[0])

yout_ratio = yout/yout.isel(t=-1)
line_plot_args = dict(x='t', hue='z',yscale='linear', xscale='linear')
yout_ratio.sel(species=o2del.name).plot.line(**line_plot_args, ax=ax[1])
plt.show()
print("The function was evaluated %d times." % info['nfe'][-1])
print("The Jacobian was evaluated %d times." % info['nje'][-1])

