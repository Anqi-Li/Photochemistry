# %%
from chem_classes import *
from chem_sympy import *
import sympy as sym
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.integrate import odeint
# %% [Set up the system (species involved and list of reactions)]
reaction_lst = get_reactions_frederick1979()
species_lst = list(set(reduce(add, [r.reactants + r.products for r in reaction_lst])))
# generate a list of (coeff, r_stoich, net_stoich)
coeff_lst = [r.name for r in reaction_lst]
r_stoich_lst = [r.get_r_stoich() for r in reaction_lst]
net_stoich_lst = [r.get_net_stoich() for r in reaction_lst]
reactions = list(zip(coeff_lst, r_stoich_lst, net_stoich_lst))
species_names = [s.name for s in species_lst]

# set up all y and ydot equations
ydot, y, k = mk_exprs_symbs(reactions, species_names) 
y_loss = loss_terms(reactions, species_names)

# %%
# set fixed species
species_fix = [o2, m]
y_fixed = tuple([y[species_lst.index(s)] for s in species_fix])

# set up ode (ydot) equations of each family
# HOx, Ox = sym.symbols('HOx, Ox', real=True, nonnegative=True, positive=True)
# families = {HOx: [oh, ho2, h], Ox: [o, o3]}
# y_solve_family = family_equilibrium(families, reactions, species_lst)
# lifetime_families = {familyname: reduce(add, [y[species_lst.index(s)] for s in families[familyname]]
#                                 )/reduce(add, [y_loss[species_lst.index(s)] for s in families[familyname]]) 
#                                 for familyname in families.keys()}

# ydot_non_eq = [reduce(add, [ydot[species_lst.index(s)] 
#     for s in family_species]).subs(y_solve_family)
#     for family_species in families.values()]
# y_non_eq = list(families.keys()) 

species_non_eq = [o, o3, h, ho2, oh]
y_non_eq = [y[species_lst.index(s)] for s in species_non_eq]
ydot_non_eq = [ydot[species_lst.index(s)] for s in species_non_eq]

#%%
t = sym.symbols('t', real=True, nonnegative=True, positive=True)
f_ode = sym.lambdify((y_non_eq, t) + k + y_fixed, ydot_non_eq)
jcb = sym.Matrix(ydot_non_eq).jacobian(y_non_eq)  
f_jcb = sym.lambdify((y_non_eq, t) + k + y_fixed, jcb)  

#%%
# set values
js_ds = xr.open_dataset('./Js_table_w0.nc')
ds = xr.open_dataset('./s&b_tables_anqi.nc')
z = np.arange(50, 105, 5) #km
T_val = ds.T.interp(z=z).drop('index') #K
m = m.set_density(ds.m.interp(z=z).drop('index')) #cm-3
o2 = o2.set_density(m.density.rename('o2') * 0.2) #cm-3

y_fixed_vals = tuple([s.density for s in species_fix])
func_ks = tuple(sym.lambdify(T, r.rate_constant.value) for r in reaction_lst)
D = np.zeros(len(y_non_eq)) #diffusion coefficient for each non_eq species

#%%
def get_js(t,z):
    js = js_ds.interp(t=t%(24*3600), kwargs=dict(fill_value=0)
                    ).interp(z=z, kwargs=dict(fill_value='extrapolate'))
    return js.J3

def f_ode_flat(y_flat, t, *func_ks):
    print(t/3600)
    f_out = np.empty(len(y_non_eq)*len(z))
    for i in range(len(z)):
        slc = slice(i*len(y_non_eq), (i+1)*len(y_non_eq))
        #diffusion contributions
        f_out[slc] = second_derivatives_spatial(i, y_flat, f_out[slc]) 
        f_out[slc] *= D
        # chemical contributions
        k_vals_z = [func_k(T_val[i]) for func_k in func_ks]
        k_vals_z[1] = get_js(t, z[i]) 
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        f_out[slc] += f_ode(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
    return f_out

def second_derivatives_spatial(i, state_flat, out):
    k = np.clip(i, 1, len(y_non_eq) - 2)  #not sure what it means
    out = np.empty(len(y_non_eq))
    for j in range(len(y_non_eq)):
        left = state_flat[(k-1)*len(y_non_eq) + j]
        cent = state_flat[(k  )*len(y_non_eq) + j]
        rght = state_flat[(k+1)*len(y_non_eq) + j]
        out[j] = (left - 2*cent + rght)/np.gradient(z)[i]**2
    return out

def f_jcb_flat(y_flat, t, *func_ks):
    j_out = np.zeros((len(y_flat), len(y_flat)))  # dense matrix
    for i in range(len(z)):
        #chemical contributions
        slc = slice(i*len(y_non_eq), (i+1)*len(y_non_eq))
        k_vals_z = [func_k(T_val[i]) for func_k in func_ks]
        k_vals_z[1] = get_js(t, z[i])
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        j_out[slc, slc] = f_jcb(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
        #diffusion contributions
        k = np.clip(i, 1, len(y_non_eq) - 2)  #not sure what it means
        for j in range(len(y_non_eq)): 
            j_out[i*len(y_non_eq) + j, (k-1)*len(y_non_eq) + j] +=    D[j]/np.gradient(z)[i]**2
            j_out[i*len(y_non_eq) + j, (k  )*len(y_non_eq) + j] += -2*D[j]/np.gradient(z)[i]**2
            j_out[i*len(y_non_eq) + j, (k+1)*len(y_non_eq) + j] +=    D[j]/np.gradient(z)[i]**2
    return j_out


#%% integration
# %%time
y0 = np.ravel([ds[s.name.lower()].interp(z=z) for s in y_non_eq], order='F')
# y0 = np.ravel([reduce(add,[ds[s.name.lower()].interp(z=z) 
#                 for s in families[familyname]]) 
#                 for familyname in families.keys()], order='F')
tout = np.arange(12*3600, (12+7*24)*3600, 100) #s

# # yout_save = yout.copy()

# yout_save = xr.open_dataset('./ode_result/ode_frederick_single_12_180.nc')
# y0 = np.ravel([yout_save.sel(t=29.5, method='nearest')[s.name] for s in y_non_eq], order='F')
# tout = np.arange(yout_save.t[-1]*3600, 48*3600, 100)  #s
# tout_temp = tout % (24*3600)
# tout = np.arange(29.5*3600, (30+24)*3600, 100) #s
yout_org, info = odeint(f_ode_flat, y0, tout, args=func_ks, Dfun=f_jcb_flat,
                        full_output=True, #atol=1e-9, rtol=1e-9,
                        # tcrit=np.array([6, 18, 6+24, 18+24])*3600
                        )
yout = yout_org.reshape(len(tout), len(z), len(y_non_eq))
yout = xr.DataArray(yout, dims=('t', 'z', 'species'), 
                    coords=(tout/3600, z, [s.name for s in y_non_eq]))
yout = yout.to_dataset(dim='species')

# %%familiy species divisions
# k_vals = [func_k(T_val) for func_k in func_ks]
# k_vals[1] = get_js(yout.t*3600, yout.z).assign_coords(t=yout.t)
# for s in reduce(add, families.values()):
#     f_density = sym.lambdify(tuple(families.keys())+k+y_fixed, y_solve_family[y[species_lst.index(s)]])
#     s.density = f_density(*[yout[s.name] for s in families.keys()], *k_vals, *y_fixed_vals)
#     yout = yout.update({s.name: s.density})

yout.to_netcdf('./ode_result/ode_frederick_single_{}_{}.nc'.format(*yout.t.isel(t=[0,-1]).values.round().astype(int)))

#%% plot results
# yout = xr.open_dataset('./ode_result/ode_frederick.nc')

# fig, ax = plt.subplots(1,2)
# line_plot_args = dict(x='t', hue='z',yscale='log', xscale='linear')
# yout.O3.plot.line(**line_plot_args, ax=ax[0])

# yout_ratio = yout/yout.isel(t=0)
# line_plot_args = dict(x='t', hue='z',yscale='linear', xscale='linear')
# yout_ratio.O3.plot.line(**line_plot_args, ax=ax[1])
# plt.show()
# print("The function was evaluated %d times." % info['nfe'][-1])
# print("The Jacobian was evaluated %d times." % info['nje'][-1])
#%% check Js
# h=18.5
# plt.subplot(211)
# js_ds.J3.assign_coords(t=js_ds.t/3600).interp(t=yout.t).plot.line(x='t', yscale='linear', add_legend=False)
# plt.axvline(x=h)
# plt.subplot(212)
# yout.Ox.plot.line(x='t', yscale='log', add_legend=False, ylim=(1e8, 2e12))
# plt.axvline(x=h)
# %% check lifetime
# yout_save = xr.open_dataset('./ode_result/ode_frederick_12_31.nc')
# k_vals = [func_k(T_val) for func_k in func_ks]

# familyname = Ox
# f_lifetime = sym.lambdify((y,)+k, lifetime_families[familyname])
# # y_vals = [o2.density] + [ds.interp(z=z)[s.name.lower()] for s in y[1:]]
# t_test = 12 #h
# y_vals = [yout_save.interp(z=z,t=t_test)[s.name] for s in y[:2]]+[o2.density]+[yout_save.interp(z=z,t=t_test)[s.name] for s in y[3:-1]]+ [m.density]
# k_vals[1] = get_js(t_test*3600, z)
# f_lifetime(y_vals, *k_vals).plot(y='z', xscale='log')

# # t_test = 20 #h
# # y_vals = [o2.density] + [yout_save.interp(z=z,t=t_test)[s.name] for s in y[1:-1]] + [m.density]
# # k_vals[1] = get_js(t_test*3600, z)
# # f_lifetime(y_vals, *k_vals).plot(y='z', xscale='log')
# ax = plt.gca()
# ax.set_xticks([60, 3600, 3600*24])
# ax.set_xticklabels('1min 1hour 1day'.split())

# %%
