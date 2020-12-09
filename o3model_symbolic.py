# %%
from chem_classes import *
from chem_sympy import *
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# %% [Set up the system (species involved and list of reactions)]
t, T = sym.symbols('t T', nonnegative=True, real=True, positive=True)
n2 = Species(name='N2')
o2 = Species(name='O2')
o3 = Species(name='O3')
o = Species(name='O')
o1d = Species(name='O_1D')
h2o = Species(name='H2O')
h = Species(name='H')
oh = Species(name='OH')
h2 = Species(name='H2')
h2o2 = Species(name='H2O2')
ho2 = Species(name='HO2')
m = Species(name='M')
ch4 = Species(name='CH4')
co = Species(name='CO')

reaction_lst = [
    Reaction(reactants=(o2,), products=(o, o), name='R1'),
    Reaction(reactants=(o2,), products=(o, o1d), name='R2'),
    Reaction(reactants=(o3,), products=(o2, o), name='R3'),
    Reaction(reactants=(o3,), products=(o2, o1d), name='R4'),
    # Reaction(reactants=(h2o,), products=(h, oh), name='R5'), #remove 
    # Reaction(reactants=(h2o,), products=(h2, o1d), name='R6'), #remove
    Reaction(reactants=(h2o2,), products=(oh, oh), name='R7'),
    Reaction(reactants=(o1d, o2), products=(o, o2), name='R8'),
    Reaction(reactants=(o1d, n2), products=(o, n2), name='R9'),
    Reaction(reactants=(o1d, h2o), products=(oh, oh), name='R10'),
    Reaction(reactants=(o1d, h2), products=(h, oh), name='R11'),
    Reaction(reactants=(o, o, m), products=(o2, m), name='R12'),
    Reaction(reactants=(o, o, o2), products=(o3, o), name='R13'),
    Reaction(reactants=(o, o2, o2), products=(o3, o2), name='R14'),
    Reaction(reactants=(o, o2, n2), products=(o3, n2), name='R15'),
    Reaction(reactants=(o, o3), products=(o2, o2), name='R16'),
    Reaction(reactants=(o, oh), products=(o2, h), name='R17'),
    Reaction(reactants=(o, ho2), products=(oh, o2), name='R18'),
    Reaction(reactants=(o, h2o2), products=(oh, ho2), name='R19'),
    Reaction(reactants=(o, h2), products=(oh, h), name='R20'),
    Reaction(reactants=(oh, o3), products=(ho2, o2), name='R21'),
    Reaction(reactants=(oh, oh), products=(h2o, o), name='R22'),
    Reaction(reactants=(oh, ho2), products=(h2o, o2), name='R23'),
    Reaction(reactants=(oh, h2o2), products=(h2o, ho2), name='R24'),
    Reaction(reactants=(oh, h2), products=(h2o, h), name='R25'),
    Reaction(reactants=(ho2, o3), products=(oh, o2, o2), name='R26'),
    Reaction(reactants=(ho2, ho2), products=(h2o2, o2), name='R27'),
    Reaction(reactants=(h, o2, m), products=(ho2, m), name='R28'),
    Reaction(reactants=(h, o3), products=(oh, o2), name='R29'),
    Reaction(reactants=(h, ho2), products=(h2, o2), name='R30'),
    Reaction(reactants=(h, ho2), products=(oh, oh), name='R31'),
    Reaction(reactants=(h, ho2), products=(h2o, o), name='R32'),
    Reaction(reactants=(h, h, m), products=(h2, m), name='R33'),
        # r34 
    Reaction(reactants=(ch4, oh), products=(co, oh, h2o, h2o), name='R35'),
    Reaction(reactants=(ch4, o), products=(co, oh, oh, h2o), name='R36'),
    Reaction(reactants=(ch4, o1d), products=(co, oh, oh, h2o), name='R37')
    ]

rate_const_lst = [
    RateConstant(name='J1', unit='s-1', reaction_name='R1'), 
    RateConstant(name='J2', unit='s-1', reaction_name='R2'), 
    RateConstant(name='J3', unit='s-1', reaction_name='R3'), 
    RateConstant(name='J4', unit='s-1', reaction_name='R4'), 
    RateConstant(name='J5', unit='s-1', reaction_name='R5'), 
    RateConstant(name='J6', unit='s-1', reaction_name='R6'), 
    RateConstant(name='J7', unit='s-1', reaction_name='R7'),
    RateConstant(value=3.2e-11*sym.exp(117/T), name='k8', unit='cm3s-1', reaction_name='R8'),
    RateConstant(value=1.8e-11*sym.exp(157/T), name='k9', unit='cm3s-1', reaction_name='R9'),
    RateConstant(value=2.3e-10*sym.exp(-100/T), name='k10', unit='cm3s-1', reaction_name='R10'),
    RateConstant(value=1.1e-10, name='k11', unit='cm3s-1', reaction_name='R11'),
    RateConstant(value=9.59e-34*sym.exp(480/T), name='k12', unit='cm6s-1', reaction_name='R12'),
    RateConstant(value=2.15e-34*sym.exp(345/T), name='k13', unit='cm6s-1', reaction_name='R13'),
    RateConstant(value=2.15e-34*sym.exp(345/T), name='k14', unit='cm6s-1', reaction_name='R14'),
    RateConstant(value=8.82e-35*sym.exp(575/T), name='k15', unit='cm6s-1', reaction_name='R15'),
    RateConstant(value=1.5e-11*sym.exp(-2218/T), name='k16', unit='cm3s-1', reaction_name='R16'),
    RateConstant(value=2.3e-11*sym.exp(-90/T), name='k17', unit='cm3s-1', reaction_name='R17'),
    RateConstant(value=2.8e-11*sym.exp(172/T), name='k18', unit='cm3s-1', reaction_name='R18'),
    RateConstant(value=1.0e-11*sym.exp(-2500/T), name='k19', unit='cm3s-1', reaction_name='R19'),
    RateConstant(value=1.6e-11*sym.exp(-4570/T), name='k20', unit='cm3s-1', reaction_name='R20'),
    RateConstant(value=1.6e-12*sym.exp(-940/T), name='k21', unit='cm3s-1', reaction_name='R21'),
    RateConstant(value=4.5e-12*sym.exp(-275/T), name='k22', unit='cm3s-1', reaction_name='R22'),
    RateConstant(value=8.4e-11, name='k23', unit='cm3s-1', reaction_name='R23'),
    RateConstant(value=2.9e-12*sym.exp(-160/T), name='k24', unit='cm3s-1', reaction_name='R24'),
    RateConstant(value=7.7e-12*sym.exp(-2100/T), name='k25', unit='cm3s-1', reaction_name='R25'),
    RateConstant(value=1.4e-14*sym.exp(-580/T), name='k26', unit='cm3s-1', reaction_name='R26'),
    RateConstant(value=2.4e-14*sym.exp(1250/T), name='k27', unit='cm3s-1', reaction_name='R27'),
    RateConstant(value=1.76e-28*T**(-1.4), name='k28', unit='cm6s-1', reaction_name='R28'),
    RateConstant(value=1.4e-10*sym.exp(-270/T), name='k29', unit='cm3s-1', reaction_name='R29'),
    RateConstant(value=6.0e-12, name='k30', unit='cm3s-1', reaction_name='R30'),
    RateConstant(value=7.0e-11, name='k31', unit='cm3s-1', reaction_name='R31'),
    RateConstant(value=2.3e-12, name='k32', unit='cm3s-1', reaction_name='R32'),
    RateConstant(value=1.0e-30*T**(-0.8), name='k33', unit='cm6s-1', reaction_name='R33'),
    #       RateConstant(value=6.0e-12, name='k34'),
    RateConstant(value=2.4e-12*sym.exp(-1710/T), name='k35', unit='cm3s-1', reaction_name='R35'),
    RateConstant(value=3.5e-11*sym.exp(-4550/T), name='k36', unit='cm3s-1', reaction_name='R36'),
    RateConstant(value=1.4e-10, name='k37', unit='cm3s-1', reaction_name='R37')]

#set available rate constant to each reaction
reaction_lst = [r.set_rate_constant(next(k for k in rate_const_lst if k.reaction_name==r.name)) for r in reaction_lst]

# %% pick some of the reactions
reaction_set = ['R{}'.format(n) for n in '3 14 15 17 18 21 28 29'.split()]
reaction_lst = [r for r in reaction_lst if r.name in reaction_set]
species_lst = list(set(reduce(add, [r.reactants + r.products for r in reaction_lst])))

# generate a list of (coeff, r_stoich, net_stoich)
coeff_lst = [r.rate_constant.name for r in reaction_lst]
r_stoich_lst = [r.get_r_stoich() for r in reaction_lst]
net_stoich_lst = [r.get_net_stoich() for r in reaction_lst]
reactions = list(zip(coeff_lst, r_stoich_lst, net_stoich_lst))
species_names = [s.name for s in species_lst]

# set up all y and ydot equations
ydot, y, k = mk_exprs_symbs(reactions, species_names) 

# %%
# set fixed species
species_fix = [n2, o2, m] #[n2, o2, m, h2, h2o, ch4]
y_fixed = tuple([y[species_lst.index(s)] for s in species_fix])

# set equilibrium species
# species_eq = [o1d]
# y_solve_eq = sym.solve([ydot[species_lst.index(s)] for s in species_eq], 
#                  [y[species_lst.index(s)] for s in species_eq])

# set non-equilibrium species
# species_non_eq = [o3, o, h, h2o2, oh, ho2]
# ydot_non_eq = [ydot[species_lst.index(s)].subs(y_solve_eq) for s in species_non_eq]
# y_non_eq = [y[species_lst.index(s)] for s in species_non_eq]
# jcb = sym.Matrix(ydot_non_eq).jacobian(y_non_eq)  
# f_ode = sym.lambdify((y_non_eq, t) + k + y_fixed, ydot_non_eq)
# f_jcb = sym.lambdify((y_non_eq, t) + k + y_fixed, jcb)  

# set up ode (ydot) equations of each family
HOx, Ox = sym.symbols('HOx, Ox', real=True, nonnegative=True, positive=True)
families = {HOx: [oh, ho2, h], Ox: [o, o3]}
y_solve_family = family_equilibrium(families, reactions, species_lst)
ydot_non_eq = [sym.lambdify(y_solve_family.keys(),
    reduce(add, [ydot[species_lst.index(s)] for s in family_species]))(*y_solve_family.values())
    for family_species in families.values()]
# ydot_non_eq = [reduce(add, [ydot[species_lst.index(s)] 
#     for s in family_species]).subs(y_solve_family)
#     for family_species in families.values()]

y_non_eq = list(families.keys()) 
f_ode = sym.lambdify((y_non_eq, t) + k + y_fixed, ydot_non_eq)
jcb = sym.Matrix(ydot_non_eq).jacobian(y_non_eq)  
f_jcb = sym.lambdify((y_non_eq, t) + k + y_fixed, jcb)  

#%%
# set values
js_ds = xr.open_dataset('./Js_table_w0.nc')
ds = xr.open_dataset('./s&b_tables_anqi.nc')
ds = ds.rename({'o1d': o1d.name.lower()})
z = np.arange(50, 105, 5) #km
# P_val = ds.p.interp(z=z).drop('index') *1e2 #Pa
T_val = ds.T.interp(z=z).drop('index') #K
m = m.set_density(ds.m.interp(z=z).drop('index')) #cm-3
o2 = o2.set_density(m.density.rename('o2') * 0.2) #cm-3
n2 = n2.set_density(m.density.rename('n2') * 0.8) #cm-3
h2.set_density(ds.h2.interp(z=z).drop('index')) #cm-3
h2o.set_density(ds.h2o.interp(z=z).drop('index')) #cm-3
h2o2.set_density(ds.h2o2.interp(z=z).drop('index')) #cm-3
ch4.set_density(ds.ch4.interp(z=z).drop('index')) #cm-3
co.set_density(ds.co.interp(z=z).drop('index')) #cm-3
y_fixed_vals = tuple([s.density for s in species_fix])
func_ks = tuple(sym.lambdify(T, r.rate_constant.value) for r in reaction_lst)
D = np.zeros(len(y_non_eq)) #diffusion coefficient for each non_eq species

#%%
def get_js(t,z):
    js = js_ds.interp(t=t, kwargs=dict(fill_value=0)
                    ).interp(z=z, kwargs=dict(fill_value='extrapolate'))
    return js.J1, js.J2, js.J3, js.J4, js.J7
def set_js(t,z, k_vals):
    Js = get_js(t, z)
    for j in range(1,8): #loop over Js from get_js function
        for i in range(len(k_vals)):
            if 'J{}'.format(j) in reaction_lst[i].rate_constant.name:
                k_vals[i] = Js[j]
    return k_vals

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
        # k_vals_z[:4] = list(get_js(t,z[i]))[:4] 
        k_vals_z[0] = get_js(t, z[i])[2]
        # k_vals_z = set_js(t, z[i], k_vals_z)
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        f_out[slc] += f_ode(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
    return f_out
    # k_vals = tuple(list(get_js(t, z)) + [func_k(T_val) for func_k in func_ks] )
    # ydot_vals_flat = np.ravel(f_ode(y_flat.reshape(len(y_non_eq), len(z)), t, *k_vals, *y_fixed_vals))
    # return ydot_vals_flat

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
        # k_vals_z[:4] = list(get_js(t,z[i]))[:4]
        k_vals_z[0] = get_js(t, z[i])[2]
        # k_vals_z = set_js(t, z[i], k_vals_z)
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        j_out[slc, slc] = f_jcb(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
        #diffusion contributions
        k = np.clip(i, 1, len(y_non_eq) - 2)  #not sure what it means
        for j in range(len(y_non_eq)): 
            j_out[i*len(y_non_eq) + j, (k-1)*len(y_non_eq) + j] +=    D[j]/np.gradient(z)[i]**2
            j_out[i*len(y_non_eq) + j, (k  )*len(y_non_eq) + j] += -2*D[j]/np.gradient(z)[i]**2
            j_out[i*len(y_non_eq) + j, (k+1)*len(y_non_eq) + j] +=    D[j]/np.gradient(z)[i]**2
    return j_out


# %%
%%time
# tout = np.arange(5.55*3600, 5.8*3600, 100) #s
# y0 = np.ravel([reduce(add,[ds[s.name.lower()].interp(z=z) 
#                 for s in families[familyname]]) 
#                 for familyname in families.keys()], order='F')
yout_save = yout.copy()
y0 = np.ravel([yout_save[familyname.name].isel(t=-1) for familyname in families.keys()], order='F')
tout = np.arange(yout_save.t[-1]*3600, (yout_save.t[-1]+5)*3600, 100)  #s

yout_org, info = odeint(f_ode_flat, y0, tout, args=func_ks, Dfun=f_jcb_flat,
                        full_output=True, #atol=1e-9, rtol=1e-9,
                        tcrit=np.array([6, 18, 30, 42])*3600)
yout = yout_org.reshape(len(tout), len(z), len(y_non_eq))
yout = xr.DataArray(yout, dims=('t', 'z', 'species'), 
                    coords=(tout/3600, z, [s.name for s in families.keys()]))
yout = yout.to_dataset(dim='species')

# print("The function was evaluated %d times." % info['nfe'][-1])
# print("The Jacobian was evaluated %d times." % info['nje'][-1])

# %% familiy species divisions
k_vals = [func_k(T_val) for func_k in func_ks]
# k_vals[1] = get_js(yout.t*3600, yout.z).assign_coords(t=yout.t)
k_vals = set_js(yout.t*3600, yout.z, k_vals)
for s in reduce(add, families.values()):
    f_density = sym.lambdify(tuple(families.keys())+k+y_fixed, y_solve_family[y[species_lst.index(s)]])
    s.density = f_density(*[yout[s.name] for s in families.keys()], *k_vals, *y_fixed_vals)
    yout = yout.update({s.name: s.density})

yout = xr.auto_combine([yout, yout_save])
# rm ./ode_result/ode_simple.nc
# yout.to_netcdf('./ode_result/ode_simple.nc')

#%% plot results
# yout = xr.open_dataset('./ode_result/ode_frederick.nc')

fig, ax = plt.subplots(1,2)
line_plot_args = dict(x='t', hue='z',yscale='log', xscale='linear')
yout.O3.plot.line(**line_plot_args, ax=ax[0])
yout_ratio = yout/yout.isel(t=0)
line_plot_args = dict(x='t', hue='z',yscale='linear', xscale='linear')
yout_ratio.O3.plot.line(**line_plot_args, ax=ax[1])
ax[0].set_title('ND')
ax[1].set_title('Ratio to t0')
plt.show()

# %%
