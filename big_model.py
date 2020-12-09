# %%
from chem_classes import *
from chem_sympy import *
from o2delta_model import gA
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# %% [Set up the system (species involved and list of reactions)]
# reaction_lst = get_reactions_frederick1979() + get_reactions_thomas1984()
reaction_lst = get_reactions_custom2()
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
species_eq = [o1d, o2sig]
y_eq = tuple([y[species_lst.index(s)] for s in species_eq])
solutions, = sym.solve_poly_system([ydot[species_lst.index(s)] for s in species_eq], 
            *y_eq)
y_solve_eq_0 = dict(zip([y[species_lst.index(s)] for s in species_eq],
                    solutions))

#%%
# set up ode (ydot) equations of each family
HOx, Ox = sym.symbols('HOx, Ox', real=True, nonnegative=True, positive=True)
families = {HOx: [oh, ho2, h], Ox: [o, o3]}
y_non_eq = list(families.keys()) + [y[species_lst.index(o2del)]] 
y_solve_family = family_equilibrium(families, reactions, species_lst)
y_solve_eq = dict(zip(y_eq, [sym.lambdify(y_solve_family.keys(), eq
        )(*y_solve_family.values()) for eq in y_solve_eq_0.values()]))

ydot_non_eq = [
    sym.lambdify(list(y_solve_family.keys()) + list(y_solve_eq.keys()),
    reduce(add, [ydot[species_lst.index(s)] for s in family_species]))(
    *y_solve_family.values(), *y_solve_eq.values())
    for family_species in families.values()
    ] + [
    sym.lambdify(list(y_solve_family.keys()) + list(y_solve_eq.keys()), 
    ydot[species_lst.index(o2del)]) (*y_solve_family.values(), *y_solve_eq.values())] 

t = sym.symbols('t', real=True, nonnegative=True, positive=True)
f_ode = sym.lambdify((y_non_eq, t) + k + y_fixed, ydot_non_eq)
jcb = sym.Matrix(ydot_non_eq).jacobian(y_non_eq)  
f_jcb = sym.lambdify((y_non_eq, t) + k + y_fixed, jcb)  

#%%
# set values
js_ds = xr.open_dataset('./Js_table_v4.nc')
ds = xr.open_dataset('./s&b_tables_anqi.nc')
ds = ds.rename({'o1d': o1d.name.lower()})
z = np.arange(50, 105, 5) #km
P_val = ds.p.interp(z=z).drop('index') *1e2 #Pa
T_val = ds.T.interp(z=z).drop('index') #K
m = m.set_density(ds.m.interp(z=z).drop('index')) #cm-3
o2 = o2.set_density(m.density.rename('o2') * 0.2) #cm-3
n2 = n2.set_density(m.density.rename('n2') * 0.8) #cm-3
o3 = o3.set_density(ds.o3.interp(z=z).drop('index'))

y_fixed_vals = tuple([s.density for s in species_fix])
func_ks = tuple(sym.lambdify(T, r.rate_constant.value) for r in reaction_lst)
D = np.zeros(len(y_non_eq)) #diffusion coefficient for each non_eq species
#%%
def func_sza(t=np.linspace(0, 3600*24), elv_max=90):
    #t in s
    elv = elv_max - 90*((np.cos(2*np.pi/(3600*24) * t)) + 1)
    return 90-elv

def set_J_vals(t, z, k_vals):
    # Translate t [s] to 0-24h range
    t_ = t%(24*3600)

    # Photolysis rate
    js = js_ds.interp(t=t_, kwargs=dict(fill_value=0)
                    ).interp(z=z, kwargs=dict(fill_value='extrapolate'))

    # Solar excitation (gA)
    sza = func_sza(t)
    gA_val = gA(P_val.interp(z=z, kwargs=dict(fill_value='extrapolate')), sza)
    if isinstance(t, float):
        if sza >= 90:
            gA_val = np.zeros_like(z)
    else:
        gA_val = xr.DataArray(gA_val, coords=(z, t.t), dims=('z', 't'))
        gA_val = gA_val.where(sza<90).fillna(0)

    # Set them into k_vals vector
    for i in [idx for idx, name in enumerate([r.rate_constant.name for r in reaction_lst]) 
                if 'J' in name or 'g_' in name]:
    # for i in range(len(reaction_lst)):
        if reaction_lst[i].rate_constant.name == 'J1':
            k_vals[i] = js.SRB + js.Herzberg
        elif reaction_lst[i].rate_constant.name == 'J2':
            k_vals[i] = js.Lya + js.SRC
        elif reaction_lst[i].rate_constant.name == 'J3':
            if 'J4' in [r.rate_constant.name for r in reaction_lst]:
                k_vals[i] = js.Huggins + js.Chappuis
            elif 'J_o3' in [r.rate_constant.name for r in reaction_lst]:
                k_vals[i] = js.Huggins + js.Chappuis
            else:
                k_vals[i] = js.Huggins + js.Chappuis + js.Hartley
        elif reaction_lst[i].rate_constant.name == 'J4':
            k_vals[i] = js.Hartley
        elif reaction_lst[i].rate_constant.name == 'J_o3':
            k_vals[i] = js.Hartley
        elif reaction_lst[i].rate_constant.name == 'J7':
            k_vals[i] = js.h2o2
        elif reaction_lst[i].rate_constant.name == 'g_A':
            k_vals[i] = gA_val
    return k_vals
    
# def get_js(t,z):
#     t_ = t%(24*3600)
#     js = js_ds.interp(t=t_, kwargs=dict(fill_value=0)
#                     ).interp(z=z, kwargs=dict(fill_value='extrapolate'))
#     sza = func_sza(t)
#     gA_val = gA(P_val.interp(z=z, kwargs=dict(fill_value='extrapolate')), sza)
#     if isinstance(t, float):
#         if sza >= 90:
#             gA_val = np.zeros_like(z)
#     else:
#         gA_val = xr.DataArray(gA_val, coords=(z, t.t), dims=('z', 't'))
#         gA_val = gA_val.where(sza<90).fillna(0)
#     return js.J1, js.J2, js.J3, js.J4, js.J7, gA_val

# def set_js(t,z, k_vals):
#     Js = get_js(t, z)
#     for i in range(len(reaction_lst)):
#         if 'J1' in reaction_lst[i].rate_constant.name:
#             k_vals[i] = Js[0]
#         elif 'J2' in reaction_lst[i].rate_constant.name:
#             k_vals[i] = Js[1]
#         elif 'J3' in reaction_lst[i].rate_constant.name:
#             if 'J4' in [r.rate_constant.name for r in reaction_lst]:
#                 k_vals[i] = Js[2] - Js[3]
#             elif 'J_o3' in [r.rate_constant.name for r in reaction_lst]:
#                 k_vals[i] = Js[2] - Js[3]
#             else:
#                 k_vals[i] = Js[2]
#         elif 'J4' in reaction_lst[i].rate_constant.name:
#             k_vals[i] = Js[3]
#         elif 'J7' in reaction_lst[i].rate_constant.name:
#             k_vals[i] = Js[4]
#         elif 'J_o3' in reaction_lst[i].rate_constant.name:
#             k_vals[i] = Js[3]
#         elif 'g_A' in reaction_lst[i].rate_constant.name:
#             k_vals[i] = Js[-1]
#     return k_vals

def f_ode_flat(y_flat_in, t, *func_ks):
    print(t/3600)
    y_flat = y_flat_in.copy()
    y_flat[y_flat<1e-10] = 1e-10
    f_out = np.empty(len(y_non_eq)*len(z))
    for i in range(len(z)):
        slc = slice(i*len(y_non_eq), (i+1)*len(y_non_eq))
        #diffusion contributions
        f_out[slc] = second_derivatives_spatial(i, y_flat, f_out[slc]) 
        f_out[slc] *= D
        # chemical contributions
        k_vals_z = [func_k(T_val[i]) for func_k in func_ks]
        k_vals_z = set_J_vals(t, z[i], k_vals_z)
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        f_out[slc] += f_ode(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
    return f_out
    # k_vals = tuple(list(get_js(t, z)) + [func_k(T_val) for func_k in func_ks] )
    # ydot_vals_flat = np.ravel(f_ode(y_flat.reshape(len(y_non_eq), len(z)), t, *k_vals, *y_fixed_vals))
    # return ydot_vals_flat

def f_jcb_flat(y_flat_in, t, *func_ks):
    y_flat = y_flat_in.copy()
    y_flat[y_flat<1e-10] = 1e-10
    j_out = np.zeros((len(y_flat), len(y_flat)))  # dense matrix
    for i in range(len(z)):
        #chemical contributions
        slc = slice(i*len(y_non_eq), (i+1)*len(y_non_eq))
        k_vals_z = [func_k(T_val[i]) for func_k in func_ks]
        k_vals_z = set_J_vals(t, z[i], k_vals_z)
        y_fixed_vals_z = [c[i] for c in y_fixed_vals]
        j_out[slc, slc] = f_jcb(y_flat[slc], t, *tuple(k_vals_z), *tuple(y_fixed_vals_z)) 
        #diffusion contributions
        k = np.clip(i, 1, len(y_non_eq) - 2)  #not sure what it means
        for j in range(len(y_non_eq)): 
            j_out[i*len(y_non_eq) + j, (k-1)*len(y_non_eq) + j] +=    D[j]/np.gradient(z)[i]**2
            j_out[i*len(y_non_eq) + j, (k  )*len(y_non_eq) + j] += -2*D[j]/np.gradient(z)[i]**2
            j_out[i*len(y_non_eq) + j, (k+1)*len(y_non_eq) + j] +=    D[j]/np.gradient(z)[i]**2
    return j_out

def second_derivatives_spatial(i, state_flat, out):
    k = np.clip(i, 1, len(y_non_eq) - 2)  #not sure what it means
    out = np.empty(len(y_non_eq))
    for j in range(len(y_non_eq)):
        left = state_flat[(k-1)*len(y_non_eq) + j]
        cent = state_flat[(k  )*len(y_non_eq) + j]
        rght = state_flat[(k+1)*len(y_non_eq) + j]
        out[j] = (left - 2*cent + rght)/np.gradient(z)[i]**2
    return out

# %% integration
# to make y0 on o2del
# f_o2del_eq = sym.lambdify(tuple([y[species_lst.index(o3)],]) + y_fixed + k, 
#     sym.solve(ydot[species_lst.index(o2del)], y[species_lst.index(o2del)])[0].subs(y_solve_eq_0))
# k_vals = [func_k(T_val) for func_k in func_ks]
# k_vals[7], k_vals[-1] = get_js(tout[0], z)[3], get_js(tout[0], z)[-1]
# # k_vals = set_js(tout[0], z, k_vals)
# o2del = o2del.set_density(f_o2del_eq(o3.density, *y_fixed_vals,  *k_vals))

filename = 'NewJs_100less_HOx_{}.nc'
model_round = 1
while model_round<10: 
    print(filename.format(model_round))
    
    if model_round == 0:
        tout = np.arange(12*3600, (12+24)*3600, 100) #s
        o2del = o2del.set_density(xr.open_dataarray('o2del_initial.nc'))
        y0 = [reduce(add,[ds[s.name.lower()].interp(z=z) 
                        for s in families[familyname]]) 
                        for familyname in families.keys()
                        ]+[o2del.density]
        # y0[0] = y0[0] * 1.5 #increase/decrease HOx
    else:
        yout_save = xr.open_dataset('./ode_result/' + filename.format(model_round-1))
        t_last = yout_save.Ox.where(yout_save.Ox!=0, drop=True).t[-1]
        y0 = [yout_save.sel(t=t_last).HOx, yout_save.sel(t=t_last).Ox, yout_save.sel(t=t_last).O2_Delta]
        tout = np.arange(t_last*3600, (t_last+24)*3600, 100) #s
    
    y0 = np.ravel(y0, order='F')
    yout_org, info = odeint(f_ode_flat, y0, tout, args=func_ks, Dfun=f_jcb_flat,
                            full_output=True, atol=1e-7, rtol=1e-7,
                            # tcrit=np.array([6, 30])*3600
                        )
    yout = yout_org.reshape(len(tout), len(z), len(y_non_eq))
    yout = xr.DataArray(yout, dims=('t', 'z', 'species'), 
                        coords=(tout/3600, z, [s.name for s in y_non_eq]))
    yout = yout.to_dataset(dim='species')

    # print("The function was evaluated %d times." % info['nfe'][-1])
    # print("The Jacobian was evaluated %d times." % info['nje'][-1])

    #%% familiy species divisions
    k_vals = [func_k(T_val) for func_k in func_ks]
    k_vals = set_J_vals(yout.t*3600, yout.z, k_vals)
    for s in reduce(add, families.values()):
        f_density = sym.lambdify(tuple(families.keys())+k+y_fixed, y_solve_family[y[species_lst.index(s)]])
        s.density = f_density(*[yout[s.name] for s in families.keys()], *k_vals, *y_fixed_vals)
        yout = yout.update({s.name: s.density})
    for s in species_eq:
        f_density = sym.lambdify(tuple(families.keys())+k+y_fixed, y_solve_eq[y[species_lst.index(s)]])
        s.density = f_density(*[yout[s.name] for s in families.keys()], *k_vals, *y_fixed_vals)
        yout = yout.update({s.name: s.density})
    
    if model_round == 0:
        yout_save = yout
    else:
        yout_save = xr.auto_combine([yout_save.sel(t=slice(t_last)), yout])
        
    yout_save.to_netcdf('./ode_result/' + filename.format(model_round))
    model_round += 1
        

#%% Post processing
# filename = 'Normal_HOx_{}.nc'
# model_round = 0
# yout = xr.open_dataset('./ode_result/'+filename.format(model_round))
# k_vals = [func_k(T_val) for func_k in func_ks]
# k_vals = set_js(yout.t*3600, yout.z, k_vals)
# for s in reduce(add, families.values()):
#     f_density = sym.lambdify(tuple(families.keys())+k+y_fixed, y_solve_family[y[species_lst.index(s)]])
#     s.density = f_density(*[yout[s.name] for s in families.keys()], *k_vals, *y_fixed_vals)
#     yout = yout.update({s.name: s.density})
# for s in species_eq:
#     f_density = sym.lambdify(tuple(families.keys())+k+y_fixed, y_solve_eq[y[species_lst.index(s)]])
#     s.density = f_density(*[yout[s.name] for s in families.keys()], *k_vals, *y_fixed_vals)
#     yout = yout.update({s.name: s.density})

# # yout.to_netcdf('./ode_result/' + 'cplt_' + filename.format(model_round))
#%%
# s = 'O2_Sigma'
# fig, ax = plt.subplots(1,2)
# yout[s].plot.line(ax=ax[0], x='t', yscale='log')
# yout[s].sel(t=[0, 24, 35], method='nearest').plot.line(ax=ax[1], y='z', xscale='log')
# # ax[0].get_legend().set(bbox_to_anchor=(1, 1), title='z [km]') 

# plt.suptitle(filename.format(model_round))
# plt.show()

#%%
# y_loss = loss_terms(reactions, species_names)
# lifetime = xr.Dataset()
# for s_focus in [o3, o, Ox]:
#     if s_focus in families.keys():
#         y_loss_s = reduce(add, [y_loss[species_lst.index(s)] for s in families[s_focus]])
#     else:
#         y_loss_s = y_loss[species_lst.index(s_focus)]
#     f_loss_s = sym.lambdify((y,)+k, y_loss_s)
#     all_data = xr.auto_combine([da.rename(da.name.upper()).to_dataset() for da in y_fixed_vals]).update(yout)
#     loss_s_vals = f_loss_s([all_data[s.name] for s in y], *k_vals)
#     lifetime_s_vals = all_data[s_focus.name]/loss_s_vals

#     lifetime = lifetime.update({s_focus.name: lifetime_s_vals})
#%%
# lifetime.to_array().sel(t=35, method='nearest').plot.line(y='z', xscale='log')
# plt.gca().set_xticks([60, 60**2, 60**2*24, 60**2*24*30])
# plt.gca().set_xticklabels('min hour day month'.split())
# #%%
# s_focus = Ox
# fig, ax = plt.subplots(1,2)
# lifetime[s_focus.name].plot.line(ax=ax[0], x='t', yscale='log')
# ax[0].set_yticks([60, 60**2, 60**2*24, 60**2*24*30])
# ax[0].set_yticklabels('min hour day month'.split())

# lifetime[s_focus.name].sel(t=[12, 24, 35], method='nearest').plot.line(ax=ax[1], y='z', xscale='log')
# ax[1].set_xticks([60, 60**2, 60**2*24, 60**2*24*30])
# ax[1].set_xticklabels('min hour day month'.split())

# plt.suptitle('lifetime of {}'.format(s_focus.name))
# plt.show()
#%% plot results
# yout = xr.open_dataset('./ode_result/ode_frederick.nc')

# fig, ax = plt.subplots(1,2)
# line_plot_args = dict(x='t', hue='z',yscale='log', xscale='linear')
# yout.O3.plot.line(**line_plot_args, ax=ax[0])
# yout_ratio = yout/yout.isel(t=0)
# line_plot_args = dict(x='t', hue='z',yscale='linear', xscale='linear')
# yout_ratio.O3.plot.line(**line_plot_args, ax=ax[1])
# ax[0].set_title('ND')
# ax[1].set_title('ratio to t0')
# plt.show()


#%%
# J = []
# for t_s in yout.t*3600:
#     J.append(get_js(t_s, yout.z))
# J = xr.DataArray(J, dims=('t', 'Js', 'z'), 
#         coords=(yout.t, ['J1', 'J2', 'J3', 'J4', 'J7', 'gA'], yout.z)).to_dataset('Js')