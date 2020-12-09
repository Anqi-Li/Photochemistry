#%%
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#%% test on random file
# filename = 'More_50_HOx_9.nc'
# ds = xr.open_dataset('./ode_result/'+filename)
# ds = ds.sel(t=~ds.indexes['t'].duplicated())
# fig, ax = plt.subplots(1,2)
# ds.O2_Delta.plot.line(ax=ax[0], x='t', yscale='log')
# ds.O2_Delta.sel(t=[0, 20, 35], method='nearest').plot.line(ax=ax[1], y='z', xscale='log')
# plt.suptitle(filename)
# plt.show()

#%%
def clear_replicate(ds):
    ds = ds.sel(t=~ds.indexes['t'].duplicated())
    return ds

def plot_species(ds, species):
    fig = plt.figure()
    ds[species].plot.line(x='t', yscale='linear', 
                            ylim=(1e6, 1e12), add_legend=True
                            )
    ax = plt.gca()
    ax.get_legend().set(bbox_to_anchor=(1, 1), title='z [km]') 
    ax.set_xticks(np.arange(ds.t[0], ds.t[-1], 12, int))
    ax.set_xticklabels(ax.get_xticks()%24)

    # fig, ax = plt.subplots(1,2)
    # ds.O2_Delta.plot.line(ax=ax[0], x='t', yscale='log', ylim=(1e6, 5e11))
    # ds.O2_Delta.sel(t=[0, 20, 35], method='nearest').plot.line(ax=ax[1], y='z', xscale='log')
    plt.suptitle(filename.format(model_round))
    plt.show()

def plot_species_contourf(ds, species):
    fig = plt.figure()
    levels = np.logspace(6, 12)
    ds[species].plot.contourf(x='t', levels=levels, cmap='viridis')
    # ds.O2_Delta.plot(x='t', norm=LogNorm(vmin=1e6, vmax=5e11), cmap='viridis')

    ax = plt.gca()
    ax.set_xticks(np.arange(ds.t[0], ds.t[-1], 12, int))
    ax.set_xticklabels(ax.get_xticks()%24)
    # cbar = fig.colorbar(im[0], ticks=[1e7, 1e8, 1e9, 1e10, 1e11], cax=ax_colorbar, extend='min', cmap='RdBu_r', label='Number density /cm-3')
    # cbar.ax.set_yticklabels(['$10^7$', '$10^8$', '$10^9$', '$10^{10}$', '$10^{11}$'])
    plt.show()

def plot_compare_species(ds0, ds1, species, ylim):
    ds_compare = ds1.interp(t=ds0.t, kwargs=dict(fill_value=0)
        ).pipe(lambda x: 100*(x-ds0)/ds0)
    ds_compare = ds_compare.where(np.logical_and(ds_compare.t%24>6, ds_compare.t%24<18))
    plt.figure()
    ds_compare[species].rename('% differences').plot.line(x='t', ylim=ylim)
    ax = plt.gca()
    ax.get_legend().set(bbox_to_anchor=(1, 1), title='z [km]') 
    ax.set_xticks(np.arange(ds_compare.t[0], ds_compare.t[-1], 12, int))
    ax.set_xticklabels(ax.get_xticks()%24)
    # plt.show()
    return

#%% individual cases
species = 'O2_Delta'
#%
filename = 'NewJJ_normal_HOx_{}.nc'
model_round = 9
ds0 = xr.open_dataset('./ode_result/' + filename.format(model_round))
ds0 = ds0.sel(t=~ds0.indexes['t'].duplicated())
# plot_species(ds0, species)
plot_species_contourf(ds0, species)
#%%
filename = 'NewJJ_half_HOx_{}.nc'
model_round = 9
ds1 = xr.open_dataset('./ode_result/' + filename.format(model_round))
ds1 = ds1.sel(t=~ds1.indexes['t'].duplicated())
plot_species(ds1, species)
#%
filename = 'NewJJ_double_HOx_{}.nc'
model_round = 9
ds2 = xr.open_dataset('./ode_result/' + filename.format(model_round))
ds2 = ds2.sel(t=~ds2.indexes['t'].duplicated())
plot_species(ds2, species)

#%% Compare
plot_compare_species(ds0, ds1, species, ylim=(-10, 250))
plt.title('Half - Normal')
#%
plot_compare_species(ds0, ds2, species, ylim=(-120, 10))
plt.title('Double - Normal')

# %%
filename = 'Half_HOx_9.nc'
ds0 = clear_replicate(xr.open_dataset('./ode_result/' + filename))
filename = 'NewJ_half_HOx_9.nc'
ds1 = clear_replicate(xr.open_dataset('./ode_result/' + filename))
filename = 'NewJJ_half_HOx_9.nc'
ds2 = clear_replicate(xr.open_dataset('./ode_result/' + filename))
species = 'O2_Delta'
plot_compare_species(ds0, ds1, species, ylim=(-100, 100))
plot_compare_species(ds0, ds2, species, ylim=(-100, 100))
plot_compare_species(ds1, ds2, species, ylim=(-100, 100))
plt.title('two interpolations')

# %%
