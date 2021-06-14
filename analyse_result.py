#%%
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#%% test on random file
filename = 'Oxfam_more_HOx_9.nc'
ds = xr.open_dataset('./ode_result/'+filename)
ds = ds.sel(t=~ds.indexes['t'].duplicated())
fig, ax = plt.subplots(1,2)
ds.O2_Delta.plot.line(ax=ax[0], x='t', yscale='log', ylim=(1e8, 1e13))
ds.O2_Delta.sel(t=[0, 20, 35], method='nearest').plot.line(ax=ax[1], y='z', xscale='log')
ax[0].set_xticks(np.arange(12,250, 48))
plt.suptitle(filename)
plt.show()

#%%
def clear_replicate(ds):
    ds = ds.sel(t=~ds.indexes['t'].duplicated())
    return ds

def plot_species(ds, species, ylim=(1e6, 1e12), rm_nights=False):
    if rm_nights:
        cond = np.logical_and(ds.t%24>6, ds.t%24<18)
    else:
        cond = ds.t>0
    ds[species].where(cond).plot.line(x='t', yscale='log', 
                            ylim=ylim, add_legend=True
                            )
    ax = plt.gca()
    ax.get_legend().set(bbox_to_anchor=(1, 1), title='z [km]') 
    ax.set_xticks(np.arange(ds.t[0], ds.t[-1], 12, int))
    ax.set_xticklabels(ax.get_xticks()%24)

    plt.suptitle(filename.format(model_round))
    plt.show()
    return

def plot_species_contourf(ds, species):
    fig = plt.figure()
    levels = np.logspace(6, 12)
    ds[species].plot.contourf(x='t', levels=levels, cmap='viridis')
    # ds.O2_Delta.plot(x='t', norm=LogNorm(vmin=1e6, vmax=5e11), cmap='viridis')

    ax = plt.gca()
    ax.set_xticks(np.arange(ds.t[0], ds.t[-1], 12, int))
    ax.set_xticklabels(ax.get_xticks()%24)
    plt.show()
    return

def plot_compare_species(ds0, ds1, species, ylim=(-100, 100), rm_nights=False):

    ds_compare = ds1.interp(t=ds0.t, kwargs=dict(fill_value=0)
        ).pipe(lambda x: 100*(x-ds0)/ds0)
    if rm_nights:
        cond = np.logical_and(ds_compare.t%24>6, ds_compare.t%24<18)
        ds_compare = ds_compare.where(cond)
    
    plt.figure()
    ds_compare[species].rename('% differences').plot.line(x='t', ylim=ylim)
    ax = plt.gca()
    ax.get_legend().set(bbox_to_anchor=(1, 1), title='z [km]') 
    ax.set_xticks(np.arange(ds_compare.t[0], ds_compare.t[-1], 12, int))
    ax.set_xticklabels(ax.get_xticks()%24)
    ax.set(title='{} - {}'.format(ds0.model_round[6:-9], ds1.model_round[6:-9]))
    plt.show()
    return ds_compare

#%% individual cases
species = 'O3'
plot_args = dict(ylim=(1e6, 1e13), rm_nights=True)
#%
filename = 'NewJs_normal_HOx_{}.nc'
model_round = 9
ds0 = xr.open_dataset('./ode_result/' + filename.format(model_round))
ds0 = clear_replicate(ds0)
plot_species(ds0, species, **plot_args)
# plot_species_contourf(ds0, species)
#%%
filename = 'NewJs_half_HOx_{}.nc'
model_round = 9
ds1 = xr.open_dataset('./ode_result/' + filename.format(model_round))
ds1 = clear_replicate(ds1)
plot_species(ds1, species, **plot_args)
#%%
filename = 'NewJs_double_HOx_{}.nc'
model_round = 9
ds2 = xr.open_dataset('./ode_result/' + filename.format(model_round))
ds2 = clear_replicate(ds2)
plot_species(ds2, species, **plot_args)

#%% Compare
plot_compare_species(ds0, ds1, species, ylim=(-10, 250))
plt.title('Half - Normal')
#%
plot_compare_species(ds0, ds2, species, ylim=(-120, 10))
plt.title('Double - Normal')

#%% Compare more
model_round = 9
def get_data(filename, model_round=9):
    ds0 = xr.open_dataset('./ode_result/' + filename.format(model_round))
    ds0 = clear_replicate(ds0).assign_attrs({'model_round': filename.format(model_round)})
    return ds0

filename = 'NewJs_{}_HOx_{}.nc'
ds0 = get_data(filename.format('normal', {}))
ds1 = get_data(filename.format('25more', {}))
ds2 = get_data(filename.format('25less', {}))
ds3 = get_data(filename.format('50more', {}))
ds4 = get_data(filename.format('half', {}))
ds5 = get_data(filename.format('75more', {}))
ds6 = get_data(filename.format('75less', {}))
ds7 = get_data(filename.format('double', {}))
ds8 = get_data(filename.format('100less', {}))

#%%
args = dict(species='O3', ylim=(-500, 500), rm_nights=True)
_ = plot_compare_species(ds0, ds1, **args)
_ = plot_compare_species(ds0, ds2, **args)
_ = plot_compare_species(ds0, ds3,  **args)
_ = plot_compare_species(ds0, ds4, **args)
_ = plot_compare_species(ds0, ds5, **args)
_ = plot_compare_species(ds0, ds6, **args)
_ = plot_compare_species(ds0, ds7, **args)
_ = plot_compare_species(ds0, ds8, **args)


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
