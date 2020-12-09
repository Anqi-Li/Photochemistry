# %%
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import copy
from scipy.integrate import odeint
s2h = 1/3600
h2s = 3600

from chem_classes import *

#%% cross sections
with xr.open_dataset('/home/anqil/Documents/Python/Photochemistry/xs_anqi.nc') as ds:
    ds = ds.sel(wavelength=~ds.indexes['wavelength'].duplicated())
    wavelength = ds.wavelength.values
    ds.so2.loc[dict(wavelength=ds.wavelength[np.where(ds.so2==0)[0][0]])
        ] = ds.so2.loc[dict(wavelength=ds.wavelength[np.where(ds.so2==0)[0][0]-1])]/10
    so2 = ds.so2.where(ds.so2!=0).pipe(lambda x: np.log(x)
        ).interpolate_na('wavelength'
        ).pipe(lambda x: np.exp(x)
        ).fillna(0).values
    # so2 = ds.so2.values
    so3 = ds.so3.values
    sh2o = ds.sh2o.values
    sh2o2 = ds.sh2o2.values
    irrad = ds.irrad.values


# %% [Set up the known densities and rate conctants]
with xr.open_dataset('./s&b_tables_anqi.nc') as ds:
    z = np.arange(50, 105, 5)
    T = ds.T.interp(z=z).drop('index')
    m.set_density(ds.m.interp(z=z).drop('index')).set_unit('cm-3')
    o2.set_density(m.density*0.2).set_unit('cm-3')
    n2.set_density(m.density*0.8).set_unit('cm-3')
    # h2o2.set_density(ds.h2o2.interp(z=z).drop('index')).set_unit('cm-3')
    # h2.set_density(ds.h2.interp(z=z).drop('index')).set_unit('cm-3')
    # h2o.set_density(ds.h2o.interp(z=z).drop('index')).set_unit('cm-3')
    # ch4.set_density(ds.ch4.interp(z=z).drop('index')).set_unit('cm-3')
    # co.set_density(ds.co.interp(z=z).drop('index')).set_unit('cm-3')

# %% [Functions]
def func_sza(t=np.linspace(0, 3600*24), elv_max=90):
    #t in s
    elv = elv_max - 90*((np.cos(2*np.pi/(3600*24) * t)) + 1)
    return 90-elv

def path_z (z_top, z_t, sol_zen, nsteps):
    if sol_zen == 0:
        z_step = np.linspace(z_top, z_t, nsteps)
        step = (z_top - z_t)/nsteps
    elif sol_zen == 180:
        z_step = np.zeros(nsteps)
        step = 0
    else:     
        Re = 6375 #km (should be the same unit as z)
        sol_zen /= 180/np.pi
        B=np.arcsin((Re+z_t) * np.sin(np.pi-sol_zen)/(Re+z_top))
        S_top=np.sin(sol_zen-B)*(Re+z_top)/np.sin(np.pi-sol_zen)

        Ret2=(Re + z_t)**2
        step=S_top/nsteps
        S_top_half= S_top - step/2
        z_step=[np.sqrt(Ret2 +(S_top_half - i*step)**2 -
                        2*(Re + z_t)*(S_top_half - 
                          i*step)*np.cos(np.pi-sol_zen))-Re for i in range(nsteps) ]
        z_step = np.array(z_step)
        #check the Earth's shadow
        if (z_step<0).any():
            z_step = np.zeros(nsteps)

    return(z_step, step)

def photolysis(z, sol_zen, density_lst, xs_lst, name_lst):
    #z in km
    #density in cm-3, xarray
    #xs (cross section) in cm2
    #global variables: z, wavelength, irrad
    j_lst = [[] for i in range(len(density_lst))]
    for z_t in z:
        z_paths, path_step= path_z(z[-1], z_t, sol_zen, 100)
        if (z_paths == 0).all():
            j_z_lst = (irrad * xs_lst * 0)
        else:
            attenuation_coeff = [xs * density.pipe(lambda x: np.log(x)).interp(z=z_paths, kwargs=dict(fill_value='extrapolate')).pipe(lambda x: np.exp(x)).sum().values #sum over z_paths
                                for xs, density in zip(xs_lst, density_lst)]
            tau = np.sum(attenuation_coeff, axis=0) * path_step *1e5 #km-->cm , sum over species   
            j_z_lst = (irrad * xs_lst * np.exp(-tau)) #
        [j.append(j_z) for j,j_z in zip(j_lst, j_z_lst)]
    
    data_j = {'j{}'.format(name): (('z', 'wavelength'), j) for name,j in zip(name_lst, j_lst)}
    return xr.Dataset({'z': (('z',), z), 'wavelength': (('wavelength',), wavelength), **data_j})

# set photolysis rates
def cal_Js(t_s, ds, elv_max=90):
    [s.set_density(ds[s.name].sel(t=t_s*s2h, method='nearest').interp(z=z)).set_unit('cm-3')
        for s in (o3, h2o2)]
    
    sol_zen = func_sza(t=t_s, elv_max=elv_max)
    ds_j = photolysis(z=z, sol_zen=sol_zen, 
                    density_lst = (xr.DataArray(o2.density, dims=('z',), coords=(z,)),
                                    xr.DataArray(o3.density, dims=('z',), coords=(z,)),
                                    xr.DataArray(h2o2.density, dims=('z',), coords=(z,))),
                    xs_lst=(so2, so3, sh2o2), 
                    name_lst=('o2', 'o3', 'h2o2'))
    J1 = ds_j.jo2.sel(wavelength=slice(177.5, 256)).sum('wavelength').rename('J1') #Herzberg + SRB
    J2 = ds_j.jo2.sel(wavelength=slice(177.5)).sum('wavelength').rename('J2') #Schuman-Runge C + lya
    J3 = ds_j.jo3.sel(wavelength=slice(200, 730)).sum('wavelength').rename('J3') #Hartley + Huggins
    J4 = ds_j.jo3.sel(wavelength=slice(167.5, 320)).sum('wavelength').rename('J4') #Hartley
    J7 = ds_j.jh2o2.sel(wavelength=slice(122,350)).sum('wavelength').rename('J7')
    return xr.merge([J1, J2, J3, J4, J7])


def test_Js(t_s, ds, elv_max=90):
    [s.set_density(ds[s.name].sel(t=t_s*s2h, method='nearest').interp(z=z)).set_unit('cm-3')
        for s in (o3, h2o2)]
    
    sol_zen = func_sza(t=t_s, elv_max=elv_max)
    ds_j = photolysis(z=z, sol_zen=sol_zen, 
                    density_lst = (xr.DataArray(o2.density, dims=('z',), coords=(z,)),
                                    xr.DataArray(o3.density, dims=('z',), coords=(z,)),
                                    xr.DataArray(h2o2.density, dims=('z',), coords=(z,))
                                    ),
                    xs_lst = (so2, so3, sh2o2), 
                    name_lst = ('o2', 'o3', 'h2o2'))
    J0 = ds_j.jo2.sel(wavelength=121.6, method='nearest').rename('Lya') #Ly-alpha line
    J1 = ds_j.jo2.sel(wavelength=slice(130, 175)).sum('wavelength').rename('SRC') #SRC
    J2 = ds_j.jo2.sel(wavelength=slice(175, 200)).sum('wavelength').rename('SRB') #SRB
    J3 = ds_j.jo2.sel(wavelength=slice(200, 242)).sum('wavelength').rename('Herzberg') #Herzberg
    
    J4 = ds_j.jo3.sel(wavelength=slice(200, 310)).sum('wavelength').rename('Hartley') #Hartley
    J5 = ds_j.jo3.sel(wavelength=slice(310, 400)).sum('wavelength').rename('Huggins') #Huggins
    J6 = ds_j.jo3.sel(wavelength=slice(400, 850)).sum('wavelength').rename('Chappuis') #Chappuis
    
    J7 = ds_j.jh2o2.sel(wavelength=slice(122,350)).sum('wavelength').rename('H2O2')
    return xr.merge([J0, J1, J2, J3, J4, J5, J6, J7])

#%%
%%time
with xr.open_dataset('./ode_result/non_symbolic/ode_result_{}_{}_{}.nc'.format(55, 127, 50)) as ds:
    t_inspect = ds.t.values * h2s 
    js_inspect = []
    for t_s in t_inspect:
        js_inspect.append(test_Js(t_s, ds))
    js_inspect = xr.concat(js_inspect, dim='t').assign_coords(t=t_inspect)

t_bins = np.linspace(0, 24*h2s, 24*60)
t_bin_labels = t_bins[:-1] + (t_bins[1] - t_bins[0])/2
js_mean = js_inspect.groupby_bins(js_inspect.t%(24*h2s), bins=t_bins, labels=t_bin_labels
                                ).mean('t').rename(dict(t_bins='t'))
# js_mean.to_netcdf('./Js_table_w3.nc')

#%%
js_mean.sel(t=12*3600, method='nearest'
    ).assign(dict(Total = js_mean.sel(t=12*3600, method='nearest'
    ).to_array('region'
    ).sel(region='Lya SRC SRB Herzberg'.split()
    ).sum('region'
    ))).to_array('region'
    ).sel(region='Lya SRC SRB Herzberg Total'.split()
    ).
    ).plot.line(y='z', hue='region', xscale='log', xlim=(1e-11, 1e-6))

# %% Play with the cross section file
with xr.open_dataset('/home/anqil/Documents/Python/Photochemistry/xs_anqi.nc') as ds:
    ds = ds.sel(wavelength=~ds.indexes['wavelength'].duplicated())

    ds.so2.loc[dict(wavelength=ds.wavelength[np.where(ds.so2==0)[0][0]])
        ] = ds.so2.loc[dict(wavelength=ds.wavelength[np.where(ds.so2==0)[0][0]-1])]/10
    ds.so2.where(ds.so2!=0
        ).pipe(lambda x: np.log(x)
        ).interpolate_na('wavelength'
        ).pipe(lambda x: np.exp(x)
        ).fillna(0
        ).plot(x='wavelength', yscale='log', xlim=(0,300))
    # plt.axvline(x=177.5, ls=':')
    # plt.axvline(x=256, ls=':')
    plt.fill_betweenx([1e-25, 1e-16], 177.5, label='O2 -> O + O(1D)', alpha=0.2)
    plt.fill_betweenx([1e-25, 1e-16], 177.5, 256, label='O2 -> 2O', alpha=0.2)
    plt.legend()
    plt.title('O2 cross section')
# %% Compare two J tables 
xr.open_dataset('./Js_table_w2.nc').pipe(lambda x: -(js_mean-x)/js_mean*100
    ).J1.plot.line(x='t', yscale='log')
# plt.gca().set_xticklabels(plt.gca().get_xticks()/3600)
plt.ylabel('% differences')
plt.title('J (O2->2O) changes from no SRB to interpolated SRB')

# %%
