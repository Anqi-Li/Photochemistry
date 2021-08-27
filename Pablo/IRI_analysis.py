#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

#%% Load Pablo model results
month_nh = 12 #nh
month_sh = 6
def assign_month_lat(ds):
    return ds.assign_coords(month=ds.month, lat=ds.lat)

model = []
for file in ['./results_nc/sim2_{}_{}.nc'.format(month_nh,lat) for lat in range(0,90,20)]:
    with assign_month_lat(xr.open_dataset(file)) as ds:
        model.append(ds)
mds_nh = xr.concat(model, dim='lat')

model = []
for file in ['./results_nc/sim2_{}_{}.nc'.format(month_sh,lat) for lat in range(20,90,20)]:
    with assign_month_lat(xr.open_dataset(file)) as ds:
        model.append(ds.assign_coords(month=month_nh))
mds_sh = xr.concat(model, dim='lat')
mds_sh = mds_sh.assign_coords(lat=-mds_sh.lat)
model = xr.concat([mds_nh,mds_sh], dim='lat').sortby('lat')

last_day = int(model.t[-1]/(24*3600))-1
model_day = model.sel(t=slice((last_day)*24*3600, (last_day+1)*24*3600))
model_day = model_day.assign_coords(t=model_day.t/3600-last_day*24)

# %% calculate [OH]v
T = xr.DataArray( #from Pablo's code
    [253.7,247.2, 235.4, 220.6, 207.3, 198.0, 195.3, 195.4, 192.5, 185.2, 176.3, 177.6, 192.5],
    dims=('z',), 
    coords=(('z', model_day.z, dict(units='km')),))
k_O3_H = 1.4e-10*np.exp(-270/T) # Allen 1984
k_OH_M = 0.2e-14 #McDate & Llewelyn 1987

model_day = model_day.assign(
    OH_v=(k_O3_H * model_day.H * model_day.O3)/(k_OH_M * (model_day.O2*5+model_day.O)) #cm-3
    )
# %% 
lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
model_day.OH_v.assign_attrs(units='cm-3').sel(
    z=slice(65,105)
).reindex(
    lat=lat_seq
).plot.line(
    x='t', hue='z', col='lat', col_wrap=2, 
    yscale='log',
    sharey=True, figsize=(8,9),
    )

#%%
model_day = model_day.sel(z=slice(65,105))

# %%
# from scipy.optimize import curve_fit
# def gauss(x, a, x0, sigma):
#     '''
#     x: data
#     a: amplitude
#     x0: mean of data
#     sigma: std
#     '''
#     return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

# def gauss_fit_integral(z, y):
#     #z:m
#     #y:cm-3
#     a0, mean0, sigma0 = 2e7, 95e3, 5e3 #cm-3, m, m
#     popt, _ = curve_fit(gauss, z, y, 
#                     p0=[a0, mean0, sigma0], 
#                     # bounds=([0, 70e3, 0], [1e5, 100e3, 40e3]) #some reasonable ranges for the airglow characteristics
#                     )
#     a, _, sigma = popt
#     return a * sigma*1e2 * np.sqrt(2*np.pi), #cm-2

# zenith_gauss_int = []
# for t in model_day.t:
#     # print(ti)
#     zenith_gauss_int.append(gauss_fit_integral(model_day.z*1e3, model_day.OH_v.sel(t=t)))
# zenith_gauss_int = xr.DataArray(zenith_gauss_int, dims=('t',), coords=(model_day.t,)) #cm-2

# %%
zenith_int = model_day.OH_v.integrate('z')*1e5 #cm-2
dz = 5e5 #cm
zenith_sum = model_day.OH_v.sum('z')*dz #cm-2

# xr.concat(
#     [zenith_int, zenith_sum], dim='var'
#     ).plot.line(
#         hue='var', col='lat', col_wrap=3, sharey=False)
zenith_int.plot.line(hue='lat')
# zenith_sum.plot.line(col='lat', col_wrap=3, sharey=False)
# zenith_gauss_int.plot()

# %% Open daily average Gauss parameters
am_pm = 'PM'
# filename = 'gauss_{}_D_96_{}.nc'.format(am_pm, '{}')
path = '~/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/zenith/'
filename = '{}_daily_zonal_mean_{}.nc'.format(am_pm, '{}')
years = list(range(2001,2016))
with xr.open_mfdataset([path+filename.format(y) for y in years]) as mds:
    mds = mds.rename({'latitude_bin': 'latitude_bins'})
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1]).load()
    mds = mds.assign_coords(latitude_bins=mds.latitude_bins.astype(int),
                            z = mds.z*1e-3, #m -> km
                            )
    # low sample size data
    mds = mds.where(mds.count_sza>50)

filename = '/home/anqil/Documents/osiris_database/composite_lya_index.nc'
with xr.open_dataset(filename) as ds_y107:
    ds_y107 = ds_y107.interp_like(mds)
ds_lya = ds_y107.copy()
ds_lya['irradiance'] *= 1e3
ds_lya['irradiance'] = ds_lya['irradiance'].assign_attrs(dict(units='$10^{-3} Wm^{-2}$'))

#%% Odin LST plot
lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
mds.mean_apparent_solar_time.rename('LST [h]').reindex(
    latitude_bins=lat_seq
    ).plot(
        col='latitude_bins', col_wrap=2, 
        sharey=False, figsize=(8,9),
        )

#%% Modelled Odin [OH]v
zenith = zenith_int.rename('zenith [OH]v [cm-2]')
oh_model_odin = zenith.interp(
    t=mds.mean_apparent_solar_time,#.pipe(lambda x: x-24),
    lat=mds.latitude_bins, #mean_latitude,
    )
# lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
# oh_model_odin.reindex(
#     latitude_bins=lat_seq
#     ).plot(
#         col='latitude_bins', col_wrap=2, 
#         sharey=False, figsize=(8,9),
#         )
# plt.ylabel('Modelled [OH]v [cm-3]')

##%% Lya correlation plot 
var = 'mean_zenith_intensity'
var_label = '[OH]v' #'ZER'
var_units = 'cm^{-2}'#'$cm^{-2} s^{-1}$'
data1 = oh_model_odin.rename('Modelled').resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)
error1 = oh_model_odin.rename('Modelled').resample(time='1Y', loffset='-6M').std('time', keep_attrs=True)

da_lya = ds_lya.irradiance.resample(time='1Y', loffset='-6M').mean('time', keep_attrs=True)
da = data1.assign_coords(lya=da_lya).swap_dims(dict(time='lya'))
pf = da.polyfit(dim='lya', deg=1, cov=True)
poly_coeff = pf.polyfit_coefficients
poly_deg1_error = pf.polyfit_covariance.sel(cov_i=0, cov_j=0).pipe(np.sqrt)
poly_deg1 = poly_coeff.sel(degree=1)

lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
fc = da.reindex(
    latitude_bins=lat_seq).plot.line(
        figsize=(8,9), sharex=True,
        y='lya', col='latitude_bins', col_wrap=2, 
        ls='-', marker='o', markersize=5,
    )

for j in range(fc.axes.shape[1]):
    for i in range(fc.axes.shape[0]):
        if (i+1,j+1) == fc.axes.shape:
            pass 
        else:
            fc.axes[i,j].errorbar(
                da.sel(**fc.name_dicts[i,j]), da.lya, 
                xerr=error1.sel(**fc.name_dicts[i,j]),
                ecolor='C0', alpha=0.5, ls='', capsize=3,
            )
            
            xr.polyval(
                coord=da.lya, coeffs=poly_coeff.sel(**fc.name_dicts[i,j])
                ).plot(
                    y='lya', ax=fc.axes[i,j], 
                    color='C3', ls='-', lw=1,
                    label='s = {}\n$\pm${} ({}%)'.format(
                        poly_deg1.sel(**fc.name_dicts[i,j]).values.astype(int),
                        poly_deg1_error.sel(**fc.name_dicts[i,j]).values.astype(int),
                        abs(poly_deg1_error/poly_deg1*100).sel(**fc.name_dicts[i,j]).values.astype(int),
                        # var_units, da_lya.units,
                    ),
                )
            fc.axes[i,j].legend(handlelength=0.5, frameon=False)
            if j==1:
                fc.axes[i,j].set_ylabel('')

fc.set_axis_labels('{} [{}]'.format(var_label, var_units), 'Ly-a [{}]'.format(da_lya.units))
fc.axes[-2,1].tick_params(labelbottom=True, labelrotation=0)
fc.axes[-2,1].set_xlabel(fc.axes[-1,0].get_xlabel())
for j, sn in enumerate('S N'.split()):
    for i, lat in enumerate(np.linspace(80,0,num=len(fc.axes)).astype(int)):
        fc.axes[i,j].set_title('{} - {} $^\circ$ {}'.format(lat-10, lat+10, sn))
fc.axes[-1,0].ticklabel_format(axis='x', style='sci', scilimits=(11,11))
