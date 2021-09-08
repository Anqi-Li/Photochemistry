#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
path_data = './results_nc/OOM/'
#%% Load Pablo model results
# NH winter (December) and SH summer (June)
# month_nh = 12 #nh
# month_sh = 6
# def assign_month_lat(ds):
#     return ds.assign_coords(month=ds.month, lat=ds.lat)

# model = []
# for file in [path_data+'sim2_{}_{}.nc'.format(month_nh,lat) for lat in range(0,90,20)]:
#     with assign_month_lat(xr.open_dataset(file)) as ds:
#         model.append(ds)
# mds_nh = xr.concat(model, dim='lat')
# mds_nh = mds_nh.assign_coords(t=mds_nh.t/3600)
# mds_nh.to_array('species').sel(
#     lat=0,
#     species='O',
#     z=slice(90,110),
# ).plot(
#     x='t',
#     hue='z',
#     # col='species', col_wrap=3,
#     yscale='log',
#     # sharey=False,
# );
#%% compile all months and latitudes (NH), and flip data for SH, combine all
# model_nh = []
# for lat in range(0,90,20):
#     concat_month = []
#     for month in range(1,13):
#         with xr.open_dataset(path_data+'sim2_{}_{}.nc'.format(month,lat)) as ds:
#             concat_month.append(ds)
#     model_nh.append(xr.concat(concat_month, dim='month'))
# model_nh = xr.concat(model_nh, dim='lat')
# model_sh = model_nh.roll(
#     month=6, roll_coords=False
#     ).assign_coords(
#         lat=-model_nh.lat
#         ).sel(lat=slice(-20,-90))
# model = xr.concat([model_nh, model_sh], dim='lat').sortby('lat')

# last_day = int(model.t[-1]/(24*3600))-1
# model_day = model.sel(t=slice((last_day)*24*3600, (last_day+1)*24*3600))
# model_day = model_day.assign_coords(t=model_day.t/3600-last_day*24)
# model_day['t'] = model_day.t.assign_attrs(units='hour')

#%% save model_day or Js_day
# model_day.to_netcdf(path_data+'model_day.nc')
# model_day.to_netcdf(path_data+'Js_day.nc')

#%% construct a combined dataset with lya_f dimenssion
# model_day_J2 = model_day.copy()
# model_day = xr.concat([model_day, model_day_J2], dim='lya_f').assign_coords(lya_f=[1,2])

#%%
with xr.open_dataset(path_data+'model_day.nc') as model_day:
    # print(model_day)
    model_day.H.sel(
        # # lambda x: x.sel(lya_f=2)/x.sel(lya_f=1)
        # lambda x: x * ds_bg.o2 * (ds_bg.o2+ds_bg.n2) * 1.07e-34*np.exp(510/ds_bg.T)
        # ).rename(
        #     # 'ratio O(x2)/O(x1)'
        #     'k[O][O2][M]'
        # ).sel(
            # lya_f=1,
            lat=slice(0,90),
            t=np.arange(0,24,1)
        ).plot.line(
            y='z',
            hue='t',
            col='lat',
            row='month',
            xscale='linear',
            # xlim=(1,1.5),
            add_legend=True, 
        )
    
    # model_day.O.sel(
    #     month=1,
    #     lat=0,
    #     t=np.arange(0,24,1)
    # ).plot.line(
    #     y='z', 
    #     hue='t', 
    #     col='lya_f',
    #     add_legend=False, xscale='linear')
    
#%% MSIS temperature
file = '/home/anqil/Documents/Python/external_data/msis_cmam_climatology_z200_lat8576.nc'
with xr.open_dataset(file) as ds_bg:
    # print(ds_bg)
    ds_bg = ds_bg.interp(
        month=model_day.month,
        lat=model_day.lat,
        lst=model_day.t,
        kwargs=dict(fill_value='extrapolate')
        )

# ds_bg.T.pipe(
#     # lambda x: 1.07e-34*np.exp(510/x) # Federik (1979)
#     lambda x: 6e-34*(300/x)**2.4 #O+O2+M Marsh(2006) Page 11 
# ).sel(
#     lat=0, month=6
# ).plot.line(y='z')



#%% Check J2*[O2] production of [O]
with xr.open_dataset(path_data+'Js_day.nc') as Js_day:
    Js_day.new_J2_day.pipe(
        lambda x: x*ds_bg.o2
    ).rename(
        'J2 * [O2]'
    ).sel(
        z=slice(70,105), 
        lat=slice(0,80), 
        t=np.arange(0,24,1),
        # lya_f=1,
    ).plot.line(
        hue='t', 
        y='z', 
        row='month', 
        col='lat',
        add_legend=True, yscale='linear'
    );
#%%

# %% calculate [OH]v
# T = xr.DataArray( #from Pablo's code
#     [253.7,247.2, 235.4, 220.6, 207.3, 198.0, 195.3, 195.4, 192.5, 185.2, 176.3, 177.6, 192.5],
#     dims=('z',), 
#     coords=(('z', np.arange(50, 115, 5), dict(units='km')),))
# T = T.interp(z=model_day.z)
T = ds_bg.T
k_O3_H = 1.4e-10*np.exp(-270/T) # Allen 1984
k_OH_M = 0.2e-14 #McDate & Llewelyn 1987, need to understand their equations!

model_day = model_day.assign(
    OH_v=(k_O3_H * model_day.H * model_day.O3)/(k_OH_M * (model_day.O2*5+model_day.O)) #cm-3
    # OH_v=(k_O3_H * model_day.H * model_day.O3)/(k_OH_M * (model_day.O2*5)) #cm-3
    )
model_day['OH_v'] = model_day.OH_v.assign_attrs(units='cm-3')

#%% plot z profile of OH_v at all months and latitudes
model_day.O3.sel(
    z=slice(70,105), 
    lat=slice(0,80), 
    t=np.arange(0,24,1),
    # lya_f=1,
).plot.line(
    hue='t', 
    y='z', 
    row='month', 
    col='lat',
    add_legend=True,
    xscale='linear'
);
#%%try the p and [O] equation
def oh_profile(O, T, p):
    return p * O/ T**3.4
model_day.O.pipe(
    oh_profile, ds_bg.T, ds_bg.plev
).sel(
    z=slice(70,105), 
    lat=slice(0,80), 
    t=np.arange(0,24,1),
    lya_f=1,
).plot.line(
    hue='t', 
    y='z', 
    row='month', 
    col='lat',
    add_legend=True, yscale='linear'
);

#%%
model_day = model_day.assign(
    L_OH_v=(k_OH_M * (model_day.O2*5+model_day.O)), #cm-3
    P_OH_v=(k_O3_H * model_day.H * model_day.O3), #cm-3
    )
model_day['H HO2 OH'.split()].to_array('var').sum('var').sel(
# model_day.H.sel(
    t=np.arange(0,24,2),#18,
    month=6,
    lat=0,
    # lya_f=1,
).plot.line(
    y='z',
    # col='var',
    hue='t',
    xscale='log',
    # sharex=False,
);


#%% exponential extrapolation
# OH_v_extra = model_day.OH_v.pipe(np.log).interp(
#         z=np.arange(70,120,1), kwargs=dict(fill_value='extrapolate') 
#         ).pipe(np.exp)

# %% plot diurnal cycle of OHv at all altitudes and latitudes in a spesific month
month = 7
lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
model_day.OH_v.sel(
    z=slice(70,105), month=month
    ).reindex(
        lat=lat_seq
    ).plot.line(
        x='t', hue='z', col='lat', col_wrap=2, 
        yscale='log',
        sharey=True, figsize=(8,9),
        );

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
with xr.open_dataset(filename) as ds_lya:
    ds_lya = ds_lya.interp_like(mds)
ds_lya['irradiance'] *= 1e3
ds_lya['irradiance'] = ds_lya['irradiance'].assign_attrs(dict(units='$10^{-3} Wm^{-2}$'))

#%% Odin LST plot
# lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
# mds.mean_apparent_solar_time.rename('LST [h]').reindex(
#     latitude_bins=lat_seq
#     ).plot(
#         col='latitude_bins', col_wrap=2, 
#         sharey=False, figsize=(8,9),
#         )

#%%
with xr.open_dataset('./results_nc/J2_factor2/model_day.nc') as model_day:
    print(model_day)

# %%
# zenith_int = model_day.OH_v.pipe(np.log).interp(
#     z=np.arange(70,120,1), kwargs=dict(fill_value='extrapolate') 
#     ).pipe(np.exp).integrate('z')*1e5 #cm-2
zenith_int = model_day.sel(z=slice(70,105)).OH_v.integrate('z')*1e5 #cm-2
# zenith_int.plot.line(hue='lat', col='month', col_wrap=3);

#%% Modelled Odin [OH]v
zenith = zenith_int.rename('zenith [OH]v [cm-2]')
oh_model_odin = zenith.interp(
    t=18,#mds.mean_apparent_solar_time,#.pipe(lambda x: x-24),
    lat=mds.latitude_bins, #mean_latitude,
    month=3,#mds.time.dt.month,
    lya_f=ds_lya.irradiance.pipe(lambda x: x/x.min()),
    )
lat_seq = [-80, 80, -60, 60, -40, 40, -20, 20, 0]
(oh_model_odin/oh_model_odin.min('time')).rename(
    'zenith [OH]v ratio'
    ).reindex(
    latitude_bins=lat_seq
    ).plot(
        col='latitude_bins', col_wrap=2, 
        sharey=True, figsize=(8,9),
        )

#%% Lya correlation plot 
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

#%%











#%%





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
