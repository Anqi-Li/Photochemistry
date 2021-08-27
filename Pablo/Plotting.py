#%%
import xarray as xr
import numpy as np

#%% plot all Js
lat = 20
month = 9
with xr.open_dataset('./results_nc/Js_{}_{}.nc'.format(month, lat)) as Js:
    # print(Js)
    Js = Js.assign_coords(t=Js.t/3600)
    Js.drop_vars(['month', 'lat']).to_array('var').plot(
        x='t', hue='z', col='var', 
        col_wrap=3, sharey=False, yscale='linear',
    );
## %% plot all simulation days
with xr.open_dataset('./results_nc/sim2_{}_{}.nc'.format(month, lat)) as ds:
    # print(ds)
    ds.drop_vars(['month', 'lat']).to_array('species').plot(
        x='t', hue='z', col='species', 
        col_wrap=3, sharey=False, yscale='linear',
    );
    
#%% plot the last simulation day
lat = 40
month = 12
with xr.open_dataset('./results_nc/sim2_{}_{}.nc'.format(month, lat)) as ds:
    last_day = int(ds.t[-1]/(24*3600))-1
    ds_day = ds.sel(t=slice((last_day)*24*3600, (last_day+1)*24*3600))
    ds_day = ds_day.assign_coords(t=ds_day.t/3600-last_day*24)

    ds_day.drop_vars(['month', 'lat']).to_array('species').plot(
        x='t', hue='z', col='species', 
        col_wrap=3, sharey=False, yscale='log',
        );
# %%
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt

dt = 60*15 #s (15min interval)
t_day = np.arange(0, 24*60*60+dt, dt) #s 
# month = 12
lat = 0
# for lat in range(0,90,20):
for month in range(3,13,3):
    time = Time(pd.date_range(
        "2001-{}-15".format(str(month).zfill(2)), 
        freq="{}min".format(int(dt/60)), 
        periods=len(t_day)))

    loc = coord.EarthLocation(lon=0 * u.deg,
                            lat=lat * u.deg)
    altaz = coord.AltAz(location=loc, obstime=time)
    sun = coord.get_sun(time)
    Xi = sun.transform_to(altaz).zen.deg
    xr.DataArray(Xi, coords=(t_day/3600,), dims=('t',)).plot(
        x='t', label=month)
plt.legend()
plt.axhline(y=90, ls=':', c='k')
plt.axhline(y=96, ls=':', c='k')
plt.show()
# %%
