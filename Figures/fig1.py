#Three-panel figure: CGT regression / OLR+Jet / Soil Moisture
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy import stats, signal
import pandas as pd
from eofs.xarray import Eof   

YEARS         = np.arange(1951, 2025)
DATA_DIR      = "1951_2024_Dailyavg_gz_u_v"
OLR_DIR       = "1951_2024_Dailyavg_OLR"
SM_FILE       = "1951_2024_Monthlyavg_SM.nc"
GZ_DIR        = "Data"
U_WIND_LEVELS = [-6, -4, 4, 6]

POS_CGT_MONTHS = [
    (1952,7),(1954,7),(1955,7),(1955,8),(1956,6),(1956,7),
    (1959,7),(1961,6),(1961,7),(1962,6),(1963,7),(1964,6),
    (1964,7),(1967,7),(1967,8),(1969,8),(1970,6),(1970,8),
    (1971,8),(1972,6),(1972,8),(1973,7),(1974,6),(1975,6),
    (1975,7),(1975,8),(1976,6),(1976,8),(1978,6),(1981,7),
    (1981,8),(1983,8),(1984,6),(1984,7),(1984,8),(1986,6),
    (1988,6),(1989,6),(1989,7),(1991,8),(1994,6),(1994,7),
    (1994,8),(1996,6),(1996,7),(1996,8),(1999,6),(1999,7),
    (2000,6),(2000,7),(2000,8),(2001,6),(2001,7),(2003,7),
    (2006,6),(2006,8),(2008,6),(2008,7),(2010,7),(2011,7),
    (2012,7),(2012,8),(2013,6),(2013,7),(2013,8),(2018,6),
    (2018,7),(2018,8),(2021,6),(2021,7),(2022,6),(2022,7),
    (2022,8),(2024,8)
]
NEG_CGT_MONTHS = [
    (1951,6),(1952,6),(1954,6),(1954,8),(1956,8),(1957,6),
    (1957,8),(1958,6),(1958,7),(1958,8),(1959,6),(1960,8),
    (1962,7),(1962,8),(1963,8),(1964,8),(1965,6),(1965,7),
    (1965,8),(1966,8),(1968,7),(1968,8),(1969,6),(1974,8),
    (1976,7),(1977,8),(1979,7),(1980,7),(1980,8),(1981,6),
    (1982,6),(1982,7),(1983,6),(1985,6),(1986,8),(1987,6),
    (1987,7),(1987,8),(1988,7),(1990,8),(1991,7),(1992,6),
    (1992,7),(1992,8),(1993,6),(1993,7),(1993,8),(1995,7),
    (1997,6),(1997,7),(1998,6),(1998,7),(2003,6),(2004,8),
    (2005,6),(2007,7),(2009,6),(2009,7),(2009,8),(2010,6),
    (2011,6),(2012,6),(2014,8),(2015,7),(2015,8),(2017,6),
    (2019,6),(2019,7),(2019,8),(2020,7),(2020,8),(2023,7),
    (2023,8),(2024,6)
]

def preprocess(ds, var, lat_slice=slice(90, -15)):
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    da = ds[var].where(np.abs(ds[var]) < 1e25)
    da = da.sel(latitude=lat_slice)
    return da.sortby("time")

def mon_mean(da):
    return da.resample(time="MS").mean()

def anomaly(da_mon):
    clim = da_mon.groupby("time.month").mean("time")
    return da_mon.groupby("time.month") - clim

def select_months(da, month_list):
    s = set(month_list)
    times = pd.to_datetime(da.time.values)
    mask = [(t.year, t.month) in s for t in times]
    return da.sel(time=mask)

def welch_sig(da_pos, da_neg, p=0.05):
    _, pv = stats.mannwhitneyu(
        da_pos.values.reshape(da_pos.shape[0], -1),
        da_neg.values.reshape(da_neg.shape[0], -1),
        axis=0, alternative="two-sided"
    )
    return pv.reshape(da_pos.shape[1:]) < p

def thin_stipple(sig_mask, lats, stride_deg=1):
    dlat = float(np.abs(np.diff(lats).mean()))
    stride = max(1, int(round(stride_deg / dlat)))
    out = np.zeros_like(sig_mask, dtype=bool)
    out[::stride, ::stride] = sig_mask[::stride, ::stride]
    return out

def add_map_features(ax, lat_min):
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_extent([-180, 180, lat_min, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.4, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator(range(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(range(0, 91, 30))
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    return gl

#1) CGT regression map
ds_gz = xr.open_dataset(f"{GZ_DIR}/MonthlyGeopotential1941to2024.nc")
gz = ds_gz['z'].squeeze('pressure_level', drop=True)
gz = gz.sel(valid_time=slice('1951-01-01', '2024-12-31')) / 9.80665

summer_gz = gz.sel(valid_time=gz.valid_time.dt.month.isin([6, 7, 8]))
summer_clim = summer_gz.groupby('valid_time.month').mean('valid_time')
summer_anom = summer_gz.groupby('valid_time.month') - summer_clim

def detrend_dim(data, dim='valid_time'):
    def detrend_1d(y):
        if np.all(np.isnan(y)): return y
        return signal.detrend(y, type='linear')
    return xr.apply_ufunc(
        detrend_1d, data,
        input_core_dims=[[dim]], output_core_dims=[[dim]],
        vectorize=True, dask='parallelized', output_dtypes=[float])

summer_anom = detrend_dim(summer_anom, dim='valid_time')
summer_anom = summer_anom.rename({'valid_time': 'time'})

nh = summer_anom.sel(latitude=slice(90, 0))
nh = nh.transpose('time', 'latitude', 'longitude').load()

coslat = np.cos(np.deg2rad(nh.latitude.values))
wgts_2d = xr.DataArray(
    np.sqrt(coslat)[:, np.newaxis] * np.ones((1, nh.longitude.size)),
    coords={'latitude': nh.latitude, 'longitude': nh.longitude},
    dims=('latitude', 'longitude'))

solver = Eof(nh, weights=wgts_2d)
pc2 = solver.pcs(npcs=2, pcscaling=1).isel(mode=1)
pc2_std = (pc2 - pc2.mean("time")) / pc2.std("time")

reg_map = xr.cov(nh, pc2_std, dim="time") / pc2_std.var("time")

#2) OLR & Jet
olr_files = [f"{OLR_DIR}/{y}_OLR.nc" for y in YEARS if os.path.exists(f"{OLR_DIR}/{y}_OLR.nc")]
ds_olr  = xr.open_mfdataset(olr_files, combine="by_coords", decode_times=True)

olr_raw = preprocess(ds_olr, "ttr") / 3600
olr     = -olr_raw
olr_mon  = mon_mean(olr)
olr_anom = anomaly(olr_mon)

u_files = [f"{DATA_DIR}/{y}/u_component_of_wind_0_daily-mean.nc"
           for y in YEARS if os.path.exists(f"{DATA_DIR}/{y}/u_component_of_wind_0_daily-mean.nc")]
ds_u   = xr.open_mfdataset(u_files, combine="by_coords", decode_times=True)
u      = preprocess(ds_u, "u").sel(pressure_level=200)
u_mon  = mon_mean(u)
u_anom = anomaly(u_mon)

olr_pos = select_months(olr_anom, POS_CGT_MONTHS)
olr_neg = select_months(olr_anom, NEG_CGT_MONTHS)
u_pos   = select_months(u_anom, POS_CGT_MONTHS)
u_neg   = select_months(u_anom, NEG_CGT_MONTHS)

olr_diff = olr_pos.mean("time") - olr_neg.mean("time")
u_diff   = u_pos.mean("time") - u_neg.mean("time")
olr_sig  = welch_sig(olr_pos, olr_neg)

#3) Soil Moisture
ds_sm   = xr.open_dataset(SM_FILE)
THICKNESS = {"swvl1": 0.07, "swvl2": 0.21, "swvl3": 0.72}
sm_tot = sum(preprocess(ds_sm, v) * dz for v, dz in THICKNESS.items()) * 1000
sm_mon  = mon_mean(sm_tot)
sm_anom = anomaly(sm_mon)

sm_pos = select_months(sm_anom, POS_CGT_MONTHS)
sm_neg = select_months(sm_anom, NEG_CGT_MONTHS)
sm_diff = sm_pos.mean("time") - sm_neg.mean("time")
sm_sig  = welch_sig(sm_pos, sm_neg)

def cgt_cmap():
    levels = np.array([-35, -25, -15, -10, -5, -2.5, 2.5, 5, 10, 15, 25, 35])
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend='both')
    return cmap, norm, levels

def olr_cmap(data):
    v = float(np.nanmax(np.abs(data)))
    cmap = LinearSegmentedColormap.from_list(
        "green_brown",
        ["#2d6a2d", "#74c476", "#c7e9c0", "white",
         "#d4b483", "#a0522d", "#5c2c0a"])
    norm = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)
    return cmap, norm

def sm_cmap(data):
    v = float(np.nanmax(np.abs(data)))
    return "BrBG", TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

#Plotting figure
proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(12, 16))
gs = gridspec.GridSpec(3, 1, hspace=0.30)

ax1 = fig.add_subplot(gs[0], projection=proj)
ax2 = fig.add_subplot(gs[1], projection=proj)
ax3 = fig.add_subplot(gs[2], projection=proj)

def add_close_colorbar(fig, ax, mappable, gap=0.05, height=0.015, **kwargs):
    fig.canvas.draw() 
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - gap, pos.width, height])
    return fig.colorbar(mappable, cax=cax, orientation="horizontal", **kwargs)

cmap1, norm1, levels1 = cgt_cmap()
data_c, lon_c = add_cyclic_point(reg_map.values, coord=reg_map.longitude)
m1 = ax1.contourf(
    lon_c, reg_map.latitude, data_c,
    levels=levels1, cmap=cmap1, norm=norm1,
    extend="both", transform=ccrs.PlateCarree())

add_map_features(ax1, lat_min=0)
ax1.set_title("(a) Z200 Regressed onto CGT (PC2)", loc="left", fontsize=16)
cb1 = add_close_colorbar(fig, ax1, m1, extend="both")
cb1.set_ticks(levels1)
cb1.set_label("Geopotential Height (m per std dev PC2)", fontsize=14)

cmap2, norm2 = olr_cmap(olr_diff)
m2 = ax2.pcolormesh(
    olr_diff.longitude, olr_diff.latitude, olr_diff,
    cmap=cmap2, norm=norm2, shading="auto", transform=ccrs.PlateCarree())

add_map_features(ax2, lat_min=-15)
lon2d_o, lat2d_o = np.meshgrid(olr_diff.longitude.values, olr_diff.latitude.values)
olr_stip = thin_stipple(olr_sig, olr_diff.latitude.values, stride_deg=2.5)
ax2.scatter(
    lon2d_o[olr_stip], lat2d_o[olr_stip],
    s=2.0, color="black", alpha=0.6,
    transform=ccrs.PlateCarree(), zorder=5)
for lvls, ls in [([l for l in U_WIND_LEVELS if l > 0], "solid"),
                 ([l for l in U_WIND_LEVELS if l < 0], "dashed")]:
    cs = ax2.contour(
        u_diff.longitude, u_diff.latitude, u_diff,
        levels=lvls, colors="black", linewidths=1.4,
        linestyles=ls, transform=ccrs.PlateCarree())
    ax2.clabel(cs, fmt="%d", fontsize=8)

ax2.set_title("(b) OLR & Jet Anomaly Difference", loc="left", fontsize=16)
cb2 = add_close_colorbar(fig, ax2, m2)
cb2.set_label("OLR Anomaly Difference (W m$^{-2}$)", fontsize=14)

cmap3, norm3 = sm_cmap(sm_diff)
m3 = ax3.pcolormesh(
    sm_diff.longitude, sm_diff.latitude, sm_diff,
    cmap=cmap3, norm=norm3, shading="auto", transform=ccrs.PlateCarree())
add_map_features(ax3, lat_min=0)
lon2d_s, lat2d_s = np.meshgrid(sm_diff.longitude.values, sm_diff.latitude.values)
sm_stip = thin_stipple(sm_sig, sm_diff.latitude.values, stride_deg=2)
ax3.scatter(
    lon2d_s[sm_stip], lat2d_s[sm_stip],
    s=1.5, color="black", alpha=0.6,
    transform=ccrs.PlateCarree(), zorder=5)
ax3.set_title("(c) Soil Moisture Anomaly Difference", loc="left", fontsize=16)
cb3 = add_close_colorbar(fig, ax3, m3)
cb3.set_label("0–100 cm Soil Moisture Anomaly Difference (mm)", fontsize=14)

plt.savefig("Fig1_CGT_ThreePanel.png", dpi=300, bbox_inches="tight")
plt.show()
