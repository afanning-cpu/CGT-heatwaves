#Evolution composite for wave activity flux, RWP envelope, and geopotential height for heat wave events

import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import psutil
from scipy.signal import hilbert
import functools

RELATIVE_DAYS = [-10, -5, 0, 5]
YEARS = np.arange(1951, 2025)

DATA_DIR_JJA = "1951_2024_Dailyavg_gz_u_v"
DATA_DIR_MS  = "1951_2024_MS_Dailyavg_gz_u_v_"
EVENTS_FILE  = "all_us_heatwaves_1951_2024.nc"
MS_RENAME_CUTOFF_YEAR = 1987

K_MIN, K_MAX = 4, 15
PLUMB_LEVEL = 200
WAF_THRESH = 0.25
QUIVER_SKIP = 20
EARTH_RADIUS = 6.371e6
OMEGA = 7.292e-5
SECONDS_PER_HOUR = 3600.0
U_MIN = 3.0

TIME_CHUNKS = {"time": 365, "pressure_level": 1, "latitude": -1, "longitude": -1}

POS_CGT_MONTHS = [
    (1952, 7), (1954, 7), (1955, 7), (1955, 8), (1956, 6), (1956, 7),
    (1959, 7), (1961, 6), (1961, 7), (1962, 6), (1963, 7), (1964, 6),
    (1964, 7), (1967, 7), (1967, 8), (1969, 8), (1970, 6), (1970, 8),
    (1971, 8), (1972, 6), (1972, 8), (1973, 7), (1974, 6), (1975, 6),
    (1975, 7), (1975, 8), (1976, 6), (1976, 8), (1978, 6), (1981, 7),
    (1981, 8), (1983, 8), (1984, 6), (1984, 7), (1984, 8), (1986, 6),
    (1988, 6), (1989, 6), (1989, 7), (1991, 8), (1994, 6), (1994, 7),
    (1994, 8), (1996, 6), (1996, 7), (1996, 8), (1999, 6), (1999, 7),
    (2000, 6), (2000, 7), (2000, 8), (2001, 6), (2001, 7), (2003, 7),
    (2006, 6), (2006, 8), (2008, 6), (2008, 7), (2010, 7), (2011, 7),
    (2012, 7), (2012, 8), (2013, 6), (2013, 7), (2013, 8), (2018, 6),
    (2018, 7), (2018, 8), (2021, 6), (2021, 7), (2022, 6), (2022, 7),
    (2022, 8), (2024, 8)]

NEG_CGT_MONTHS = [
    (1951, 6), (1952, 6), (1954, 6), (1954, 8), (1956, 8), (1957, 6),
    (1957, 8), (1958, 6), (1958, 7), (1958, 8), (1959, 6), (1960, 8),
    (1962, 7), (1962, 8), (1963, 8), (1964, 8), (1965, 6), (1965, 7),
    (1965, 8), (1966, 8), (1968, 7), (1968, 8), (1969, 6), (1974, 8),
    (1976, 7), (1977, 8), (1979, 7), (1980, 7), (1980, 8), (1981, 6),
    (1982, 6), (1982, 7), (1983, 6), (1985, 6), (1986, 8), (1987, 6),
    (1987, 7), (1987, 8), (1988, 7), (1990, 8), (1991, 7), (1992, 6),
    (1992, 7), (1992, 8), (1993, 6), (1993, 7), (1993, 8), (1995, 7),
    (1997, 6), (1997, 7), (1998, 6), (1998, 7), (2003, 6), (2004, 8),
    (2005, 6), (2007, 7), (2009, 6), (2009, 7), (2009, 8), (2010, 6),
    (2011, 6), (2012, 6), (2014, 8), (2015, 7), (2015, 8), (2017, 6),
    (2019, 6), (2019, 7), (2019, 8), (2020, 7), (2020, 8), (2023, 7),
    (2023, 8), (2024, 6)]


def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def ms_year_dir(year):
    "Folder holding May (and, pre-1987, September) data for a given year."
    if year < MS_RENAME_CUTOFF_YEAR:
        return f"{DATA_DIR_MS}/{year}"
    return f"{DATA_DIR_MS}/era5_daily_pl_may_{year}"

def sep_year_dir(year):
    "Folder holding September data for a given year (post-1987 only"
    return f"{DATA_DIR_MS}/era5_daily_pl_sep_{year}"

def collect_files(filename):
    files = []
    for year in YEARS:
        jja_path = f"{DATA_DIR_JJA}/{year}/{filename}"
        if os.path.exists(jja_path):
            files.append(jja_path)
        ms_path = f"{ms_year_dir(int(year))}/{filename}"
        if os.path.exists(ms_path):
            files.append(ms_path)
        if year >= MS_RENAME_CUTOFF_YEAR:
            sep_path = f"{sep_year_dir(int(year))}/{filename}"
            if os.path.exists(sep_path):
                files.append(sep_path)
    return files

def preprocess_data(ds, var, lat_slice=slice(90, 0)):
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    da = ds[var]
    da = da.where(np.abs(da) < 1e25)
    da = da.sel(latitude=lat_slice)
    return da.sortby("time")

def calculate_climatology(da):
    return da.groupby(da.time.dt.dayofyear).mean("time")

def zonal_wavenumber_filter(da, kmin, kmax):
    nlon = da.sizes["longitude"]
    ks = np.fft.rfftfreq(nlon) * nlon
    mask = ((ks >= kmin) & (ks <= kmax)).astype(float)

    def filt(x):
        X = np.fft.rfft(x, axis=-1)
        Xf = X * mask
        return np.fft.irfft(Xf, n=nlon, axis=-1)

    return xr.apply_ufunc(
        filt, da,
        input_core_dims=[["longitude"]],
        output_core_dims=[["longitude"]],
        vectorize=True, dask="parallelized",
        output_dtypes=[float])

def apply_rolling_mean(da, window=11):
    return da.rolling(time=window, center=True, min_periods=1).mean()

def calculate_rossby_wave_envelope(da):
    def hilbert_env(x):
        return np.abs(hilbert(x, axis=-1))
    return xr.apply_ufunc(
        hilbert_env, da,
        input_core_dims=[["longitude"]],
        output_core_dims=[["longitude"]],
        vectorize=True, dask="parallelized",
        output_dtypes=[float])

def filter_events_by_month(events_ds, month_list):
    event_months = [(pd.Timestamp(d).year, pd.Timestamp(d).month)
                    for d in events_ds.start_date.values]
    mask = [em in month_list for em in event_months]
    return events_ds.sel(event=mask)

def get_available_days(*time_arrays):
    day_sets = [set(pd.to_datetime(ta.values).normalize()) for ta in time_arrays]
    available = day_sets[0]
    for s in day_sets[1:]:
        available &= s
    return available

def get_valid_events(events_ds, month_list, available_days, label):
    events = filter_events_by_month(events_ds, month_list)
    valid_idx = []
    invalid_info = []
    for i in range(len(events.event)):
        start = pd.Timestamp(events.start_date.isel(event=i).values)
        missing_days = [d for d in RELATIVE_DAYS
                         if (start + pd.Timedelta(days=d)).normalize() not in available_days]
        if not missing_days:
            valid_idx.append(i)
        else:
            invalid_info.append((start, missing_days))
    valid_events = events.isel(event=valid_idx)
    print(f"{label}: {len(valid_events.event)} of {len(events.event)} candidate "
          f"events have complete data across all {len(RELATIVE_DAYS)} composite days")

    if invalid_info:
        print(f"{label}: {len(invalid_info)} events excluded due to missing data:")
        for start, missing_days in invalid_info:
            missing_dates = [(start + pd.Timedelta(days=d)).normalize().date() for d in missing_days]
            print(f"  start={start.date()}  missing offsets={missing_days}  missing dates={missing_dates}")

    return valid_events

def get_needed_dates(events_ds):
    dates = set()
    for i in range(len(events_ds.event)):
        start = events_ds.start_date.isel(event=i).values
        for d in RELATIVE_DAYS:
            dates.add(start + np.timedelta64(d, "D"))
    return np.array(sorted(dates))

def prepare_streamfunction_anom(z_anom_level):
    lat_rad = np.deg2rad(z_anom_level.latitude)
    f = 2.0 * OMEGA * np.sin(lat_rad)
    f_da = xr.DataArray(f, coords=[z_anom_level.latitude], dims=["latitude"])
    f_da = f_da.where(np.abs(f_da) > 2 * OMEGA * np.sin(np.deg2rad(5)))
    return (z_anom_level / 9.81) / f_da

def compute_tn_flux(psi_anom, U_bg, V_bg, pressure_hPa):
    p = pressure_hPa / 1000.0
    a = EARTH_RADIUS
    lat = np.deg2rad(psi_anom.latitude.values)
    cos_phi = np.cos(lat)
    cos_phi_da = xr.DataArray(cos_phi, coords=[psi_anom.latitude], dims=["latitude"])
    spd = np.sqrt(U_bg**2 + V_bg**2)
    spd = spd.where(spd >= U_MIN)
    prefactor = (p * cos_phi_da) / (2.0 * spd * a**2)
    deg2rad = np.rad2deg(1.0)
    psi_lam = (psi_anom.differentiate("longitude", edge_order=2) * deg2rad
               ).compute().astype(np.float32)
    psi_phi = (psi_anom.differentiate("latitude",  edge_order=2) * deg2rad
               ).compute().astype(np.float32)

    psi_lamlam = (psi_lam.differentiate("longitude", edge_order=2) * deg2rad
                  ).compute().astype(np.float32)
    psi_lamphi = (psi_lam.differentiate("latitude",  edge_order=2) * deg2rad
                  ).compute().astype(np.float32)
    psi_phiphi = (psi_phi.differentiate("latitude",  edge_order=2) * deg2rad
                  ).compute().astype(np.float32)

    cos2_da = (cos_phi_da**2).where(cos_phi_da**2 > 1e-4)
    A = psi_lam**2  - psi_anom * psi_lamlam
    B = psi_lam * psi_phi - psi_anom * psi_lamphi
    C = psi_phi**2  - psi_anom * psi_phiphi
    del psi_lamlam, psi_lamphi, psi_phiphi
    Wx = (prefactor * (U_bg * A / cos2_da + V_bg * B / cos_phi_da)).compute().astype(np.float32)
    Wy = (prefactor * (U_bg * B / cos_phi_da + V_bg * C           )).compute().astype(np.float32)
    del A, B, C, psi_lam, psi_phi
    return Wx, Wy

def incremental_composite(events_ds, wx_sm, wy_sm):
    wx_by_day = {d: [] for d in RELATIVE_DAYS}
    wy_by_day = {d: [] for d in RELATIVE_DAYS}

    for i in range(len(events_ds.event)):
        start = events_ds.start_date.isel(event=i).values
        for d in RELATIVE_DAYS:
            date = start + np.timedelta64(d, "D")
            try:
                wx_val = wx_sm.sel(time=date, method="nearest", tolerance=np.timedelta64(1, "D"))
                wy_val = wy_sm.sel(time=date, method="nearest", tolerance=np.timedelta64(1, "D"))
                wx_by_day[d].append(wx_val.values)
                wy_by_day[d].append(wy_val.values)
            except KeyError:
                pass

    coords = {"relative_day": RELATIVE_DAYS, "latitude": wx_sm.latitude, "longitude": wx_sm.longitude}
    dims = ["relative_day", "latitude", "longitude"]

    wx_arr = np.stack([np.nanmean(wx_by_day[d], axis=0) if wx_by_day[d]
                       else np.full(wx_sm.shape[1:], np.nan) for d in RELATIVE_DAYS])
    wy_arr = np.stack([np.nanmean(wy_by_day[d], axis=0) if wy_by_day[d]
                       else np.full(wy_sm.shape[1:], np.nan) for d in RELATIVE_DAYS])

    wx_out = xr.DataArray(wx_arr, dims=dims, coords=coords)
    wy_out = xr.DataArray(wy_arr, dims=dims, coords=coords)
    return wx_out, wy_out

def create_composite(events_ds, field):
    comps = []
    for i in range(len(events_ds.event)):
        start_date = events_ds.start_date.isel(event=i).values
        event_dates = [start_date + np.timedelta64(d, "D") for d in RELATIVE_DAYS]
        try:
            comp = field.sel(time=event_dates, method="nearest",
                              tolerance=np.timedelta64(1, "D"))
            if len(comp.time) == len(RELATIVE_DAYS):
                comps.append(comp)
        except Exception:
            continue
    comps_rd = [c.assign_coords({"relative_day": ("time", RELATIVE_DAYS)})
                for c in comps]
    stack = xr.concat(comps_rd, dim="event").stack(stacked=("event", "time"))
    return stack.groupby("relative_day").mean("stacked")

def plot_composite(wx_pos, wy_pos, z200_pos, env_pos,
                    wx_neg, wy_neg, z200_neg, env_neg,
                    filename, n_pos, n_neg):
    proj = ccrs.PlateCarree(central_longitude=180)
    z_norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    z_cmap = "RdBu_r"
    env_levels = [16, 18]
    n_days = len(RELATIVE_DAYS)
    fig = plt.figure(figsize=(11, 3.2 * n_days))
    gs = gridspec.GridSpec(
        n_days + 1, 2,
        figure=fig,
        hspace=0.07, wspace=0.04,
        height_ratios=[1] * n_days + [0.04])
    panels = [
        [(wx_pos, wy_pos, z200_pos, env_pos, day),
         (wx_neg, wy_neg, z200_neg, env_neg, day)]
        for day in RELATIVE_DAYS]

    cf_last = None
    all_axes = []

    for row, row_panels in enumerate(panels):
        for col, (wx_all, wy_all, z200_all, env_all, day) in enumerate(row_panels):
            ax = fig.add_subplot(gs[row, col], projection=proj)
            all_axes.append(ax)

            wx_map   = wx_all.sel(relative_day=day).sel(latitude=slice(90, 20))
            wy_map   = wy_all.sel(relative_day=day).sel(latitude=slice(90, 20))
            z200_map = z200_all.sel(relative_day=day).squeeze()
            env_map  = env_all.sel(relative_day=day).squeeze()

            ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=0)
            ax.add_feature(cfeature.OCEAN,     facecolor="white",     zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.set_extent([90, 300, 0, 90], crs=ccrs.PlateCarree())

            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                  alpha=0.5, linestyle="--")
            gl.top_labels   = False
            gl.right_labels = False
            gl.left_labels  = (col == 0)
            gl.bottom_labels = (row == n_days - 1)
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}

            cf = ax.pcolormesh(
                z200_map.longitude, z200_map.latitude, z200_map,
                cmap=z_cmap, norm=z_norm, shading="auto",
                transform=ccrs.PlateCarree(), zorder=1)
            cf_last = cf

            ax.contour(
                env_map.longitude, env_map.latitude, env_map,
                levels=env_levels,
                colors="black", linewidths=1, linestyles="solid",
                transform=ccrs.PlateCarree(), zorder=2)

            s = QUIVER_SKIP
            lons = wx_map.longitude.values[::s]
            lats = wx_map.latitude.values[::s]
            U    = wx_map.values[::s, ::s]
            V    = wy_map.values[::s, ::s]
            mag  = np.sqrt(U**2 + V**2)
            U    = np.where(mag >= WAF_THRESH, U, np.nan)
            V    = np.where(mag >= WAF_THRESH, V, np.nan)
            lon2d, lat2d = np.meshgrid(lons, lats)
            ax.quiver(lon2d, lat2d, U, V,
                     transform=ccrs.PlateCarree(),
                     scale=1.2, scale_units="inches", width=0.005,
                     headwidth=4, headlength=4,
                     color="black", zorder=3, alpha=0.9)

            ax.set_title(f"Day {day:+d}", fontsize=11, loc="left", pad=3)

    labels = [f"Positive CGT", f"Negative CGT"]
    for col, label in enumerate(labels):
        ax_top = all_axes[col]
        x0, y0, w, h = ax_top.get_position().bounds
        fig.text(
            x0 + w / 2,
            y0 + h + 0.01,
            label,
            ha="center", va="bottom",
            fontsize=15,
            transform=fig.transFigure)

    cbar_ax = fig.add_subplot(gs[n_days, :])
    cbar_ax.set_axis_off()

    cbar = fig.colorbar(cf_last, ax=all_axes, orientation="horizontal", pad=0.03, aspect=30, fraction=0.04)
    cbar.set_label("200 hPa Geopotential Height Anomaly (m)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

events_ds = xr.open_dataset(EVENTS_FILE)

u_files  = collect_files("u_component_of_wind_0_daily-mean.nc")
v_files  = collect_files("v_component_of_wind_0_daily-mean.nc")
gz_files = collect_files("geopotential_stream-oper_daily-mean.nc")

def preprocess_file(ds):
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds.sel(latitude=slice(90, 0))

open_mfdataset_time = functools.partial(
    xr.open_mfdataset,
    combine="nested", concat_dim="time",
    coords="minimal", compat="override", join="override",
    decode_times=True, preprocess=preprocess_file)

ds_u  = open_mfdataset_time(u_files,  chunks=TIME_CHUNKS)
ds_v  = open_mfdataset_time(v_files,  chunks=TIME_CHUNKS)
ds_gz = open_mfdataset_time(gz_files, chunks=TIME_CHUNKS)

u_full = preprocess_data(ds_u, "u").sel(pressure_level=PLUMB_LEVEL)
v_full = preprocess_data(ds_v, "v").sel(pressure_level=PLUMB_LEVEL)
z_full = preprocess_data(ds_gz, "z").sel(pressure_level=PLUMB_LEVEL)

jja_mask_u = u_full.time.dt.month.isin([6, 7, 8])
jja_mask_v = v_full.time.dt.month.isin([6, 7, 8])
U_bg = u_full.sel(time=jja_mask_u).mean("time").compute().astype(np.float32)
V_bg = v_full.sel(time=jja_mask_v).mean("time").compute().astype(np.float32)

available_days = get_available_days(u_full.time, v_full.time, z_full.time)
print(f"{len(available_days)} days have u, v, and z all available")

valid_events_pos = get_valid_events(events_ds, POS_CGT_MONTHS, available_days, "Positive CGT")
valid_events_neg = get_valid_events(events_ds, NEG_CGT_MONTHS, available_days, "Negative CGT")

needed_dates = np.array(sorted(set(get_needed_dates(valid_events_pos)) | set(get_needed_dates(valid_events_neg))))

z_clim   = calculate_climatology(z_full).compute().astype(np.float32)
z_subset = z_full.sel(time=needed_dates, method="nearest",
                      tolerance=np.timedelta64(1, "D")).compute().astype(np.float32)

z_anom = (z_subset.groupby(z_subset.time.dt.dayofyear) - z_clim).compute().astype(np.float32)
z_anom = z_anom.chunk({"longitude": -1})
z_anom = zonal_wavenumber_filter(z_anom, K_MIN, K_MAX).compute().astype(np.float32)
z_anom_m = z_anom / 9.81

v_mean = v_full.mean("time").compute().astype(np.float32)
v_subset = v_full.sel(time=needed_dates, method="nearest", tolerance=np.timedelta64(1, "D")).compute().astype(np.float32)

v_anom = (v_subset - v_mean).chunk({"longitude": -1})
v_filt = zonal_wavenumber_filter(v_anom, K_MIN, K_MAX).compute().astype(np.float32)
envelope = calculate_rossby_wave_envelope(v_filt).compute().astype(np.float32)

del v_subset, v_anom
ds_u.close()
ds_v.close()
ds_gz.close()

z200_pos = create_composite(valid_events_pos, z_anom_m)
z200_neg = create_composite(valid_events_neg, z_anom_m)
env_pos  = create_composite(valid_events_pos, envelope)
env_neg  = create_composite(valid_events_neg, envelope)

psi_anom = prepare_streamfunction_anom(z_anom).compute().astype(np.float32)
Wx, Wy   = compute_tn_flux(psi_anom, U_bg, V_bg, PLUMB_LEVEL)
wx_pos, wy_pos = incremental_composite(valid_events_pos, Wx, Wy)
wx_neg, wy_neg = incremental_composite(valid_events_neg, Wx, Wy)

plot_composite(wx_pos, wy_pos, z200_pos, env_pos,
               wx_neg, wy_neg, z200_neg, env_neg,
               "WAF_HeatWave_Composite_Evolution.png",
               n_pos=len(valid_events_pos.event), n_neg=len(valid_events_neg.event))
