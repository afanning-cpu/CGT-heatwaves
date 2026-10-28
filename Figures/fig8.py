#surface sensible heat flux evolution composite

import functools
import glob
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import mannwhitneyu
import pandas as pd
from collections import defaultdict
from global_land_mask import globe

DATA_DIR_JJA  = "1951_2024_sshf"
DATA_DIR_MS   = "1951_2024_MS_sshf"
EVENTS_FILE   = "all_heatwave_events_1951_2024.nc"
OUTFILE       = "SSHF_CGT_Evolution.png"

YEARS         = np.arange(1951, 2025)
P_THRESHOLD   = 0.05
RELATIVE_DAYS = [-20, -15, -10, -5, 0, 5]
US_EXTENT     = [-130, -60, 23, 52]

TIME_CHUNKS = {"time": 365, "latitude": -1, "longitude": -1}

#Region boxes (matched to the fig2 composite/distribution figure)
REGION_BOXES = {
    'Western U.S.':    dict(lon_min=-125, lon_max=-105, lat_min=30, lat_max=50, color='black'),
    'SE U.S.':         dict(lon_min=-105, lon_max=-75,  lat_min=25, lat_max=37, color='blue'),
    'N. Great Plains': dict(lon_min=-105, lon_max=-85,  lat_min=37, lat_max=50, color='green'),
}

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

def may_dir(year):
    return f"{DATA_DIR_MS}/era5_daily_sl_may_{year}"

def sep_dir(year):
    return f"{DATA_DIR_MS}/era5_daily_sl_sep_{year}"

def find_data_file(dir_path, prefix):
    if not os.path.isdir(dir_path):
        return None
    matches = sorted(glob.glob(os.path.join(dir_path, f"{prefix}*")))
    return matches[0] if matches else None

def collect_sshf_files():
    """Gather every year's sshf file across the JJA directory and the
    May/September shoulder-month directories"""
    jja_files, may_files, sep_files = [], [], []
    incomplete_years = {}
    for year in YEARS:
        y = int(year)

        jja_p = f"{DATA_DIR_JJA}/{y}_sshf.nc"
        has_jja = os.path.exists(jja_p)
        if has_jja:
            jja_files.append(jja_p)

        m_dir = may_dir(y)
        m_p = find_data_file(m_dir, f"era5_daily_sl_may_{y}")
        has_may = m_p is not None
        if has_may:
            may_files.append(m_p)

        s_dir = sep_dir(y)
        s_p = find_data_file(s_dir, f"era5_daily_sl_sep_{y}")
        has_sep = s_p is not None
        if has_sep:
            sep_files.append(s_p)

        missing = [
            (name, d) for name, found, d in
            [("JJA", has_jja, DATA_DIR_JJA), ("May", has_may, m_dir), ("Sept", has_sep, s_dir)]
            if not found]
        if missing:
            incomplete_years[y] = missing

    #print(f"  JJA files found:  {len(jja_files)} / {len(YEARS)} years")
    #print(f"  May files found:  {len(may_files)} / {len(YEARS)} years")
    #print(f"  Sept files found: {len(sep_files)} / {len(YEARS)} years")

    if incomplete_years:
        print(f"  {len(incomplete_years)} year(s) missing at least one file:")
        for y, missing in incomplete_years.items():
            names = ", ".join(name for name, _ in missing)
            print(f"    {y}: missing {names}")
            for name, d in missing:
                if name == "JJA":
                    continue  # JJA is a single file, not a folder -- nothing to list
                if os.path.isdir(d):
                    actual = sorted(os.listdir(d))
                    print(f"{d} actually contains: {actual}")
                else:
                    print(f"{d} does not exist")
    else:
        print("Every year has all three files.")

    return jja_files + may_files + sep_files

def preprocess_file(ds):
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds.sel(latitude=slice(90, 0))

def convert_sshf(ds):
    da = ds["sshf"].where(np.abs(ds["sshf"]) < 1e25)
    da = -da / 86400.0
    if float(da.longitude.max()) > 180:
        lon = da.longitude.values.copy()
        lon[lon > 180] -= 360
        da = da.assign_coords(longitude=lon).sortby("longitude")
    return da.sortby("time")

def calculate_climatology(da):
    return da.groupby(da.time.dt.dayofyear).mean("time")

def filter_events_by_month(events_ds, month_list):
    month_set = set(month_list)
    event_months = [
        (pd.Timestamp(d).year, pd.Timestamp(d).month)
        for d in events_ds.start_date.values
    ]
    mask = [em in month_set for em in event_months]
    return events_ds.sel(event=mask)

def get_available_days(time_array):
    return set(pd.to_datetime(time_array.values).normalize())

def get_valid_events(events_ds, month_list, available_days, label):
    """ Keep only events where every RELATIVE_DAYS offset lands on a day
    we have sshf data for """
    events = filter_events_by_month(events_ds, month_list)
    valid_idx = []
    invalid_info = []
    for i in range(len(events.event)):
        start = pd.Timestamp(events.start_date.isel(event=i).values)
        missing_days = [
            d for d in RELATIVE_DAYS
            if (start + pd.Timedelta(days=d)).normalize() not in available_days
        ]
        if not missing_days:
            valid_idx.append(i)
        else:
            invalid_info.append((start, missing_days))

    valid_events = events.isel(event=valid_idx)
    #print(f"{label}: {len(valid_events.event)} of {len(events.event)} candidate "
          #"events have complete SSHF data across all {len(RELATIVE_DAYS)} composite days")

    if invalid_info:
        #print(f"{label}: {len(invalid_info)} events excluded due to missing data:")
        for start, missing_days in invalid_info:
            missing_dates = [(start + pd.Timedelta(days=d)).normalize().date() for d in missing_days]
            #print(f"  start={start.date()}  missing offsets={missing_days}  missing dates={missing_dates}")

    return valid_events

def collect_event_composites(events_ds, sshf_anom, relative_days):
    event_maps = defaultdict(list)
    for i in range(len(events_ds.event)):
        start_date = events_ds.start_date.isel(event=i).values
        event_dates = [start_date + np.timedelta64(d, "D") for d in relative_days]
        try:
            comp = sshf_anom.sel(
                time=event_dates,
                method="nearest",
                tolerance=np.timedelta64(1, "D")
            )
            if len(comp.time) == len(relative_days):
                comp_np = comp.values
                for j, day in enumerate(relative_days):
                    event_maps[day].append(comp_np[j])
        except Exception:
            continue
    return {day: np.stack(maps, axis=0) for day, maps in event_maps.items()}

def create_mean_composite(event_maps, sshf_anom, relative_days):
    lats = sshf_anom.latitude.values
    lons = sshf_anom.longitude.values
    mean_list = [np.nanmean(event_maps[d], axis=0) for d in relative_days]
    arr = np.stack(mean_list, axis=0)
    return xr.DataArray(
        arr,
        dims=["relative_day", "latitude", "longitude"],
        coords={"relative_day": relative_days, "latitude": lats, "longitude": lons})

def get_hw_exclusion_dates(events_ds, relative_days):
    day_min, day_max = min(relative_days), max(relative_days)
    excluded = set()
    for i in range(len(events_ds.event)):
        start_date = pd.Timestamp(events_ds.start_date.isel(event=i).values)
        for offset in range(day_min, day_max + 1):
            excluded.add((start_date + pd.Timedelta(days=offset)).normalize())
    return excluded

def build_background_sample(da, month_list, excluded_dates):
    month_set = set(month_list)
    times = pd.to_datetime(da.time.values)
    mask = np.array([
        (t.year, t.month) in month_set and t.normalize() not in excluded_dates
        for t in times])
    return da.values[mask]

def compute_mwu_pvalue_vs_background(event_maps, background, relative_days):
    nlat, nlon = background.shape[1], background.shape[2]
    pval_maps = {}
    for day in relative_days:
        event_data = event_maps[day]
        pvals = np.ones((nlat, nlon))
        for iy in range(nlat):
            for ix in range(nlon):
                ev = event_data[:, iy, ix]
                bg = background[:, iy, ix]
                ev = ev[~np.isnan(ev)]
                bg = bg[~np.isnan(bg)]
                if len(ev) < 3 or len(bg) < 3:
                    pvals[iy, ix] = np.nan
                    continue
                _, p = mannwhitneyu(ev, bg, alternative="two-sided")
                pvals[iy, ix] = p
        pval_maps[day] = pvals
    return pval_maps

sshf_files = collect_sshf_files()
events_ds = xr.open_dataset(EVENTS_FILE)
open_mfdataset_time = functools.partial(
    xr.open_mfdataset,
    combine="nested", concat_dim="time",
    coords="minimal", compat="override", join="override",
    decode_times=True, preprocess=preprocess_file,
    engine="netcdf4",)

ds_sshf = open_mfdataset_time(sshf_files, chunks=TIME_CHUNKS)
sshf    = convert_sshf(ds_sshf)

lon_min, lon_max, lat_min, lat_max = US_EXTENT

sshf = sshf.sel(
    latitude=slice(lat_max + 5, lat_min - 5),
    longitude=slice(lon_min - 5, lon_max + 5))

sshf_clim = calculate_climatology(sshf)
sshf_anom = (sshf.groupby(sshf.time.dt.dayofyear) - sshf_clim).sortby("time").compute()

available_days = get_available_days(sshf_anom.time)
#print(f"{len(available_days)} days have SSHF data available")

events_pos = get_valid_events(events_ds, POS_CGT_MONTHS, available_days, "Positive CGT")
events_neg = get_valid_events(events_ds, NEG_CGT_MONTHS, available_days, "Negative CGT")

pos_event_maps = collect_event_composites(events_pos, sshf_anom, RELATIVE_DAYS)
neg_event_maps = collect_event_composites(events_neg, sshf_anom, RELATIVE_DAYS)

sshf_pos_mean = create_mean_composite(pos_event_maps, sshf_anom, RELATIVE_DAYS)
sshf_neg_mean = create_mean_composite(neg_event_maps, sshf_anom, RELATIVE_DAYS)

#Significance: each phase's HW composite tested against that phase's own non-HW background (all non-event days in that phase's months)
excluded_dates_pos = get_hw_exclusion_dates(events_pos, RELATIVE_DAYS)
excluded_dates_neg = get_hw_exclusion_dates(events_neg, RELATIVE_DAYS)

background_pos = build_background_sample(sshf_anom, POS_CGT_MONTHS, excluded_dates_pos)
background_neg = build_background_sample(sshf_anom, NEG_CGT_MONTHS, excluded_dates_neg)

#print(f"Positive CGT background sample size: {background_pos.shape[0]} days")
#print(f"Negative CGT background sample size: {background_neg.shape[0]} days")

pval_maps_pos = compute_mwu_pvalue_vs_background(pos_event_maps, background_pos, RELATIVE_DAYS)
pval_maps_neg = compute_mwu_pvalue_vs_background(neg_event_maps, background_neg, RELATIVE_DAYS)

us_proj  = ccrs.PlateCarree()
data_crs = ccrs.PlateCarree()

norm_sshf = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)

n_days = len(RELATIVE_DAYS)
fig = plt.figure(figsize=(11, 3.0 * n_days))

gs = gridspec.GridSpec(
    n_days + 1, 2,
    figure=fig,
    hspace=0.04,
    wspace=0.04,
    height_ratios=[1] * n_days + [0.04])

day_labels = {
    -20: "Day -20", -15: "Day -15", -10: "Day -10",
    -5:  "Day -5",  0:   "Day 0",   5:   "Day +5",}

panels = [
    [(sshf_pos_mean, day), (sshf_neg_mean, day)]
    for day in RELATIVE_DAYS]

LABEL_ROW, LABEL_COL = n_days - 1, 0

cf_last = None
all_axes = []

for row, row_panels in enumerate(panels):
    for col, (sshf_mean, day) in enumerate(row_panels):

        ax = fig.add_subplot(gs[row, col], projection=us_proj)
        all_axes.append(ax)

        sshf_map = sshf_mean.sel(relative_day=day).squeeze()

        lons = sshf_map.longitude.values
        lats = sshf_map.latitude.values
        lon2d, lat2d = np.meshgrid(lons, lats)
        land_mask = globe.is_land(lat2d, lon2d)

        sshf_land = np.where(land_mask, sshf_map.values, np.nan)

        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")
        ax.set_extent(US_EXTENT, crs=data_crs)
        ax.set_aspect("equal")

        cf = ax.pcolormesh(
            lons, lats, sshf_land,
            cmap="RdBu_r",
            norm=norm_sshf,
            shading="auto",
            transform=data_crs,
            zorder=1)
        cf_last = cf

        pval_maps_for_col = pval_maps_pos if col == 0 else pval_maps_neg
        pvals = pval_maps_for_col[day]
        sig   = (pvals < P_THRESHOLD) & land_mask

        ax.scatter(
            lon2d[sig][::10], lat2d[sig][::10],
            s=0.3, color="black", alpha=0.6,
            transform=data_crs, zorder=2)

        #Region boxes
        region_handles = []
        for label, r in REGION_BOXES.items():
            w, h = r['lon_max'] - r['lon_min'], r['lat_max'] - r['lat_min']
            ax.add_patch(mpatches.Rectangle(
                (r['lon_min'], r['lat_min']), w, h,
                linewidth=1.2, edgecolor=r['color'], facecolor='none',
                transform=data_crs, zorder=5
            ))
            if row == 0 and col == 0:
                region_handles.append(
                    mpatches.Patch(edgecolor=r['color'], facecolor='none',
                                   linewidth=1.5, label=label))

        if row == 0 and col == 0:
            leg = ax.legend(handles=region_handles, loc='upper right', fontsize=6,
                             framealpha=0.85, title='Subregions', title_fontsize=6)
            leg.set_zorder(10)

        show_lat_labels = (col == 0)
        show_lon_labels = (row == n_days - 1)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3, color="gray",
            alpha=0.4, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = show_lat_labels
        gl.bottom_labels = show_lon_labels
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}
        gl.rotate_labels = False
        gl.xpadding = 12
        ax.set_title(day_labels[day], fontsize=11, loc="left", pad=3)

for col, label in enumerate(["Positive CGT", "Negative CGT"]):
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

cbar = fig.colorbar(
    cf_last,
    ax=all_axes,
    orientation="horizontal",
    pad=0.03,
    aspect=30,
    fraction=0.04)
cbar.set_label("Sensible Heat Flux Anomaly (W m\u207b\u00b2)", fontsize=11)
cbar.ax.tick_params(labelsize=9)
plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.show()
