#t2m composite and regional distributions figure

import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import regionmask
from eofs.xarray import Eof
from scipy import signal
from scipy.stats import gaussian_kde, mannwhitneyu

GEOPOTENTIAL_FILE = "MonthlyGeopotential1941to2024.nc"
T2M_PATTERN = "1951_2024_Dailyavg_2mT/*_2mT_dailyavg*"
OUTPUT_FILE = "CGT_Composite_and_Regional_Distributions_1951_2024.png"

START_YEAR, END_YEAR = 1951, 2024
JJA = [6, 7, 8]
US_LON = (-125, -65)
US_LAT = (25, 50)
HW_AREA_FRACTION = 0.05  #a heat wave day: >5% of region's land cells above daily 95th pct

REGIONS = [
    ("Whole U.S.",      -125, -65,  25, 50, "grey"),
    ("Western U.S.",    -125, -105, 30, 50, "black"),
    ("SE U.S.",         -105, -75,  25, 37, "blue"),
    ("N. Great Plains", -105, -85,  37, 50, "green"),]

def detrend_dim(data, dim):
    def _detrend(y):
        if np.all(np.isnan(y)):
            return y
        return signal.detrend(y, type="linear")

    return xr.apply_ufunc(
        _detrend, data,
        input_core_dims=[[dim]], output_core_dims=[[dim]],
        vectorize=True, dask="parallelized", output_dtypes=[float],)

def land_mask(da):
    mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        da.longitude, da.latitude)
    return mask == 0

def wrap_lon(da):
    return da.assign_coords(
        longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")

def select_us(da):
    return da.sel(latitude=slice(US_LAT[1], US_LAT[0]),
                  longitude=slice(US_LON[0], US_LON[1]))


#1) CGT Index
def compute_pc2():
    ds = xr.open_dataset(GEOPOTENTIAL_FILE)
    z200 = ds["z"].squeeze("pressure_level", drop=True)
    z200 = z200.sel(valid_time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    z200 = z200 / 9.80665  # geopotential -> geopotential height

    summer = z200.sel(valid_time=z200.valid_time.dt.month.isin(JJA))
    anom = summer.groupby("valid_time.month") - summer.groupby("valid_time.month").mean("valid_time")
    anom = detrend_dim(anom, dim="valid_time")

    nh = anom.sel(latitude=slice(90, 0))
    nh = nh.squeeze().drop_vars(
        [v for v in nh.coords if v not in ["valid_time", "latitude", "longitude"]],
        errors="ignore")
    nh = nh.rename({"valid_time": "time"}).transpose("time", "latitude", "longitude")

    coslat = np.cos(np.deg2rad(nh.latitude.values))
    wgts = xr.DataArray(
        np.sqrt(coslat)[:, None] * np.ones((1, nh.longitude.size)),
        coords={"latitude": nh.latitude, "longitude": nh.longitude},
        dims=("latitude", "longitude"))

    solver = Eof(nh, weights=wgts)
    pc2 = solver.pcs(npcs=4, pcscaling=1).isel(mode=1)
    pc2["time"] = nh.time
    return pc2.rename({"time": "valid_time"})


#2) T2M 
def load_t2m_daily():
    files = sorted(glob.glob(T2M_PATTERN))
    t2m = xr.open_mfdataset(
        files, concat_dim="valid_time", combine="nested",
        chunks={"valid_time": 365, "latitude": 50, "longitude": 50},
        parallel=False)["t2m"] - 273.15
    return select_us(wrap_lon(t2m))

def monthly_anomalies(t2m_daily):
    monthly = t2m_daily.resample(valid_time="1MS").mean()
    monthly = monthly.sel(valid_time=monthly.valid_time.dt.month.isin(JJA))
    monthly = monthly.sel(valid_time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    clim = monthly.groupby("valid_time.month").mean("valid_time")
    return monthly.groupby("valid_time.month") - clim

def heat_wave_days_by_region(t2m_daily_land, is_land):
    #Monthly count of heat wave days per region.
    p95_doy = (t2m_daily_land
               .groupby("valid_time.dayofyear")
               .quantile(0.95, dim="valid_time", skipna=True))
    exceed = (t2m_daily_land.groupby("valid_time.dayofyear") > p95_doy).astype(float)
    exceed = exceed.where(is_land)

    hw = {}
    for name, lon_min, lon_max, lat_min, lat_max, _ in REGIONS:
        reg = exceed.sel(latitude=slice(lat_max, lat_min),
                         longitude=slice(lon_min, lon_max))
        frac = reg.mean(dim=["latitude", "longitude"], skipna=True)
        hw_flag = (frac > HW_AREA_FRACTION).astype(float)
        hw[name] = hw_flag.resample(valid_time="1MS").sum(skipna=True)
    return hw

#3) Composite difference with Mann-Whitney U significance
def composite_diff(t2m_anom, pc, alpha=0.05):
    t2m_anom, pc = xr.align(t2m_anom, pc, join="inner")
    pc_33 = pc.quantile(0.33, dim="valid_time")
    pc_67 = pc.quantile(0.67, dim="valid_time")
    pos_months = pc.where(pc >= pc_67, drop=True).valid_time.values
    neg_months = pc.where(pc <= pc_33, drop=True).valid_time.values
    #print(f"Positive CGT months: {len(pos_months)}")
    #print(f"Negative CGT months: {len(neg_months)}")

    t2m_pos = t2m_anom.sel(valid_time=pos_months)
    t2m_neg = t2m_anom.sel(valid_time=neg_months)
    diff = t2m_pos.mean("valid_time") - t2m_neg.mean("valid_time")

    pos_vals = t2m_pos.load().values
    neg_vals = t2m_neg.load().values
    nlat, nlon = pos_vals.shape[1:]
    pvals = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            p = pos_vals[:, i, j]
            n = neg_vals[:, i, j]
            p, n = p[~np.isnan(p)], n[~np.isnan(n)]
            if len(p) >= 3 and len(n) >= 3:
                try:
                    _, pvals[i, j] = mannwhitneyu(p, n, alternative="two-sided")
                except ValueError:
                    pass

    sig = xr.DataArray(pvals < alpha,
                       coords={"latitude": diff.latitude, "longitude": diff.longitude},
                       dims=["latitude", "longitude"])
    return diff, sig

#Plotting figure
def make_cmap(v_min=-3.0, v_max=3.0, n_levels=21, zero_width=0.2):
    levels_neg = np.linspace(v_min, -zero_width, n_levels // 2 + 1)
    levels_pos = np.linspace(zero_width, v_max, n_levels // 2 + 1)
    levels = np.sort(np.unique(np.concatenate([levels_neg, levels_pos])))

    base = plt.get_cmap("RdBu_r")
    colors = []
    for i in range(len(levels) - 1):
        mid = (levels[i] + levels[i + 1]) / 2
        if -zero_width <= mid <= zero_width:
            colors.append([1, 1, 1, 1])
        else:
            nv = (mid - v_min) / (0 - v_min) * 0.5 if mid < 0 else 0.5 + (mid / v_max) * 0.5
            colors.append(base(nv))
    cmap = mcolors.ListedColormap(colors)
    return cmap, mcolors.BoundaryNorm(levels, cmap.N), levels

def plot_map(ax, fig, diff, sig, title, stipple_stride=2):
    proj = ccrs.PlateCarree()
    cmap, norm, levels = make_cmap()

    im = ax.contourf(diff.longitude.values, diff.latitude.values, diff.squeeze().values, cmap=cmap, transform=proj, levels=levels, extend="both", norm=norm)
    ax.set_extent([US_LON[0], US_LON[1], US_LAT[0], US_LAT[1]], crs=proj)

    lon2d, lat2d = np.meshgrid(diff.longitude.values, diff.latitude.values)
    sig2d = sig.values & land_mask(diff).values
    s = (slice(None, None, stipple_stride), slice(None, None, stipple_stride))
    ax.scatter(lon2d[s][sig2d[s]], lat2d[s][sig2d[s]], s=3, color="black", alpha=0.5, transform=proj, zorder=6)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="grey")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.text(-0.02, 1.05, "(a)", transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom", ha="right")
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-125, -60, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(20, 50, 10))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    handles = []
    for name, lon_min, lon_max, lat_min, lat_max, color in REGIONS[1:]:
        ax.add_patch(mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2, edgecolor=color, facecolor="none", transform=proj, zorder=5))
        handles.append(mpatches.Patch(edgecolor=color, facecolor="none",
                                      linewidth=2, label=name))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=10,
                    framealpha=0.85, title="Subregions", title_fontsize=10)
    leg.set_zorder(10)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        label="T2M Anomaly [°C]", pad=0.08, shrink=0.8, aspect=30)
    cbar.set_ticks(np.arange(-3, 4, 1))

def plot_kde_pair(ax, pos_vals, neg_vals, x_eval):
    out = []
    for vals, color, label in [(pos_vals, "firebrick", "Positive CGT"),
                               (neg_vals, "navy", "Negative CGT")]:
        dens = gaussian_kde(vals, bw_method="scott")(x_eval)
        line, = ax.plot(x_eval, dens, color=color, lw=2.5, alpha=0.85, label=label)
        ax.fill_between(x_eval, dens, alpha=0.18, color=color)
        mean = np.mean(vals)
        ax.axvline(mean, color=color, ls="--", lw=1.8, alpha=0.8)
        out.append((mean, line))
    return out
  
def label_side(pos_mean, neg_mean, offset):
    if pos_mean >= neg_mean:
        return ("left", offset), ("right", -offset)
    return ("right", -offset), ("left", offset)

def make_figure(diff, sig, pc2, t2m_daily_land, hw_monthly, title, filename):
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[2, 1, 1], hspace=0.25, wspace=0.35)
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    axes_t2m = [fig.add_subplot(gs[1, i]) for i in range(4)]
    axes_hw = [fig.add_subplot(gs[2, i]) for i in range(4)]

    plot_map(ax_map, fig, diff, sig, title)

    period = slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
    p_slice = pc2.sel(valid_time=period)
    t_slice = t2m_daily_land.sel(valid_time=period)

    upper = np.percentile(p_slice.values, 66.67)
    lower = np.percentile(p_slice.values, 33.33)
    p_df = p_slice.to_dataframe(name="pc2").reset_index()
    p_df["year_month"] = p_df["valid_time"].dt.to_period("M")
    p_df["flag"] = np.where(p_df["pc2"] >= upper, 1,
                            np.where(p_df["pc2"] <= lower, -1, 0))
    flag_lookup = p_df.set_index("year_month")["flag"]

    t_ym = pd.PeriodIndex(t_slice.valid_time.dt.strftime("%Y-%m").values, freq="M")
    daily_flag = xr.DataArray(flag_lookup.reindex(t_ym).values,
                              coords={"valid_time": t_slice.valid_time}, dims="valid_time")

    t_anom = t_slice - t_slice.mean("valid_time")
    t2m_pos = t_anom.where(daily_flag == 1)
    t2m_neg = t_anom.where(daily_flag == -1)

    x_eval = np.linspace(-10, 10, 300)
    t2m_pvals, handles = {}, None
    for ax, (name, lon_min, lon_max, lat_min, lat_max, color) in zip(axes_t2m, REGIONS):
        monthly = []
        for field in (t2m_pos, t2m_neg):
            reg = field.sel(latitude=slice(lat_max, lat_min),
                            longitude=slice(lon_min, lon_max))
            daily_mean = reg.mean(dim=["latitude", "longitude"], skipna=True)
            ym = pd.PeriodIndex(daily_mean.valid_time.dt.strftime("%Y-%m").values, freq="M")
            monthly.append(pd.Series(daily_mean.values, index=ym)
                           .groupby(level=0).mean().dropna().values)
        pos_vals, neg_vals = monthly
        t2m_pvals[name] = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")[1]

        (pos_mean, l_pos), (neg_mean, l_neg) = plot_kde_pair(ax, pos_vals, neg_vals, x_eval)
        if handles is None:
            handles = [l_pos, l_neg]

        offset = 0.9 if abs(pos_mean - neg_mean) < 1.0 else 0.5
        (pos_ha, pos_dx), (neg_ha, neg_dx) = label_side(pos_mean, neg_mean, offset)
        pos_y, neg_y = (0.90, 0.75) if name == "Whole U.S." else (0.80, 0.90)
        ax.text(pos_mean + pos_dx, pos_y, f"{pos_mean:.2f}°C",
                transform=ax.get_xaxis_transform(), color="firebrick",
                fontsize=10, fontweight="bold", ha=pos_ha, va="center")
        ax.text(neg_mean + neg_dx, neg_y, f"{neg_mean:.2f}°C",
                transform=ax.get_xaxis_transform(), color="navy",
                fontsize=10, fontweight="bold", ha=neg_ha, va="center")

        ax.axvline(0, color="black", ls=":", lw=1, alpha=0.45)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color, pad=6)
        ax.set_xlabel("T2M Anomaly (°C)", fontsize=11)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(-9, 9)

    y_max = max(ax.get_ylim()[1] for ax in axes_t2m)
    for ax in axes_t2m:
        ax.set_ylim(0, y_max * 1.15)
    axes_t2m[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    axes_t2m[0].annotate("T2M Anomaly", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=14, fontweight="bold", ha="center", va="center",
                         rotation=90, color="dimgrey")
    axes_t2m[0].text(-0.25, 1.13, "(b)", transform=axes_t2m[0].transAxes,
                     fontsize=18, fontweight="bold", va="bottom", ha="right")

    x_eval_hw = np.linspace(0, 31, 300)
    hw_pvals = {}
    for ax, (name, *_, color) in zip(axes_hw, REGIONS):
        hw_period = hw_monthly[name].sel(valid_time=period)
        hw_al, pc_al = xr.align(hw_period, p_slice, join="inner")
        up = np.percentile(pc_al.values, 66.67)
        lo = np.percentile(pc_al.values, 33.33)
        pos_vals = hw_al.values[pc_al.values >= up]
        neg_vals = hw_al.values[pc_al.values <= lo]
        pos_vals = pos_vals[~np.isnan(pos_vals)]
        neg_vals = neg_vals[~np.isnan(neg_vals)]
        hw_pvals[name] = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")[1]

        (pos_mean, _), (neg_mean, _) = plot_kde_pair(ax, pos_vals, neg_vals, x_eval_hw)

        offset = 2.5 if abs(pos_mean - neg_mean) < 3.0 else 1.0
        (pos_ha, pos_dx), (neg_ha, neg_dx) = label_side(pos_mean, neg_mean, offset)
        ax.text(pos_mean + pos_dx, 0.80, f"{pos_mean:.1f}d",
                transform=ax.get_xaxis_transform(), color="firebrick",
                fontsize=10, fontweight="bold", ha=pos_ha, va="center")
        ax.text(neg_mean + neg_dx + 1.2, 0.90, f"{neg_mean:.1f}d",
                transform=ax.get_xaxis_transform(), color="navy",
                fontsize=10, fontweight="bold", ha=neg_ha, va="center")

        ax.axvline(0, color="black", ls=":", lw=1, alpha=0.45)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color, pad=6)
        ax.set_xlabel("# Heat Wave Days / Month", fontsize=11)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0, 31)

    y_max = max(ax.get_ylim()[1] for ax in axes_hw)
    for ax in axes_hw:
        ax.set_ylim(0, y_max * 1.15)
    axes_hw[0].text(-0.25, 1.13, "(c)", transform=axes_hw[0].transAxes,
                    fontsize=18, fontweight="bold", va="bottom", ha="right")
    axes_hw[0].annotate("# HW Days", xy=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=14, fontweight="bold", ha="center", va="center",
                        rotation=90, color="black")

    plt.tight_layout(rect=[0, 0.03, 1, 1], h_pad=4)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    #print significance testing results
    fmt = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    print("Mann-Whitney U test, positive vs negative CGT months")
    print("T2M anomaly (panel b):")
    for name, p in t2m_pvals.items():
        print(f"  {name:16s} {fmt(p)}")
    print("# Heat wave days / month (panel c):")
    for name, p in hw_pvals.items():
        print(f"  {name:16s} {fmt(p)}")
      
def main():
    pc2 = compute_pc2()
    t2m_daily = load_t2m_daily()
    is_land = land_mask(t2m_daily)
    t2m_daily_land = t2m_daily.where(is_land)
    t2m_anom = monthly_anomalies(t2m_daily)
    diff, sig = composite_diff(t2m_anom, pc2)
    hw_monthly = heat_wave_days_by_region(t2m_daily_land, is_land)
    make_figure(diff, sig, pc2, t2m_daily_land, hw_monthly,
                title=f"CGT T2m Anomaly Difference Composite ({START_YEAR}–{END_YEAR})",
                filename=OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")
  
if __name__ == "__main__":
    main()
