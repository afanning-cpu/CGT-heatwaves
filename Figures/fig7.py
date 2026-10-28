#group velocity evolution plot

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.signal import hilbert
from scipy.stats import mannwhitneyu, gaussian_kde

ENV_THRESH = 16.0
FOCUS_DAYS = [-10, -5, 0, 5]
RAD_PER_DEG = np.rad2deg(1.0)
LON_MIN, LON_MAX = 120, 300 
LAT_MIN, LAT_MAX = 20, 65

def draw_sector_map(ax, lon_min=LON_MIN, lon_max=LON_MAX, lat_min=LAT_MIN, lat_max=LAT_MAX):
    ax.set_extent([lon_min - 15, lon_max + 10, lat_min - 15, lat_max + 10],
                   crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="0.98", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False

    box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(box_lons, box_lats, color="black", linewidth=2.2,
            transform=ccrs.PlateCarree(), zorder=3)
    ax.fill(box_lons, box_lats, color="black", alpha=0.08,
            transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title("Analysis domain: Pacific-North America sector", fontsize=12)

def local_wavevector(v_filt_day):
    def analytic_phase(x):
        analytic = hilbert(x, axis=-1)
        return np.unwrap(np.angle(analytic), axis=-1)

    phase = xr.apply_ufunc(
        analytic_phase, v_filt_day,
        input_core_dims=[["longitude"]],
        output_core_dims=[["longitude"]],
        vectorize=True)

    lat_rad = np.deg2rad(v_filt_day.latitude)
    cos_lat_da = xr.DataArray(np.cos(lat_rad), coords=[v_filt_day.latitude], dims=["latitude"])

    dphase_dlambda = (phase.differentiate("longitude", edge_order=2) * RAD_PER_DEG)
    dphase_dphi    = (phase.differentiate("latitude",  edge_order=2) * RAD_PER_DEG)

    k = dphase_dlambda / (EARTH_RADIUS * cos_lat_da.where(cos_lat_da > 1e-3))
    l = dphase_dphi / EARTH_RADIUS
    return k, l

def local_group_velocity(v_filt_day, U_bg, V_bg):
    lat_rad = np.deg2rad(v_filt_day.latitude)
    beta = 2 * OMEGA * np.cos(lat_rad) / EARTH_RADIUS
    beta_da = xr.DataArray(beta, coords=[v_filt_day.latitude], dims=["latitude"])

    k, l = local_wavevector(v_filt_day)
    K2 = k**2 + l**2
    K2_safe = K2.where(K2 > 1e-16)

    cgx = U_bg + beta_da * (k**2 - l**2) / (K2_safe**2)
    cgy = V_bg + 2 * beta_da * k * l / (K2_safe**2)
    return cgx, cgy

def sector_mask(da):
    lon = da.longitude
    lat = da.latitude
    lon_ok = (lon >= LON_MIN) & (lon <= LON_MAX)
    lat_ok = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    return lon_ok & lat_ok

def compute_phase_conditioned_background(u_full, v_full, month_list):
    u_year = u_full.time.dt.year.values
    u_month = u_full.time.dt.month.values
    v_year = v_full.time.dt.year.values
    v_month = v_full.time.dt.month.values

    month_set = set(month_list)
    u_mask = np.array([(y, m) in month_set for y, m in zip(u_year, u_month)])
    v_mask = np.array([(y, m) in month_set for y, m in zip(v_year, v_month)])

    U_bg_cond = u_full.sel(time=u_mask).mean("time").compute().astype(np.float32)
    V_bg_cond = v_full.sel(time=v_mask).mean("time").compute().astype(np.float32)
    return U_bg_cond, V_bg_cond

def event_group_velocity_distribution(events_ds, v_filt, envelope, U_bg, V_bg, day):
    speed_vals = []
    for i in range(len(events_ds.event)):
        start = events_ds.start_date.isel(event=i).values
        date = start + np.timedelta64(day, "D")
        try:
            v_day = v_filt.sel(time=date, method="nearest", tolerance=np.timedelta64(1, "D"))
            env_day = envelope.sel(time=date, method="nearest", tolerance=np.timedelta64(1, "D"))
        except KeyError:
            continue
        cgx_field, cgy_field = local_group_velocity(v_day, U_bg, V_bg)
        speed_field = np.sqrt(cgx_field**2 + cgy_field**2)
        mask = (env_day > ENV_THRESH) & sector_mask(env_day)
        if mask.sum() == 0:
            continue
        speed_vals.append(float(speed_field.where(mask).mean()))
    return np.array(speed_vals)

def summarize(name, arr):
    return f"{name}: mean={np.nanmean(arr):.2f}, std={np.nanstd(arr):.2f}, n={np.sum(~np.isnan(arr))}"

def plot_smoothed_kde(ax, data_pos, data_neg, p_value, bw_method=None, n_grid=400, show_legend=True):
    x_min = min(data_pos.min(), data_neg.min())
    x_max = max(data_pos.max(), data_neg.max())
    pad = 0.1 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, n_grid)
    y_max = 0.30
    mean_vals = {}
    densities = {}
    for data, color, label in [
        (data_pos, "firebrick", "Positive CGT"),
        (data_neg, "steelblue", "Negative CGT"),]:
        kde = gaussian_kde(data, bw_method=bw_method)
        density = kde(x_grid)
        mean_val = np.nanmean(data)
        mean_vals[color] = mean_val
        densities[color] = kde
        ax.plot(x_grid, density, color=color, linewidth=2, label=f"{label}")
        ax.fill_between(x_grid, density, color=color, alpha=0.3)
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.2)
    ax.set_ylim(top=y_max)
    ax.annotate(f"{mean_vals['firebrick']:.1f} m/s",
                xy=(mean_vals["firebrick"], densities["firebrick"](mean_vals["firebrick"])[0]),
                xytext=(-10, 20), textcoords="offset points", ha="right", va="bottom",
                color="firebrick", fontsize=10, fontweight="bold")
    ax.annotate(f"{mean_vals['steelblue']:.1f} m/s",
                xy=(mean_vals["steelblue"], densities["steelblue"](mean_vals["steelblue"])[0]),
                xytext=(24, 18), textcoords="offset points", ha="left", va="bottom",
                color="steelblue", fontsize=10, fontweight="bold")
    if show_legend:
        ax.legend(fontsize=8, loc="upper right", title=f"p = {p_value:.4f}", title_fontsize=8)
    else:
        ax.text(0.98, 0.98, f"p = {p_value:.4f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.8"))

U_bg_pos, V_bg_pos = compute_phase_conditioned_background(u_full, v_full, POS_CGT_MONTHS)
U_bg_neg, V_bg_neg = compute_phase_conditioned_background(u_full, v_full, NEG_CGT_MONTHS)

results = {}
for day in FOCUS_DAYS:
    speed_pos = event_group_velocity_distribution(valid_events_pos, v_filt, envelope, U_bg_pos, V_bg_pos, day)
    speed_neg = event_group_velocity_distribution(valid_events_neg, v_filt, envelope, U_bg_neg, V_bg_neg, day)

    u_stat, p_speed = mannwhitneyu(speed_pos, speed_neg, alternative="two-sided")
    results[day] = dict(speed_pos=speed_pos, speed_neg=speed_neg, p_value=p_speed)

fig = plt.figure(figsize=(11, 12.5))
gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1, 1], hspace=0.3, wspace=0.2)

map_ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree(central_longitude=180))
draw_sector_map(map_ax)

kde_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

for i, (ax, day) in enumerate(zip(kde_axes, FOCUS_DAYS)):
    r = results[day]
    plot_smoothed_kde(ax, r["speed_pos"], r["speed_neg"], r["p_value"], show_legend=(i == 0))
    ax.set_xlabel("Group velocity magnitude (m/s)")
    ax.set_ylabel("Density")
    ax.set_title(f"Day {day:+d}: speed distribution")

plt.savefig("fig7_group_velocity_kde_withmap.png", dpi=300, bbox_inches="tight")
plt.show()
