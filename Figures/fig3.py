import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from scipy.ndimage import gaussian_filter

ds = xr.open_dataset('heatwaves_1951_2024.nc')

positive_cgt_months = [
    '1952-07', '1954-07', '1955-07', '1955-08', '1956-06', '1956-07',
    '1959-07', '1961-06', '1961-07', '1962-06', '1963-07', '1964-06',
    '1964-07', '1967-07', '1967-08', '1969-08', '1970-06', '1970-08',
    '1971-08', '1972-06', '1972-08', '1973-07', '1974-06', '1975-06',
    '1975-07', '1975-08', '1976-06', '1976-08', '1978-06', '1981-07',
    '1981-08', '1983-08', '1984-06', '1984-07', '1984-08', '1986-06',
    '1988-06', '1989-06', '1989-07', '1991-08', '1994-06', '1994-07',
    '1994-08', '1996-06', '1996-07', '1996-08', '1999-06', '1999-07',
    '2000-06', '2000-07', '2000-08', '2001-06', '2001-07', '2003-07',
    '2006-06', '2006-08', '2008-06', '2008-07', '2010-07', '2011-07',
    '2012-07', '2012-08', '2013-06', '2013-07', '2013-08', '2018-06',
    '2018-07', '2018-08', '2021-06', '2021-07', '2022-06', '2022-07',
    '2022-08', '2024-08']

negative_cgt_months = [
    '1951-06', '1952-06', '1954-06', '1954-08', '1956-08', '1957-06',
    '1957-08', '1958-06', '1958-07', '1958-08', '1959-06', '1960-08',
    '1962-07', '1962-08', '1963-08', '1964-08', '1965-06', '1965-07',
    '1965-08', '1966-08', '1968-07', '1968-08', '1969-06', '1974-08',
    '1976-07', '1977-08', '1979-07', '1980-07', '1980-08', '1981-06',
    '1982-06', '1982-07', '1983-06', '1985-06', '1986-08', '1987-06',
    '1987-07', '1987-08', '1988-07', '1990-08', '1991-07', '1992-06',
    '1992-07', '1992-08', '1993-06', '1993-07', '1993-08', '1995-07',
    '1997-06', '1997-07', '1998-06', '1998-07', '2003-06', '2004-08',
    '2005-06', '2007-07', '2009-06', '2009-07', '2009-08', '2010-06',
    '2011-06', '2012-06', '2014-08', '2015-07', '2015-08', '2017-06',
    '2019-06', '2019-07', '2019-08', '2020-07', '2020-08', '2023-07',
    '2023-08', '2024-06']

start_dates = ds['start_date'].values
event_year_months = np.array([str(np.datetime64(date, 'M')) for date in start_dates])

pos_mask = np.array([ym in positive_cgt_months for ym in event_year_months])
neg_mask = np.array([ym in negative_cgt_months for ym in event_year_months])
neu_mask = ~(pos_mask | neg_mask)

def get_latlon(mask):
    lat = ds['max_lat'].values[mask]
    lon = ((ds['max_lon'].values[mask] + 180) % 360) - 180
    return lat, lon

pos_lat, pos_lon = get_latlon(pos_mask)
neg_lat, neg_lon = get_latlon(neg_mask)
neu_lat, neu_lon = get_latlon(neu_mask)

lon_bins = np.linspace(-125, -70, 30)
lat_bins = np.linspace(25, 50, 20)
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

sigma = 2

def make_prob(lat, lon, n):
    h, _, _ = np.histogram2d(lat, lon, bins=[lat_bins, lon_bins])
    return gaussian_filter(h / n, sigma=sigma)

pos_prob = make_prob(pos_lat, pos_lon, pos_mask.sum())
neg_prob = make_prob(neg_lat, neg_lon, neg_mask.sum())

all_lat = np.concatenate([pos_lat, neg_lat, neu_lat])
all_lon = np.concatenate([pos_lon, neg_lon, neu_lon])
total = pos_mask.sum() + neg_mask.sum() + neu_mask.sum()
all_prob = make_prob(all_lat, all_lon, total)

low_data = all_prob < 0.001

def likelihood_ratio(cond_prob):
    lr = np.where(all_prob > 0, cond_prob / all_prob, np.nan)
    #lr[low_data] = np.nan
    return lr

pos_lr = likelihood_ratio(pos_prob)
neg_lr = likelihood_ratio(neg_prob)
clim_cmap = plt.cm.YlOrRd
lr_cmap   = LinearSegmentedColormap.from_list("bwr_custom",
                [(0, 0, 1), (1, 1, 1), (1, 0, 0)], N=256)

lr_norm   = LogNorm(vmin=0.25, vmax=4.0)
clim_norm = Normalize(vmin=0, vmax=np.nanmax(all_prob))

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[1, 1, 1],
    height_ratios=[1, 1],
    hspace=0.35,
    wspace=0.08)

ax_clim = fig.add_subplot(gs[:, :2], projection=ccrs.PlateCarree())

ax_pos = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax_neg = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())

panels = [
    (ax_clim, 'Climatological Heat Wave Probability (1951-2024)',       all_prob, clim_cmap, clim_norm,
     'Probability',                                          '(a)', 18),
    (ax_pos,  'Positive CGT Heat Wave Likelihood Ratio',    pos_lr,   lr_cmap,   lr_norm,
     'Likelihood Ratio (Positive CGT / Climatology)',                        '(b)', 14),
    (ax_neg,  'Negative CGT Heat Wave Likelihood Ratio',    neg_lr,   lr_cmap,   lr_norm,
     'Likelihood Ratio (Negative CGT / Climatology)',                        '(c)', 14),]

for ax, title, data, cmap, norm, clabel, panel_label, title_fontsize in panels:
    ax.set_extent([-125, -70, 25, 47])

    pcm = ax.pcolormesh(
        lon2d, lat2d, data,
        cmap=cmap, norm=norm,
        shading='auto',
        transform=ccrs.PlateCarree() )
    
    ax.contourf(
        lon2d, lat2d, low_data.astype(int),
        levels=[0.5, 1.5],
        hatches=['///'],
        colors='none',
        transform=ccrs.PlateCarree(),
        zorder=5)

    ax.add_feature(
                cfeature.OCEAN.with_scale("50m"),
                facecolor="white",
                edgecolor="none",
                zorder=5)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.8)
    ax.add_feature(cfeature.STATES,    linewidth=0.4)

    cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal',
                        pad=0.04, shrink=0.85, aspect=28)
    cbar.set_label(clabel, fontsize=14)

    if norm is lr_norm:
        cbar.set_ticks([0.25, 0.5, 1.0, 2.0, 4.0])
        cbar.set_ticklabels(['0.25', '0.5', '1.0', '2.0', '4.0'])
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(title, fontsize=title_fontsize, pad=8)
    ax.text(-0.03, 1.14, panel_label, transform=ax.transAxes,
            fontsize=15, va='top')

#fig.suptitle(
    #'U.S. Heatwave Hotspot Analysis: 1951–2024',
    #fontsize=13, fontweight='bold', y=1.01)

plt.savefig(
    'fig3_Hotspotmap_3panel.png',
    dpi=300, bbox_inches='tight')

plt.show()
