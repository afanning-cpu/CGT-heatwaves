#Scatterplots: CGT and HW Stats
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal
from eofs.xarray import Eof
from scipy.stats import spearmanr, linregress, ttest_ind
import matplotlib.lines as mlines

POSITIVE_CGT_MONTHS = [
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
    '2022-08', '2024-08'
]

NEGATIVE_CGT_MONTHS = [
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
    '2023-08', '2024-06'
]

EVENTS_FILE = "heatwaves_1951_2024.nc"
GEOP_FILE = 'MonthlyGeopotential1941to2024.nc'

events_ds = xr.open_dataset(EVENTS_FILE)
ds = xr.open_dataset(GEOP_FILE)

geopotential_200 = ds['z'].squeeze('pressure_level', drop=True).sel(valid_time=slice('1951-01-01', '2024-12-31')) / 9.80665
summer_data = geopotential_200.sel(valid_time=geopotential_200.valid_time.dt.month.isin([6, 7, 8]))
summer_monthly_anom = (summer_data.groupby('valid_time.month') - summer_data.groupby('valid_time.month').mean('valid_time')).rename({'valid_time': 'time'})

def detrend_dim(data, dim='time'):
    def detrend_1d(y):
        return signal.detrend(y, type='linear') if not np.all(np.isnan(y)) else y
    return xr.apply_ufunc(detrend_1d, data, input_core_dims=[[dim]], output_core_dims=[[dim]],
                         vectorize=True, dask='parallelized', output_dtypes=[float])

nh = detrend_dim(summer_monthly_anom.sel(latitude=slice(90, 0)))
coslat = np.cos(np.deg2rad(nh.latitude.values))
wgts_2d = xr.DataArray(np.sqrt(coslat)[:, None] * np.ones((1, nh.longitude.size)), 
                       coords={'latitude': nh.latitude, 'longitude': nh.longitude}, dims=('latitude', 'longitude'))

solver = Eof(nh.transpose('time', 'latitude', 'longitude'), weights=wgts_2d)
pc2 = solver.pcs(npcs=4, pcscaling=1).isel(mode=1)

cgt_vals, h_durations, h_sizes, h_intensities, categories = [], [], [], [], []

for i in range(len(events_ds.event)):
    try:
        t = events_ds.start_date.isel(event=i).values
        ym_str = str(np.datetime64(t, 'M'))
        cgt_val = float(pc2.sel(time=np.datetime64(t, 'M'), method='nearest').values)
    
        dur = float(events_ds.duration.isel(event=i).values) if 'duration' in events_ds else 1.0
        size = float(events_ds.max_extent_cells.isel(event=i).values) if 'max_extent_cells' in events_ds else 0.0
        inte = float(events_ds.max_intensity.isel(event=i).values) if 'max_intensity' in events_ds else 0.0

        cat = 'Positive CGT' if ym_str in POSITIVE_CGT_MONTHS else 'Negative CGT' if ym_str in NEGATIVE_CGT_MONTHS else 'Neutral'
        cgt_vals.append(cgt_val); h_durations.append(dur); h_sizes.append(size); h_intensities.append(inte); categories.append(cat)
    except: continue

data = {
    'CGT': np.array(cgt_vals),
    'Duration': np.array(h_durations),
    'Size': np.array(h_sizes),
    'Intensity': np.array(h_intensities),
    'Cat': np.array(categories)}

#quantifies how much hotter positive CGT heat waves are
pos_mask = data['Cat'] == 'Positive CGT'
neg_mask = data['Cat'] == 'Negative CGT'
neu_mask = data['Cat'] == 'Neutral'

pos_intensity = data['Intensity'][pos_mask]
neg_intensity = data['Intensity'][neg_mask]
neu_intensity = data['Intensity'][neu_mask]
all_intensity = data['Intensity']

pos_mean, neg_mean, neu_mean, all_mean = (pos_intensity.mean(), neg_intensity.mean(),
                                           neu_intensity.mean(), all_intensity.mean())

diff_pos_neg = pos_mean - neg_mean
diff_pos_all = pos_mean - all_mean
diff_pos_neu = pos_mean - neu_mean

t_stat_pn, p_val_pn = ttest_ind(pos_intensity, neg_intensity, equal_var=False)
t_stat_pa, p_val_pa = ttest_ind(pos_intensity, all_intensity, equal_var=False)

#Plotting figure
fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharex=True)
metrics = ['Duration', 'Size', 'Intensity']
y_labels = ['Duration (Days)', 'Max Size (Grid Points)', 'Max Intensity (°C)']
colors = {'Positive CGT': '#d62728', 'Negative CGT': '#1f77b4', 'Neutral': '#7f7f7f'}
panel_labels = ['a', 'b', 'c']

min_s, max_s = 30, 300
s_scale = min_s + (data['Duration'] - data['Duration'].min()) / (data['Duration'].max() - data['Duration'].min()) * (max_s - min_s)

for i, ax in enumerate(axes):
    m = metrics[i]
    ax.text(-0.05, 1.07, f"{panel_labels[i]})", transform=ax.transAxes, 
            fontsize=20, fontweight='bold', va='top', ha='right')
    for cat in ['Negative CGT', 'Neutral', 'Positive CGT']:
        mask = data['Cat'] == cat
        current_size = 35

        ax.scatter(
            data['CGT'][mask], 
            data[m][mask], 
            s=current_size,
            c=colors[cat], 
            alpha=0.6, 
            edgecolors='black', 
            linewidth=0.5, 
            label=f'{cat}' if i == 0 else "_")
   
    slope, intercept, r_val, p_val, _ = linregress(data['CGT'], data[m])
    x_range = np.linspace(data['CGT'].min(), data['CGT'].max(), 100)
    p_str = 'p < .001' if p_val < 0.001 else f'p = {p_val:.3f}'
    ax.plot(x_range, slope*x_range + intercept, 'k--', alpha=0.8, label=f'r={r_val:.2f}, {p_str}')
    
    ax.set_ylabel(y_labels[i], fontweight='bold', fontsize=18)
    ax.set_xlabel('CGT Index (PC2)', fontweight='bold', fontsize=18)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2)
    if i==0: ax.legend(loc='upper right', fontsize=16)
    if i==1: ax.legend(loc='upper right', fontsize=18)
    if i==2: ax.legend(loc='upper left', fontsize=18)

plt.suptitle('Relationship between CGT Index and Heat Wave Characteristics (1951-2024)', fontsize=30, y=1.02)
plt.tight_layout()
plt.savefig('Fig4_HeatWave_CGT_Scatterplots.png', dpi=300, bbox_inches='tight')
plt.show()
