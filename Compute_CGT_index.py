"""
Compute the circumglobal teleconnection (CGT) index and save it to file.

The CGT is defined as the second EOF / PC of detrended JJA monthly 200 hPa
geopotential height anomalies over the Northern Hemisphere (0-90N), using
sqrt(cos(lat)) area weights. The PC is scaled to unit variance (pcscaling=1).

Outputs:
  cgt_index_1951_2024.nc   monthly JJA CGT index (netCDF)
  cgt_index_1951_2024.csv  same index as a CSV (date, year, month, cgt_index)

Data: ERA5 monthly 200 hPa geopotential.
"""

import numpy as np
import xarray as xr
from eofs.xarray import Eof
from scipy import signal

GEOPOTENTIAL_FILE = "MonthlyGeopotential1941to2024.nc"
START_YEAR, END_YEAR = 1951, 2024
JJA = [6, 7, 8]
N_EOFS = 4
OUTPUT_NC = f"cgt_index_{START_YEAR}_{END_YEAR}.nc"
OUTPUT_CSV = f"cgt_index_{START_YEAR}_{END_YEAR}.csv"

def detrend_dim(data, dim):
    #Linearly detrend along `dim`
    def _detrend(y):
        if np.all(np.isnan(y)):
            return y
        return signal.detrend(y, type="linear")

    return xr.apply_ufunc(
        _detrend, data,
        input_core_dims=[[dim]], output_core_dims=[[dim]],
        vectorize=True, dask="parallelized", output_dtypes=[float], )

def compute_cgt_index():
    ds = xr.open_dataset(GEOPOTENTIAL_FILE)
    z200 = ds["z"].squeeze("pressure_level", drop=True)
    z200 = z200.sel(valid_time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    z200 = z200 / 9.80665  #geopotential -> geopotential height(m)

    #JJA monthly anomalies relative to each calendar month's climatology, then detrend
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
    pc2 = solver.pcs(npcs=N_EOFS, pcscaling=1).isel(mode=1, drop=True)
    pc2["time"] = nh.time
    return pc2

def main():
    cgt = compute_cgt_index()
    cgt = cgt.rename("cgt_index")
    cgt.attrs = {
        "long_name": "Circumglobal teleconnection (CGT) index",
        "description": ("PC2 of detrended JJA monthly 200 hPa geopotential height "
                        "anomalies, 0-90N, sqrt(cos(lat)) weighted; unit variance"),
        "units": "standardized",
        "source": "ERA5 monthly 200 hPa geopotential",}
    cgt.to_netcdf(OUTPUT_NC)

    df = cgt.to_dataframe().reset_index().rename(columns={"time": "date"})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df[["date", "year", "month", "cgt_index"]].to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()
