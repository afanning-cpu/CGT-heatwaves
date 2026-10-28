#Linear QG Analysis and Fig 5 output

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

DATA_DIR = "1951_2024_Dailyavg_gz_u_v"
YEARS = range(1951, 2025)
A = 6.371e6 #earth radius (m)
OMEGA = 7.292e-5 #earth rotation rate (s^-1)
G = 9.80665 #gravity

REF_LAT = 45.0 #f0 = f(REF_LAT), used for the constant-f0 forcing and to convert streamfunction to Z
EPS = 1.0 / (5 * 86400) #rayleigh damping, 1/(5 days)
NU = 1.0e16 #del^4 hyperdiffusion (m^4/s); damps grid-scale noise
LAT_MIN, LAT_MAX = 5, 80 #solve domain
MMAX = 15 #keep zonal wavenumbers up to this value
DROP_ZONAL_MEAN = True 

USE_VBAR = False         #False = drops the mean meridional wind
USE_ZBAR_GRAD = False     #False = vorticity gradient is beta only 
ZONAL_MEAN_U = True     #True = use the zonal mean of u-bar instead of the 2D field
CONST_F0_FORCING = True   #True = forcing is -f0*div with constant f0

TAPER_DEG = 5.0 

#forcing region: Indian monsoon box
FORC_LAT_MIN, FORC_LAT_MAX = 5, 35
FORC_LON_MIN, FORC_LON_MAX = 60, 100

ISM_BOX = (5, 35, 60, 100)

DT = 1800.0 #time step (s)
TMAX_DAYS = 150 #max integration length (days)
TOL = 1e-5 #stop when relative change in psi per CHECK_DAYS drops below this
CHECK_DAYS = 1.0 #how often to check convergence (days)

#plotting
PLOT_LIM = 10 

POS_CGT_MONTHS = [
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

NEG_CGT_MONTHS = [
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

def load_monthly(varname, filename):
    files = [f"{DATA_DIR}/{y}/{filename}" for y in YEARS]
    ds = xr.open_mfdataset(files, combine="by_coords")
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    da = ds[varname].sel(pressure_level=200).sel(latitude=slice(90, -10))
    da = da.resample(time="MS").mean()                     #resamples from daily to monthly
    da = da.sel(time=da.time.dt.month.isin([6, 7, 8]))
    return da.coarsen(latitude=8, longitude=8, boundary="trim").mean().load()  #coarsen to 2 degrees

def spherical_divergence(u, v):
    #div(V) = 1/(a cos(phi)) * [du/dlambda + d(v cos(phi))/dphi], periodic in longitude.
    coslat = np.cos(np.deg2rad(u.latitude))
    dlon = np.deg2rad(float(u.longitude[1] - u.longitude[0]))
    dudlam = (u.roll(longitude=-1, roll_coords=False)
              - u.roll(longitude=1, roll_coords=False)) / (2 * dlon)
    dvcos_dphi = (v * coslat).differentiate("latitude") * (180 / np.pi)
    return (dudlam + dvcos_dphi) / (A * coslat)

def in_months(da, months):
    return da.time.dt.strftime("%Y-%m").isin(months)

def region_mask(da, lat_min, lat_max, lon_min, lon_max, taper=TAPER_DEG):
    def ramp(x, lo, hi):
        if taper <= 0:
            return ((x >= lo) & (x <= hi)).astype(float)
        w = np.clip(np.minimum(x - lo, hi - x) / taper, 0, 1)
        return 0.5 * (1 - np.cos(np.pi * w))
    lat = da.latitude
    lon = da.longitude % 360
    return ramp(lat, lat_min, lat_max) * ramp(lon, lon_min, lon_max)

class BarotropicModel:
    def __init__(self, Ubar, Vbar, S, eps=EPS, nu=NU, mmax=MMAX, drop_zonal_mean=DROP_ZONAL_MEAN,
                 use_vbar=USE_VBAR, use_zbar_grad=USE_ZBAR_GRAD, zonal_mean_u=ZONAL_MEAN_U):

        self.lats = Ubar.latitude.values
        self.lons = Ubar.longitude.values
        self.N = self.lons.size
        self.phi = np.deg2rad(self.lats)
        self.dphi = self.phi[1] - self.phi[0]
        self.c = np.cos(self.phi)[:, None]
        lat = self.lats[:, None]
        tropical_weight = 0.5 * (1 - np.tanh((np.abs(lat) - 15.0) / 4.0))

        eps_tropics = 1.0 / (1 * 86400) 
        self.eps = eps + (eps_tropics - eps) * tropical_weight
        self.nu = nu        
        self.mmax = mmax

        self.m = np.arange(self.N // 2 + 1)
        self.keep = self.m <= mmax
        if drop_zonal_mean:
            self.keep &= self.m >= 1
        self.M1 = mmax + 1

        self._build_laplacian()

        #basic state
        U = self.trunc(Ubar.values, keep_mean=True)
        V = self.trunc(Vbar.values, keep_mean=True)
        if zonal_mean_u:
            U = np.repeat(U.mean(axis=1, keepdims=True), self.N, axis=1)
        if not use_vbar:
            V = np.zeros_like(V)
        self.U, self.V = U, V

        if use_zbar_grad:
            zbar = (self.dlam(V) - self.dphi_(U * self.c)) / (A * self.c)
            self.qbar_lam = self.dlam(zbar) #d(f + zetabar)/dlambda
            self.qbar_phi = 2 * OMEGA * self.c + self.dphi_(zbar) #d(f + zetabar)/dphi
        else:
            #planetary vorticity only
            self.qbar_lam = np.zeros_like(U)
            self.qbar_phi = 2 * OMEGA * self.c * np.ones_like(U)

        Sh = self.rfft(S.values) * self.keep
        Sh[0], Sh[-1] = 0, 0
        self.S = self.irfft(Sh)

    def rfft(self, x):
        return np.fft.rfft(x, axis=1)

    def irfft(self, xh):
        return np.fft.irfft(xh, n=self.N, axis=1)

    def trunc(self, x, keep_mean=False):
        k = self.m <= self.mmax
        if not keep_mean:
            k = self.keep
        return self.irfft(self.rfft(x) * k)

    def dlam(self, x):
        return self.irfft(1j * self.m * self.rfft(x))

    def dphi_(self, x):
        return np.gradient(x, self.phi, axis=0)

    def _build_laplacian(self):
        #Laplacian for each zonal wavenumber m
        J = self.lats.size
        c = np.cos(self.phi)
        ch = np.cos(0.5 * (self.phi[1:] + self.phi[:-1])) 
        d2 = A ** 2 * self.dphi ** 2
        Lm = np.zeros((self.M1, J, J))
        for m in range(self.M1):
            for j in range(1, J - 1):
                Lm[m, j, j - 1] = ch[j - 1] / (d2 * c[j])
                Lm[m, j, j + 1] = ch[j] / (d2 * c[j])
                Lm[m, j, j] = -(ch[j - 1] + ch[j]) / (d2 * c[j]) - m ** 2 / (A ** 2 * c[j] ** 2)
        self.Lm = Lm
        self.Linv = np.linalg.inv(Lm[:, 1:-1, 1:-1]) 

    def lap(self, xh):
        out = np.zeros_like(xh)
        out[:, :self.M1] = np.einsum("mij,jm->im", self.Lm, xh[:, :self.M1])
        return out

    def inv_lap(self, zh):
        psih = np.zeros_like(zh)
        psih[1:-1, :self.M1] = np.einsum("mij,jm->im", self.Linv, zh[1:-1, :self.M1])
        return psih

    #model
    def tendency(self, zh):
        zh = zh * self.keep
        psih = self.inv_lap(zh)
        up = -self.dphi_(self.irfft(psih)) / A
        vp = self.irfft(1j * self.m * psih) / (A * self.c)

        z = self.irfft(zh)
        z_lam = self.irfft(1j * self.m * zh)
        z_phi = self.dphi_(z)

        adv_basic = self.U * z_lam / (A * self.c) + self.V * z_phi / A 
        adv_pert = up * self.qbar_lam / (A * self.c) + vp * self.qbar_phi / A 
        del4 = self.irfft(self.lap(self.lap(zh)))

        R = self.S - adv_basic - adv_pert - self.eps * z - self.nu * del4
        Rh = self.rfft(R) * self.keep
        Rh[0], Rh[-1] = 0, 0 #walls
        return Rh

    def zeta_from_uv(self, u0, v0):
        u0, v0 = np.asarray(u0, float), np.asarray(v0, float)
        return (self.dlam(v0) - self.dphi_(u0 * self.c)) / (A * self.c)

    #integrate forward to steady state
    def integrate(self, u0=None, v0=None, dt=DT, tmax_days=TMAX_DAYS, tol=TOL, check_days=CHECK_DAYS, verbose=True):
        J = self.lats.size
        if u0 is None and v0 is None:
            zh = np.zeros((J, self.m.size), dtype=complex)
        else:
            u0 = np.zeros((J, self.N)) if u0 is None else u0
            v0 = np.zeros((J, self.N)) if v0 is None else v0
            zh = self.rfft(self.zeta_from_uv(u0, v0)) * self.keep
            zh[0], zh[-1] = 0, 0

        nsteps = int(round(tmax_days * 86400 / dt))
        ncheck = max(1, int(round(check_days * 86400 / dt)))
        psi_prev = self.inv_lap(zh)
        hist = {"day": [], "rel_change": [], "rms_Z": []}
        change = np.nan
        f0 = 2 * OMEGA * np.sin(np.deg2rad(REF_LAT))

        for n in range(1, nsteps + 1):
            a1 = self.tendency(zh)
            a2 = self.tendency(zh + 0.5 * dt * a1)
            a3 = self.tendency(zh + 0.5 * dt * a2)
            a4 = self.tendency(zh + dt * a3)
            zh = zh + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)

            if n % ncheck == 0:
                psih = self.inv_lap(zh)
                change = np.linalg.norm(psih - psi_prev) / max(np.linalg.norm(psih), 1e-30)
                day = n * dt / 86400
                Zg = f0 / G * self.irfft(psih)
                hist["day"].append(day)
                hist["rel_change"].append(change)
                hist["rms_Z"].append(np.sqrt(np.mean(Zg ** 2)))
                if verbose and (len(hist["day"]) % 10 == 0):
                    print(f"day {day:6.1f}  rel change {change:.2e}  rms Z {hist['rms_Z'][-1]:.2f} m")
                if not np.isfinite(change):
                    raise RuntimeError(f"Blew up at day {day:.1f}; try a smaller DT or larger NU.")
                if change < tol:
                    if verbose:
                        print(f"Steady after {day:.1f} days (rel change {change:.2e})")
                    break
                psi_prev = psih.copy()
        else:
            if verbose:
                print(f"Not steady after {tmax_days} days (last rel change {change:.2e})")

        psih = self.inv_lap(zh)
        psi = self.irfft(psih)
        up = -self.dphi_(psi) / A
        vp = self.irfft(1j * self.m * psih) / (A * self.c)
        coords = {"latitude": self.lats, "longitude": self.lons}
        dims = ("latitude", "longitude")
        out = {
            "psi": xr.DataArray(psi, coords=coords, dims=dims, name="psi"),
            "Z": xr.DataArray(f0 / G * psi, coords=coords, dims=dims, name="Z"), #Z ~ f0 psi / g
            "u": xr.DataArray(up, coords=coords, dims=dims, name="u"),
            "v": xr.DataArray(vp, coords=coords, dims=dims, name="v"),
            "zeta": xr.DataArray(self.irfft(zh), coords=coords, dims=dims, name="zeta"),
        }
        hist = {key: np.array(val) for key, val in hist.items()}
        return out, hist

if __name__ == "__main__":
    u = load_monthly("u", "u_component_of_wind_0_daily-mean.nc")
    v = load_monthly("v", "v_component_of_wind_0_daily-mean.nc")

    pos, neg = in_months(u, POS_CGT_MONTHS), in_months(u, NEG_CGT_MONTHS)

    #basic states for each CGT phase
    Ubar_pos, Vbar_pos = u.sel(time=pos).mean("time"), v.sel(time=pos).mean("time")
    Ubar_neg, Vbar_neg = u.sel(time=neg).mean("time"), v.sel(time=neg).mean("time")

    #forcing: phase-specific divergence anomaly composites (anoms vs. 1951-2024 monthly clim)
    div = spherical_divergence(u, v)
    div_anom = div.groupby("time.month") - div.groupby("time.month").mean("time")

    div_pos = div_anom.sel(time=pos).mean("time")
    div_neg = div_anom.sel(time=neg).mean("time")

    #keep the divergence anomaly only over the Indian monsoon region
    mask = region_mask(div_pos, FORC_LAT_MIN, FORC_LAT_MAX, FORC_LON_MIN, FORC_LON_MAX)

    div_pos = (div_pos * mask).transpose("latitude", "longitude")
    div_neg = (div_neg * mask).transpose("latitude", "longitude")

    if CONST_F0_FORCING:
        f_forc = 2 * OMEGA * np.sin(np.deg2rad(REF_LAT))
    else:
        f_forc = 2 * OMEGA * np.sin(np.deg2rad(div_pos.latitude))

    S_pos = -f_forc * div_pos
    S_neg = -f_forc * div_neg

    #solve domain, latitude ascending
    dom = dict(latitude=slice(LAT_MAX, LAT_MIN))
    Ubar_pos, Vbar_pos, Ubar_neg, Vbar_neg, S_pos, S_neg, div_pos, div_neg = (
        x.sel(**dom).sortby("latitude").transpose("latitude", "longitude")
        for x in (Ubar_pos, Vbar_pos, Ubar_neg, Vbar_neg, S_pos, S_neg, div_pos, div_neg))

    u0_pos, v0_pos = None, None
    u0_neg, v0_neg = None, None

    print("Positive CGT basic state:")
    out_pos, hist_pos = BarotropicModel(Ubar_pos, Vbar_pos, S_pos).integrate(u0=u0_pos, v0=v0_pos)
    print("Negative CGT basic state:")
    out_neg, hist_neg = BarotropicModel(Ubar_neg, Vbar_neg, S_neg).integrate(u0=u0_neg, v0=v0_neg)

    Z_pos, Z_neg = out_pos["Z"], out_neg["Z"]
    Z_diff = Z_pos - Z_neg

    #convergence plot
    figc, axc = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for hist, lab in ((hist_pos, "Positive CGT"), (hist_neg, "Negative CGT")):
        axc[0].semilogy(hist["day"], hist["rel_change"], label=lab)
        axc[1].plot(hist["day"], hist["rms_Z"], label=lab)
    axc[0].axhline(TOL, color="k", ls="--", lw=0.8)
    axc[0].set_xlabel("day"); axc[0].set_ylabel("relative change in psi per check")
    axc[1].set_xlabel("day"); axc[1].set_ylabel("rms Z (m)")
    axc[0].legend(); axc[1].legend()
    #figc.savefig("qg_cgt_convergence_simple.png", dpi=200)

    #plot
    proj = ccrs.PlateCarree(central_longitude=180)
    pc = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), subplot_kw={"projection": proj}, constrained_layout=True)
    fig.subplots_adjust(top=0.92, hspace=0.15)

    def panel(ax, da, title, units, lim):
        data, lon = add_cyclic_point(da.values, coord=da.longitude.values)
        lev = np.linspace(-lim, lim, 21)
        negative_data = np.ma.masked_greater(data, 0)
        positive_data = np.ma.masked_less(data, 0)
        ax.contourf(lon, da.latitude, negative_data, levels=lev, cmap="RdBu_r",
                    extend="both", transform=pc)
        cf = ax.contourf(lon, da.latitude, positive_data, levels=lev, cmap="RdBu_r",
                         extend="both", transform=pc)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_extent([0, 359.9, LAT_MIN, LAT_MAX], crs=pc)
        gl = ax.gridlines(crs=pc, draw_labels=True, xlocs=np.arange(-180, 180, 60), ylocs=np.arange(0, 81, 20), linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}
        ax.set_title(title, fontsize=12)
        plt.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.7, pad=0.1, label=units)

    contour_fields = (div_pos, div_neg, div_pos - div_neg)
    contour_levels = np.arange(-0.7, 0.51, 0.2)  #units of 1e-6 s^-1

    for ax, div_da in zip(axes, contour_fields):
        div_values, lon = add_cyclic_point(
            div_da.values * 1e6,
            coord=div_da.longitude.values)
        cs = ax.contour(
            lon, div_da.latitude, div_values,
            levels=contour_levels,
            colors="#9156A7", linewidths=0.7,
            transform=pc)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%g")

    panel(axes[0], Z_pos, "Positive CGT: Z Response", "m", 8)
    panel(axes[1], Z_neg, "Negative CGT: Z Response", "m", 8)

    for ax in axes:
           ax.plot(
               [FORC_LON_MIN, FORC_LON_MAX, FORC_LON_MAX,
               FORC_LON_MIN, FORC_LON_MIN],
               [FORC_LAT_MIN, FORC_LAT_MIN, FORC_LAT_MAX,
               FORC_LAT_MAX, FORC_LAT_MIN],
               color="green", lw=1.2, transform=pc)
    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.text(0.02, 1.22, label, transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        zorder=10,)

    plt.savefig("Fig5_qg_cgt_response_simple_ism.png", dpi=200)
    plt.show()
