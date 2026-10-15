# Magnitude to Photon Count

In this notebook we will use the [PySynPhot](https://pysynphot.readthedocs.io/en/latest/index.html) package to simulate the spectra of the two (brightest) stars in the Alpha Centauri system through the TOLIMAN bandpass. We will then calculate the photon count rate of the two stars.

We will use the PySynPhot built-in Phoenix models for the two stars, which are set of stellar atmosphere models that can be used to generate synthetic spectra. We use the known magnitudes of the two stars (taken from [SIMBAD](http://simbad.cds.unistra.fr/simbad/)) to renormalise the spectra, and then restrict this to the TOLIMAN bandpass of 530nm ~ 640nm. Finally we integrate the resultant spectrum to calculate the photon count rate.

## Setup
First we import the necessary packages and set the parameters of the two stars.


```python
import numpy as np
import pysynphot as S
import os
from pandas import read_csv
from scipy.interpolate import interp1d

# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams["figure.dpi"] = 120
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["legend.facecolor"] = "white"
mpl.rcParams["legend.framealpha"] = 1.0

ALPHA_CEN_A_SURFACE_TEMP = 5790.0  # K
ALPHA_CEN_A_METALICITY = 0.2
ALPHA_CEN_A_SURFACE_GRAV = 4.0  # log(g)
ALPHA_CEN_A_MAGNITUDE = 0.01  # vega magnitude

ALPHA_CEN_B_SURFACE_TEMP = 5260.0  # K
ALPHA_CEN_B_METALICITY = 0.23
ALPHA_CEN_B_SURFACE_GRAV = 4.37  # log(g)
ALPHA_CEN_B_MAGNITUDE = 1.33  # vega magnitude
```

    /Users/mc/miniconda3/envs/pysynphot/lib/python3.6/site-packages/pysynphot/locations.py:46: UserWarning: PYSYN_CDBS is undefined; functionality will be SEVERELY crippled.
      warnings.warn("PYSYN_CDBS is undefined; functionality will be SEVERELY "
    /Users/mc/miniconda3/envs/pysynphot/lib/python3.6/site-packages/pysynphot/locations.py:345: UserWarning: Extinction files not found in extinction
      warnings.warn('Extinction files not found in %s' % (extdir, ))


Next we have to do some housekeeping -- PySynPhot requires us to set the directory of some reference data. We need at a minimum the `mtab/` and `comp/nonhst/` directories which include the master tables and some other stuff, and the Phoenix stellar atlas. Unfortunately they are somewhat sizeable, but it is entirely possible not all of it is required for what we are doing. They are downloadable from the following links:

Master tables and Comp (~1MB and ~30MB): http://ssb.stsci.edu/trds/tarfiles/synphot1.tar.gz

Download and unzip, and just drag the `mtab/` and `comp/nonhst/` directories into your working directory. You can delete everything else to save space.

Phoenix stellar atlas (~3GB): http://ssb.stsci.edu/trds/tarfiles/synphot5.tar.gz

Do the same with the `grid/` directory. I've discovered only `grid/phoenix/phoenixm00/`, `grid/phoenix/phoenixp03/`, and `grid/phoenix/catalog.fits` are required for Alpha Centauri, so you can delete the rest to save space. However this is likely not the case for different stars.

We also have to set the area of the telescope -- I have used the TOLIMAN primary area of 12.5cm diameter and 3.2cm diameter hole.



```python
primary = (
    np.pi / 4 * (12.5**2 - 3.2**2)
)  # primary subtract secondary in cm^2; 12.5 and 3.2 are diameters
S.setref(area=primary)  # not sure that this actually does anyting tbh

print(f"TOLIMAN Primary Area: {primary:.2f} square cm.")
```

    TOLIMAN Primary Area: 114.68 square cm.


## Generating Spectra
Now we can generate the spectra of the two stars using the [Phoenix](https://pysynphot.readthedocs.io/en/stable/appendixa.html#pysynphot-appendixa-phoenix) models. We feed the models the surface temperature, metallicity and surface gravity of the stars, and then renormalise the spectra to the known magnitudes of the stars in the V band. We then convert the flux to units of photons $s^{-1}cm^{-2}\AA^{-1}$ and the wavelengths to $\AA$. Using the `fe_h` query in [SIMBAD](http://simbad.cds.unistra.fr/simbad/) can get you $\text{T}_\text{{eff}}$, $\log{g}$ and $\text{Fe}_\text{H}$ for different stars.


```python
A_sp = S.Icat(
    "phoenix",
    ALPHA_CEN_A_SURFACE_TEMP,
    ALPHA_CEN_A_METALICITY,
    ALPHA_CEN_A_SURFACE_GRAV,
)

B_sp = S.Icat(
    "phoenix",
    ALPHA_CEN_B_SURFACE_TEMP,
    ALPHA_CEN_B_METALICITY,
    ALPHA_CEN_B_SURFACE_GRAV,
)

# Renormalising to respective magnitudes
VBand = S.ObsBandpass("v")
A_sp = A_sp.renorm(RNval=ALPHA_CEN_A_MAGNITUDE, RNUnits="vegamag", band=VBand)
B_sp = B_sp.renorm(RNval=ALPHA_CEN_B_MAGNITUDE, RNUnits="vegamag", band=VBand)

# Converting units
A_sp.convert("Photlam")
B_sp.convert("Photlam")

sum_sp = A_sp.flux + B_sp.flux
```

We can now plot the spectra of the two stars and the sum of the two spectra, as a sanity check.


```python
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(A_sp.wave, A_sp.flux, label=r"$\alpha$ Cen A", alpha=0.5, lw=0.4)
ax.plot(B_sp.wave, B_sp.flux, label=r"$\alpha$ Cen B", alpha=0.5, lw=0.4)
ax.plot(A_sp.wave, sum_sp, label="Sum", c="k", lw=0.4)
ax.set(
    title=r"$\alpha$ Cen Spectrum",
    xlabel=f"Wavelength ({A_sp.waveunits})",
    ylabel=r"Flux (photons $s^{-1}cm^{-2}\AA^{-1}$)",
    xlim=(3000, 9000),
    xticks=np.linspace(3000, 9000, 13),
    ylim=(0, sum_sp.max() * 1.1),
)
# ax.grid(alpha=0.5)
ax.legend()
plt.show()
```


    
![png](assets/photon_count_files/photon_count_8_0.png)
    


## Creating Bandpass
Great. Now we can generate the bandpass filter for TOLIMAN. Here we consider the spectral throughput of the bandpass filter, the quantum efficiency of the detector, and an approximation of the throughput of the sheet of glass encasing the diffractive pupil. We will have to square the glass throughput as the beam passes through two sheets of glass.


```python
# bandpass, wavelength (nm) and throughput (%)
bp = read_csv("throughput/bandpass.csv")
bp_tp = bp.Throughput / 100
wavels = 10 * np.array(bp.Wavelength)  # angstrom, canonical wavelength grid

# approximate transmission factor of ONE sheet of glass
glass = 0.91  # this is a visual estimation from a piece of paper
glass_tp = glass * glass * np.ones_like(wavels)  # TWICE for two sheets

# quantum efficiency, wavelength (nm) and QE (%)
qe = read_csv("throughput/QE.csv")
# We want to interpolate the QE data to the bandpass data
interp_func = interp1d(
    10 * qe.Wavelength, qe.QE / 100, kind="linear", bounds_error=False, fill_value=0
)
qe_tp = interp_func(wavels)

# Combine the bandpass, glass and QE data
total_bp = bp_tp * glass_tp * qe_tp

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(wavels, bp_tp, label="Bandpass", linestyle="--", linewidth=1)
ax.plot(wavels, glass_tp, label="Glass x Glass", linestyle="--", linewidth=1)
ax.plot(wavels, qe_tp, label="Q. Efficiency", linestyle="--", linewidth=1)
ax.plot(wavels, total_bp, label="Total", linewidth=2)
ax.set(
    ylim=(0, 1.0),
    xlim=(4000, 8000),
    xlabel=f"Wavelength (Å)",
    ylabel="Throughput",
    title="TOLIMAN Throughput",
)
ax.legend(loc="upper right")
plt.show()
```

    /Users/mc/miniconda3/envs/pysynphot/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py:238: RuntimeWarning: Glyph 197 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /Users/mc/miniconda3/envs/pysynphot/lib/python3.6/site-packages/matplotlib/backends/backend_agg.py:201: RuntimeWarning: Glyph 197 missing from current font.
      font.set_text(s, 0, flags=flags)



    
![png](assets/photon_count_files/photon_count_10_1.png)
    


## Creating an Observation Object
Next we create a PySynPhot `Observation` object for each star, which combines the spectra and the bandpass filter we created.


```python
# creating pysynphot bandpass object
bandpass = S.ArrayBandpass(wavels, total_bp, waveunits="angstrom")

# creating observation objects
A_obs = S.Observation(A_sp, bandpass)
B_obs = S.Observation(B_sp, bandpass)
sum_obs = A_obs.flux + B_obs.flux
```

    (UnnamedArrayBandpass) does not have a defined binset in the wavecat table. The waveset of the spectrum will be used instead.
    (UnnamedArrayBandpass) does not have a defined binset in the wavecat table. The waveset of the spectrum will be used instead.


Plotting as a final sanity check.


```python
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(A_obs.wave, A_obs.flux, label=r"$\alpha$ Cen A", alpha=0.8, lw=0.4)
ax.plot(B_obs.wave, B_obs.flux, label=r"$\alpha$ Cen B", alpha=0.8, lw=0.4)
ax.plot(A_obs.wave, sum_obs, label="Sum", c="k", lw=0.4)
ax.set(
    title=r"$\alpha$ Cen spectrum through TOLIMAN bandpass",
    xlabel=f"Wavelength ({A_obs.waveunits})",
    ylabel=r"Flux (photons $s^{-1}cm^{-2}\AA^{-1}$)",
    xlim=(3000, 9000),
    xticks=np.linspace(3000, 9000, 13),
    ylim=(0, sum_obs.max() * 1.1),
)
ax.grid(alpha=0.2)
ax.legend()
plt.show()
```


    
![png](assets/photon_count_files/photon_count_14_0.png)
    


## Finding the Count Rate
Looks good! Now we can calculate the count rate using the `integrate` method of the `Observation` object. This sums the flux over the wavelength range of the filter, giving a flux value in units of photons per second per square centimetre. We can then multiply this by the primary mirror area to get the total count rate in photons per second. This will be the photon count rate at the pupil plane.


```python
countrate = A_obs.integrate() + B_obs.integrate()
print(
    f"Alpha Cen A: {A_obs.integrate():.3e} photons/s/cm/cm; {A_obs.integrate()*primary:.3e} photons/s"
)
print(
    f"Alpha Cen B: {B_obs.integrate():.3e} photons/s/cm/cm; {B_obs.integrate()*primary:.3e} photons/s"
)
print(
    f"Total count rate: {countrate:.3e} photons/s/cm/cm; {countrate*primary:.3e} photons/s"
)
```

    Alpha Cen A: 6.342e+05 photons/s/cm/cm; 7.272e+07 photons/s
    Alpha Cen B: 1.967e+05 photons/s/cm/cm; 2.256e+07 photons/s
    Total count rate: 8.309e+05 photons/s/cm/cm; 9.528e+07 photons/s


Great. If we know the TOLIMAN exposure time is 0.1 seconds, then we can calculate the number of photons present in a single frame.


```python
exp_time = 0.1  # seconds

tot_phot = countrate * primary * exp_time
print(f"Total photons in {exp_time} seconds: {tot_phot:.3e} photons")
print(f"Log10 of total photons: {np.log10(tot_phot):.3f}")
```

    Total photons in 0.1 seconds: 9.528e+06 photons
    Log10 of total photons: 6.979


## Final Plot


```python
xlim = (4000, 7500)

# --- Colour scheme (Okabe–Ito) ---
COL_A = "#0072B2"  # blue
COL_B = "#D55E00"  # vermillion
COL_SUM = "#000000"  # black
COL_BP = "#009E73"  # green
COL_GLASS = "#E69F00"  # orange
COL_QE = "#CC79A7"  # purple
COL_TOTAL = "#56B4E9"  # light blue

fig, (ax0, ax1, ax2) = plt.subplots(
    3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"hspace": 0.1}
)

ax0.plot(A_sp.wave, A_sp.flux, label=r"$\alpha$ Cen A", alpha=0.8, lw=0.4, color=COL_A)
ax0.plot(B_sp.wave, B_sp.flux, label=r"$\alpha$ Cen B", alpha=0.8, lw=0.4, color=COL_B)
ax0.plot(A_sp.wave, sum_sp, label=r"$A+B$", lw=0.6, color=COL_SUM)
ax0.set(
    ylabel=r"Flux (photons $s^{-1}cm^{-2}\AA^{-1}$)",
    xlim=xlim,
    ylim=(0, sum_sp.max() * 1.1),
)
ax0.grid(alpha=0.2)
ax0.legend(loc="upper right")

ax1.plot(wavels, bp_tp, label="Filter", linestyle="-", linewidth=1.5, color=COL_BP)
ax1.plot(
    wavels,
    glass_tp,
    label=r"Glass$\times$Glass",
    linestyle="-",
    linewidth=1.5,
    color=COL_GLASS,
)
ax1.plot(
    wavels, qe_tp, label="Q. Efficiency", linestyle="-", linewidth=1.5, color=COL_QE
)
ax1.fill_between(
    wavels, total_bp, label="Total", color=COL_TOTAL, alpha=0.7, edgecolor=None
)
ax1.set(
    ylim=(0, 1.0),
    xlim=xlim,
    ylabel="Throughput",
)
ax1.grid(alpha=0.2)
ax1.legend(loc="upper right")

ax2.plot(
    A_obs.wave, A_obs.flux, label=r"$\alpha$ Cen A", alpha=0.8, lw=0.4, color=COL_A
)
ax2.plot(
    B_obs.wave, B_obs.flux, label=r"$\alpha$ Cen B", alpha=0.8, lw=0.4, color=COL_B
)
ax2.plot(A_obs.wave, sum_obs, label=r"$A+B$", lw=0.6, color=COL_SUM)
ax2.set(
    xlabel=f"Wavelength ({A_obs.waveunits})",
    ylabel=r"Flux (photons $s^{-1}cm^{-2}\AA^{-1}$)",
    xlim=xlim,
    ylim=(0, sum_obs.max() * 1.1),
)
ax2.grid(alpha=0.2)
ax2.legend(loc="upper right")

bbox = {
    "facecolor": "white",
    "edgecolor": "black",
    "alpha": 1.0,
    "pad": 3,
}

ax0.text(
    0.18,
    0.5,
    r"$\alpha$ Cen Spectra",
    transform=ax0.transAxes,
    fontsize=13,
    fontweight="bold",
    bbox=bbox,
    verticalalignment="center",
    horizontalalignment="center",
)

ax1.text(
    0.18,
    0.5,
    "TOLIMAN Bandpass",
    transform=ax1.transAxes,
    fontsize=13,
    fontweight="bold",
    bbox=bbox,
    verticalalignment="center",
    horizontalalignment="center",
)

ax2.text(
    0.18,
    0.5,
    r"$\alpha$ Cen Spectra through" + "\n TOLIMAN Bandpass",
    transform=ax2.transAxes,
    verticalalignment="center",
    fontsize=13,
    fontweight="bold",
    horizontalalignment="center",
    bbox=bbox,
)
plt.tight_layout()
plt.savefig("figs/bandpass.pdf", dpi=500, bbox_inches="tight")
plt.savefig("figs/bandpass.png", dpi=500, bbox_inches="tight")
plt.show()
```

    /Users/mc/miniconda3/envs/pysynphot/lib/python3.6/site-packages/ipykernel_launcher.py:108: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.



    
![png](assets/photon_count_files/photon_count_20_1.png)
    

