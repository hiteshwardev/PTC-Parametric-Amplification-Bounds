# PTC-Parametric-Amplification-Bounds

Code and complete computational record for the article *Limits on parametric amplification in photonic time crystals: a sharp waveform bound and loss thresholds in dispersive media* by Hitesh Kumar Singh.

The study has two parts.

1. **Nondispersive media.** For a spatially uniform medium with arbitrary periodic permittivity $\varepsilon(t)$, the Floquet amplitude exponent obeys $\mu T \le \mathrm{TV}_T[\ln\varepsilon]/4$, where $\mathrm{TV}_T$ is the total variation over one period, and the logarithm of the energy norm of the period map obeys the same bound. Sudden two-level switching attains the bound; harmonic modulation reaches $\pi/4$ of it.
2. **Lossy Lorentz media.** For a single Lorentz oscillator with a modulated resonance or plasma frequency, a two-mode reduction gives closed-form participation factors for the gain and loss of each first instability tongue, a ceiling on the gain that holds in every medium, and net-gain thresholds for the two channels. Interbranch crossings, where the reduction fails, are described by a four-mode secular theory.

Every number, table entry and figure of the article is produced by `notebook.ipynb`.

## Repository layout

| Path | Content |
|---|---|
| `notebook.ipynb` | The whole study, executed top to bottom: theory, study protocol, solver validation, all computations, figures and the summary of results |
| `src/hill.py` | Hill equation for nondispersive media: waveforms, exact total variation, monodromy, Floquet exponents, energy norm, exact sudden switching |
| `src/lorentz.py` | Lossy Lorentz medium: polariton branches, participation factors, thresholds, Floquet monodromy, tongue-pair selection, integrated growth inequality, four-mode secular matrix |
| `src/figstyle.py` | Figure style (Okabe-Ito palette with distinct markers and line styles), legends below the axes, automatic layout check, PDF and 600 dpi PNG export |
| `tests/` | Unit tests of the three modules |
| `results/` | JSON output of every computation; `summary.json` collects the quantities quoted in the article |
| `figures/` | The ten figures of the article, as PDF and PNG |

## Installation

Python 3.13 with the pinned packages of `requirements.txt`:

```
pip install -r requirements.txt
```

or, with conda,

```
conda env create -f environment.yml
conda activate ptc-bounds
```

## Reproducing the results

From the repository root,

```
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

regenerates `results/` and `figures/` from scratch; nothing is read from earlier runs. A full run takes 20 to 30 minutes on one CPU core. All random draws are seeded, so repeated runs give identical results. To work interactively, open the notebook in JupyterLab (`pip install jupyterlab`) and run all cells.

The unit tests take a few seconds:

```
python -m pytest
```

## Conventions

Units with $c = \varepsilon_0 = \mu_0 = 1$. In the nondispersive part the modulation frequency is $\Omega = 1$, so the period is $T = 2\pi$. In the dispersive part frequencies, rates and $ck$ are given in units of the static resonance frequency $\omega_{r0}$; the reference medium has $\omega_{p0}/\omega_{r0} = 0.8$. The modulated parameter is either $\omega_r^2$ (resonance channel, `"wr2"` in the code) or $\omega_p^2$ (plasma channel, `"wp2"`).

## License

MIT; see `LICENSE`.

## Citation

If you use this code, please cite the article and the repository:

H. K. Singh, PTC-Parametric-Amplification-Bounds, https://github.com/hiteshwardev/PTC-Parametric-Amplification-Bounds (2026).
