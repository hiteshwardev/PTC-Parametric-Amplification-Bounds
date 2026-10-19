# Limits on Parametric Amplification in Photonic Time Crystals: A Sharp Waveform Bound and Loss Thresholds in Dispersive Media

**Author:** Hitesh Kumar Singh · [ORCID: 0009-0008-1583-4848](https://orcid.org/0009-0008-1583-4848)
*Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India*

---

## Abstract

Photonic time crystals amplify light inside momentum gaps, but how strongly they amplify in realistic media is disputed: energy arguments predict weak gain at optical frequencies, while resonance-engineered designs report strongly widened momentum gaps, in one case without bound. This work separates the two quantities at stake, gap width and growth rate. For a nondispersive medium with arbitrary periodic permittivity, the Floquet amplitude exponent is proven to obey μT ≤ TV<sub>T</sub>[ln ε]/4, where TV<sub>T</sub> is the total variation over one period, and the logarithm of the period map's energy norm obeys the same bound. Sudden two-level switching attains the bound exactly, and harmonic modulation reaches π/4 of it. Over 540 test waveforms the bound is never violated; optimized smooth switching reaches 0.9996 of it. For a lossy Lorentz medium with a modulated resonance or plasma frequency, closed-form participation factors for gain and loss predict the exact Floquet exponents with median errors of 0.006% and 0.002% on the lower and upper polariton branches, cap the gain at its bare parametric value in every medium, and give net-gain thresholds that are independent of detuning for resonance modulation but diverge at resonance for plasma modulation. Along a path toward resonance the gap widens 13.0 ± 1.2-fold while the net exponent turns negative: gap width and amplification rate are independent design targets.

## Core Objectives

- Establish a bound on the parametric growth rate of a photonic time crystal that holds over a general class of periodic drives, rather than for a single waveform.
- Determine how material dispersion and loss limit the gain and set net-gain thresholds in a Lorentz medium.
- Test whether widening the momentum gap, as in resonance-engineered designs, also increases the net amplification rate.

## Scientific Significance

The literature on photonic time crystals contains two lines that appear to conflict: cautionary energy and dispersion arguments, and designs that widen the momentum gap at modest modulation strength. This project shows that the two concern different quantities. Gap width counts unstable wavevectors; the exponent measures how fast they grow. A sharp, waveform-independent bound gives a figure of merit (saturation of the bound) that separates waveform efficiency from raw modulation strength, and closed-form participation factors rank materials and modulation channels by gain and loss.

## Methodological Approach

- **Analytical:** an energy-angle identity for the Hill equation gives the total-variation bound for every initial state and every period; attainability is proven by an explicit bang-bang protocol. A conserved quadratic form, an exact energy balance and a two-mode projection at each tongue centre give the participation factors for the Lorentz medium.
- **Exact numerics:** Floquet monodromy matrices from an eighth-order Runge-Kutta integrator with step caps, audited by determinant identities, with the total variation evaluated exactly.
- **Waveform ensembles:** 400 log-Fourier, two-level and sinusoidal profiles (24,960 monodromies), a post hoc extended ensemble of 124 profiles, and optimization of two-switch waveforms.
- **Dispersive scans:** 32 parameter sets, both modulation channels and both polariton branches, with a four-mode secular theory developed after the registered scan revealed failures at interbranch crossings.
- **Protocol:** hypotheses, tolerances and benchmarks were fixed in a written protocol before the production runs; one criterion failed as stated and is reported with its resolution.

## Contributions

1. A **sharp total-variation bound** on the Floquet exponent and on the per-period growth of the energy norm for arbitrary periodic permittivity, attained exactly by sudden two-level switching.
2. A quantitative **harmonic efficiency** of π/4 and a saturation metric that allows reported growth rates to be compared across waveforms.
3. **Closed-form participation factors** for gain and loss in a lossy Lorentz medium, which cap the gain at its bare parametric value in any medium and give detuning-independent thresholds for resonance-frequency modulation and thresholds that diverge at resonance for plasma-frequency modulation.
4. A **four-mode theory** that explains the registered failures at interbranch crossings.
5. A demonstration along a path toward resonance that **gap width and net rate decouple**.
6. **Positioning relative to prior work.** Exact Hill-equation solutions for step-varying permittivity (e.g. *IEEE Trans. Antennas Propag.* 66, 5300, 2018), energy and dispersion constraints (Hayran, Khurgin and Monticone) and gap-engineering proposals are established.

## Key Findings

- The bound was never violated in 24,960 registered and 4,960 extended monodromies; the largest saturation is 0.9968 (exponent) and 0.9987 (energy norm) in the extended ensemble.
- Exact sudden switching attains the bound to round-off; optimized smooth switching reaches 0.9996 at ramp width 0.005 T, with the deficit scaling quadratically in the ramp width.
- Harmonic modulation stays within 0.058% of π/4 up to modulation depth 0.6.
- The two-mode prediction reproduces the exact exponent with median errors of 0.006% (lower branch, outside the crossings) and 0.002% (upper branch); the gain ceiling held in every tested medium with a smallest margin of 2.0 × 10⁻⁵ ω<sub>r0</sub>.
- One registered accuracy criterion failed at 19 of 1,842 points, all within a narrow window of the two interbranch crossings; the four-mode theory reproduces the exact exponents there to within 4.5 × 10⁻⁵ ω<sub>r0</sub>.
- Along the resonance path the momentum gap widens 13.0 ± 1.2-fold while the net exponent of the tongue pair falls from +1.83 × 10⁻³ ω<sub>r0</sub> to −4.65 × 10⁻³ ω<sub>r0</sub>.

## Scope and Limitations

The work is analytical and computational; no experiment was performed. The nondispersive bound constrains the Floquet exponent and per-period norm growth, not transient growth within stable bands, and it is tight only near parametric resonances. The dispersive results are derived at first order in the modulation depth and loss, verified up to m = 0.1 and γ = 0.03 ω<sub>r0</sub>, for a single Lorentz oscillator; multi-resonance, Drude, active and nonlocal media lie outside the derivation. The four-mode theory is a post hoc extension verified on two crossings of one medium.

## Roles

Hitesh Kumar Singh is the sole author. Hitesh Kumar Singh conceived the study, derived the bound and the participation factors, wrote the code, designed the protocol and benchmarks, carried out the simulations and analysis, prepared the figures and wrote the manuscript.

## Reproducibility and Availability

The repository contains the code, the executable notebook that reproduces every number, table and figure of the manuscript, and the raw results.
