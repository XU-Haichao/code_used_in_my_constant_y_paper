# 5 eV Blackbody Slab Validation

## Setup

- HEASoft 6.36, XSPEC 12.15.1, PyXspec 2.1.5
- Reference model: built-in `compps`
- Candidate model: local `compPSc*bbodyrad`
- `kTbb = 0.005 keV`, `geom = 1`, `cosIncl = 0.5`, no reflection
- Common logarithmic grid: 0.001--1000 keV, 1200 bins
- Comparison band: bins centered at or above 0.002 keV and above `1e-8` of each spectrum's peak
- One constant candidate-to-reference scale is removed before measuring shape residuals

Run from the repository root:

```bash
conda run -n heasoft_full python compps_conv/validate_bb5ev_slab.py
```

The command exits with status 1 when one or more cases fail the agreed acceptance criteria. This is expected for the current results.

## Results

| kTe (keV) | tau | Candidate scale | Median difference | 95th percentile | Result |
|---:|---:|---:|---:|---:|:---:|
| 51.1 | 0.1 | 0.975120 | 0.808% | 2.964% | PASS |
| 51.1 | 1.0 | 1.051620 | 4.485% | 6.097% | FAIL |
| 255.5 | 0.1 | 0.990253 | 0.123% | 1.534% | PASS |
| 255.5 | 1.0 | 1.014104 | 1.298% | 2.330% | FAIL |

The accepted limits are median difference below 1% and 95th-percentile difference below 3%.

## Interpretation

The two models agree to the requested accuracy at `tau = 0.1`. They do not both satisfy the requested accuracy at `tau = 1`: the residual is smooth with energy and therefore is not explained by isolated numerical outliers.

Increasing the external blackbody grid from 600 to 4800 bins did not change the `kTe = 51.1 keV`, `tau = 1` discrepancy. Extending the seed-input grid from `1e-3` down to `1e-6 keV` also did not change it. A temporary `MAXTAU = 20` diagnostic did not reproduce the built-in result, and the production source and library were restored to `MAXTAU = 60`. A `60/59` optical-depth rescaling improves the `tau = 1` cases but degrades both `tau = 0.1` cases, so it is not a valid general conversion.

The current evidence therefore supports agreement in the optically thin regime but not strict same-parameter equivalence across the full requested `tau <= 1` range. The next diagnostic should compare the convolution routine against the exact HEASoft 6.36 `compps` source with matched compile-time grids and an identical tabulated blackbody input.

## Artifacts

- `validation_bb5ev_slab/summary.csv`: case-level metrics
- `validation_bb5ev_slab/spectra.csv`: bin-level spectra and residuals
- `validation_bb5ev_slab/compps_bb5ev_slab_comparison.png`: visual comparison
