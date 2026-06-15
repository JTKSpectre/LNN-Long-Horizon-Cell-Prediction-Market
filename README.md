# LNN-Long-Horizon-Cell-Prediction-Market

A Liquid Neural Network cell built for long-horizon forecasting, tested on
synthetic market-like data. It runs three independent continuous-time hidden
states — fast, mid, and slow — each governed by its own Liquid Time-Constant
(LTC) ODE and advancing on its own update cadence (every 1 / 5 / 20 steps),
so the cell tracks short-term swings and long-horizon drift in the same pass.

This repo doesn't just ship the architecture — it characterizes where it
breaks. Under a naive explicit-Euler integration, a high-variance input
regime drives the **slow (long-horizon) state** into unbounded divergence
while the fast and mid states remain bounded: the instability lives exactly
at the long-horizon end. The fused semi-implicit step from the LTC
literature stabilizes it unconditionally. Both behaviors are reproduced by
one script, with the figures to show it.

## Results

- **Divergence demo** (`figures/norm_trace.png`): naive explicit Euler — the
  slow state's norm departs to ~10^10 when the variance shock hits, after a
  settled plateau (so the blowup is the input regime, not the initial
  transient). Fused semi-implicit — bounded throughout, and the cell
  self-reports warm-up completion (`is_warm`) before scoring begins.
- **Long-horizon frontier** (`figures/horizon_frontier.png`): the stabilized
  cell beats the persistence baseline (y_hat = current value) at every tested
  horizon on held-out post-shock data, out to 100 steps.

| Horizon (steps) | HorizonCell RMSE | Persistence baseline |
|---|---|---|
| 5   | 0.108 | 0.130 |
| 10  | 0.172 | 0.225 |
| 20  | 0.333 | 0.410 |
| 50  | 0.595 | 0.713 |
| 100 | 0.609 | 0.864 |

## What's in here

- `horizon_cell.py` — the cell. Three `LTCSubState`s (own tau, own weights,
  own cadence; off-tick states hold; on tick they consume the *mean* of
  inputs since their last tick at dt_eff = dt x tick). Reservoir-style
  training: recurrent weights fixed, readout fit by closed-form ridge
  regression. Numpy only.
- `synthetic_market.py` — drift + cycles + regime-switched, AR(1)-clustered
  volatility. Ground truth is known by construction, so "the cell recovers
  injected structure" is a measurable claim.
- `run_horizon.py` — one command, both figures: `python run_horizon.py`

## Honest scope

This is a dynamical-systems study on synthetic data with known structure.
It demonstrates that a multi-timescale LTC recovers injected drift and cycle
structure over long horizons, and precisely where the naive version fails.
It makes **no claim of forecasting skill on real markets.** Robust
long-horizon stability under severe regime shift — without sacrificing the
long memory that makes the cell useful — remains an open tradeoff, discussed
in [FAILURE_MODES.md](FAILURE_MODES.md).

## Relationship to the original work

This repository began as a fork of
[HusseinJammal/Liquid-Neural-Networks-in-Stock-Market-Prediction](https://github.com/HusseinJammal/Liquid-Neural-Networks-in-Stock-Market-Prediction)
(Apache-2.0), which predicts TSLA/AAPL adjusted-close prices from Yahoo
Finance data using a single-cell recurrent model and a web front-end. Credit
to the original author for the starting point. This is an independent
reimplementation; it diverges substantially in goal, data, and architecture:

| Aspect | Original | This repository |
|---|---|---|
| Goal | Maximize next-day price accuracy | Characterize where a multi-timescale LTC holds and fails at long horizons |
| Data | Live Yahoo Finance (TSLA/AAPL) | Synthetic, known-structure (drift + cycle + regime-switched volatility) |
| Cell | Single recurrent cell (tanh step) | Three independent LTC states (fast/mid/slow), staggered cadence, fused semi-implicit ODE |
| Training | Pretrained Keras model loaded at inference | Reservoir-style: fixed recurrent weights, closed-form ridge readout |
| Stack | Python + Node/JS web app + .h5 | numpy + matplotlib, single script |
| Claim | Predictive accuracy on real stocks | No real-market claim; recovers injected structure, documents the instability |

The deliberate departure is the last row. This repo does not claim to forecast
markets. It studies the dynamics of the cell on data where ground truth is
known, and reports the failure mode honestly.

## Setup

\`\`\`
pip install -r requirements.txt
python run_horizon.py
\`\`\`

## Attribution & license

This work began as a fork of
[HusseinJammal/Liquid-Neural-Networks-in-Stock-Market-Prediction](https://github.com/HusseinJammal/Liquid-Neural-Networks-in-Stock-Market-Prediction)
(Apache-2.0); the original LICENSE is retained and changes are noted in
NOTICE per the license terms. The architecture and implementation in this
repository are my own. LTC dynamics follow Hasani et al., "Liquid
Time-constant Networks" (2021); spectral-radius conditioning follows the
echo state property (Jaeger, 2001).

Copyright (c) 2026 Matthew J. Werner.
