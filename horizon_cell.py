'''
horizon_cell.py — Multi-timescale Liquid Time-Constant cell.
Copyright (c) 2026 Matthew J. Werner. All rights reserved.
Status: active development — research prototype, not yet deployable.
'''

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass
class HorizonConfig:
    input_dim: int = 2
    hidden_dim: int = 32                 # D per substate
    output_dim: int = 1
    tau: tuple = (1.0, 10.0, 100.0)      # fast, mid, slow
    tick: tuple = (1, 5, 20)             # update cadence per substate
    dt: float = 0.125
    ode_unfolds: int = 1
    solver: str = "fused"                # "fused" (stable) | "euler" (naive, diverges)
    rec_gain: float = 1.5                # naive recurrent scale (~spectral radius)
    stabilize: bool = False              # echo-state spectral rescale
    rho_target: float = 0.9
    cell_clip: float = 0.0               # 0 = off
    washout_min: int = 3000              # hard floor before is_warm may fire
    warm_window: int = 200               # settle-detector window
    warm_var_tol: float = 1e-4           # relative variance tolerance
    ridge: float = 1e-3
    seed: int = 0


class LTCSubState:
    """One timescale. Owns weights, runs the LTC step, holds off-tick."""

    def __init__(self, cfg, tau, tick, rng):
        # TODO: store tau/tick/dt/solver/clip/unfolds
        # TODO: init Wx (D,U), Wh (D,D) random; b zeros; A reversal vector
        # TODO: if cfg.stabilize -> rescale_spectral(rho_target)
        # TODO: state h zeros; interval-mean accumulator (_acc, _n)
        ...

    def rescale_spectral(self, rho):
        """Scale Wh to spectral radius rho < 1 (echo state property)."""
        # TODO: eig = max|eigvals(Wh)|; Wh *= rho/eig
        ...

    def step(self, x, global_step):
        """Accumulate input; on tick, advance the ODE; else hold."""
        # TODO: accumulate x into _acc; if global_step % tick != 0 -> return h
        # TODO: u = interval mean; reset accumulator
        # TODO: dt_u = dt*tick / unfolds
        # TODO: unfold loop:
        #         f = sigmoid(Wx@u + Wh@h + b)        # positive conductance
        #         lam = 1/tau + f
        #         euler:  h = h + dt_u*(-lam*h + f*A)
        #         fused:  h = (h + dt_u*f*A) / (1 + dt_u*lam)
        # TODO: optional clip; store h; return h
        ...

    @property
    def norm(self):
        # TODO: ||h||
        ...

    def reset(self):
        # TODO: zero h and accumulator
        ...


class HorizonCell:
    """Three independent LTCSubStates + concat readout. No coupling, no gate."""

    NAMES = ("fast", "mid", "slow")

    def __init__(self, cfg):
        # TODO: rng from seed; build 3 substates over zip(tau, tick)
        # TODO: W_out = None; t = 0; warm_step = None
        # TODO: per-name norm trace; warm-detector ring buffer (deque maxlen=warm_window)
        ...

    def _state(self):
        # TODO: concat([s.h for s in sub])
        ...

    def step(self, x):
        """One input -> one output. Advances the global clock."""
        # TODO: t += 1; step every substate with global step t
        # TODO: record norms to trace + warm buffer; _check_warm()
        # TODO: if W_out is None -> zeros; else W_out @ [state, 1.0]
        ...

    def _check_warm(self):
        """Set warm_step when norm-variance settles, past washout_min floor."""
        # TODO: gate on warm_step None, t >= washout_min, buffer full
        # TODO: rel = var/mean^2 across window; all < warm_var_tol -> warm_step = t
        ...

    @property
    def is_warm(self):
        # TODO: warm_step is not None
        ...

    def run(self, X):
        """Sequence (T, input_dim) -> outputs (T, output_dim)."""
        # TODO: step over rows
        ...

    def collect_states(self, X):
        """Reset, drive over X, return design matrix (T, 3*D + 1)."""
        # TODO: reset; per row step + append [state, 1.0]
        ...

    def fit_readout(self, X, y, ridge=None):
        """Ridge-solve W_out on POST-warm-up, finite rows. Returns rows used."""
        # TODO: collect_states; slice off [:warm_step] (or washout_min)
        # TODO: mask non-finite rows (diverged cells); raise if none left
        # TODO: G = A.T@A + lam*I; W_out = solve(G, A.T@b).T
        ...

    def state_trace(self):
        """Per-step norms from step 0 — first-class instrument (failure figure)."""
        # TODO: dict name -> np.array(trace)
        ...

    def reset(self, keep_readout=True):
        # TODO: reset substates; zero t/warm_step/trace/buffer
        # TODO: if not keep_readout -> W_out = None
        ...
  
