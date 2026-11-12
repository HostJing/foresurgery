"""Battery dispatch LP and open-loop execution.

Units: kWh per 15-min step throughout; power limits are converted with
``DT = 0.25 h``.  Variable order per step ``t`` (T steps)::

    imp_t  >= 0                 grid import
    dump_t >= 0                 curtailed surplus (zero value)
    ch_t   in [0, p_lim]        charging energy
    dis_t  in [0, p_lim]        discharging energy
    soc_t  in [soc_lo, soc_hi]  state of charge (soc_T lb = max(soc_lo, soc0))
    D      >= 0                 daily peak import (epigraph), only if c_dem > 0

Constraints::

    balance: imp_t - dump_t - ch_t + dis_t = nl_t
    SOC:     soc_{t+1} - soc_t - eta_c*ch_t + dis_t/eta_d = 0
    peak:    imp_t - D <= 0

Objective::

    sum_t price_t * imp_t + c_dem * D

The peak epigraph is what couples the whole horizon: without it the bill is
linear in the total energy import and the timing of battery actions does not
matter (see ``Corollary 1'' of the paper).

Peak-variable slot
------------------
``peak_slot`` selects where the epigraph variable ``D`` lives in the LP vector.

``"own"`` (default)
    ``D`` gets its own slot at ``5T + 1``, i.e. the model as described in the
    paper: ``imp_t <= D`` with ``D`` free, planned cost equal to realised cost.
    Every number in version 1.1 of this repository and in the revised
    manuscript comes from this slot.
``"legacy"``
    ``D`` occupies index ``5T``, which is also the last SOC variable.  This is
    the formulation behind version 1.0 of the repository and the first
    submitted draft; it is kept only so that those numbers can be regenerated
    bit-exactly.  It has a consequence the reader should know about: ``D`` is
    then tied to the *terminal SOC* rather than to the peak import, so the
    demand charge does not enter the dispatch decision at all (the charging
    plan is identical for ``c_dem = 0`` and ``c_dem = 2``), and the planned
    cost is not the realised cost.

Set the environment variable ``FORESURGERY_PEAK_SLOT=legacy`` to run every
script under the old formulation at once, or pass ``peak_slot="legacy"`` per
call.  ``docs/LIMITATIONS.md`` tabulates what moves between the two.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.optimize import linprog

from .config import BATT, DT

__all__ = ["solve_dispatch", "execute_open_loop", "realized_cost", "PEAK_SLOT"]

#: default slot for the peak variable; override with ``FORESURGERY_PEAK_SLOT``
PEAK_SLOT = os.environ.get("FORESURGERY_PEAK_SLOT", "own")

#: retry a failed LP with another HiGHS algorithm; ``FORESURGERY_LP_RETRY=0``
#: restores the v1.0 behaviour, which silently drops a fraction of issue times
LP_RETRY = os.environ.get("FORESURGERY_LP_RETRY", "1").lower() not in ("0", "", "false")


def solve_dispatch(nl: np.ndarray, price: np.ndarray, p_rated: float,
                   e_cap: float, c_dem: float = 0.0,
                   soc_start: float | None = None,
                   soc_term: float | None = None,
                   soc_floor: float | None = None,
                   return_duals: bool = False,
                   peak_slot: str | None = None,
                   fix_initial_soc: bool = False,
                   retry: bool | None = None):
    """Optimal battery plan for the net-load *forecast* ``nl`` [T].

    Parameters
    ----------
    nl
        Net-load forecast in kWh per step (load minus PV).
    price
        Energy price per step, EUR/kWh, already aligned to the site's local
        clock (see :func:`foresurgery.surgery.shifted_price`).
    p_rated, e_cap
        Power [kW] and energy [kWh] rating; see :func:`config.battery_specs`.
    c_dem
        Demand charge, EUR per kW and settlement period. Zero disables the
        peak epigraph and the LP becomes a pure arbitrage problem.
    soc_start
        Initial SOC in kWh (default ``BATT['soc_init'] * e_cap``).
    soc_term
        Absolute terminal-SOC lower bound in kWh (default
        ``max(soc_lo, soc_start)``, the cycle-neutrality convention).
    soc_floor
        Absolute minimum SOC in kWh over *all* steps (default
        ``BATT['soc_min'] * e_cap``); used by the reserve-constrained variant.
    return_duals
        Also return the energy-balance duals, i.e. the shadow price
        ``lambda[T]`` of net load at each step (EUR/kWh).
    peak_slot
        ``"legacy"`` (default) or ``"own"``; see the module docstring.
    fix_initial_soc
        Pin ``soc_0`` to the given initial value instead of only bounding it
        below.  As published the lower bound alone is enforced, which lets the
        LP start the day with a fuller battery than ``soc_start`` (free energy,
        worth up to ``(soc_max - soc_init) * e_cap`` per day).

    Returns
    -------
    ``(planned_cost, ch[T], dis[T], imp[T])``, or with ``return_duals`` the
    same tuple plus ``lam[T]``. ``None`` when the LP is infeasible.
    """
    T = len(nl)
    ec, ed = BATT["eta_c"], BATT["eta_d"]
    p_lim = p_rated * DT
    soc0 = (BATT["soc_init"] * e_cap) if soc_start is None else float(soc_start)
    soc_lo, soc_hi = BATT["soc_min"] * e_cap, BATT["soc_max"] * e_cap
    if soc_floor is not None:
        soc_lo = max(soc_lo, float(soc_floor))
        soc0 = max(soc0, soc_lo)
    term_lb = max(soc_lo, soc0) if soc_term is None else float(soc_term)

    V = 5 * T + 1 + (1 if c_dem > 0 else 0)
    slot = PEAK_SLOT if peak_slot is None else peak_slot
    if slot not in ("legacy", "own"):
        raise ValueError(f"peak_slot must be 'legacy' or 'own', got {slot!r}")
    # "legacy" shares index 5T with the terminal SOC; "own" uses the extra slot.
    iD = 5 * T if slot == "legacy" else 5 * T + 1
    c = np.concatenate([price, np.zeros(V - T)])
    if c_dem > 0:
        c[iD] = c_dem

    lb = np.zeros(V)
    ub = np.empty(V)
    ub[:T] = np.inf          # imp
    ub[T:2 * T] = np.inf     # dump
    ub[2 * T:4 * T] = p_lim  # ch, dis
    ub[4 * T:5 * T + 1] = soc_hi
    if c_dem > 0:
        ub[iD] = np.inf
    lb[4 * T] = soc0
    if fix_initial_soc:
        ub[4 * T] = soc0
    lb[4 * T + T] = term_lb   # soc_T
    lb[4 * T + 1:4 * T + T] = soc_lo

    Aeq = np.zeros((2 * T, V))
    idx = np.arange(T)
    Aeq[idx, idx] = 1.0            # imp
    Aeq[idx, T + idx] = -1.0       # dump
    Aeq[idx, 2 * T + idx] = -1.0   # ch
    Aeq[idx, 3 * T + idx] = 1.0    # dis
    beq = np.zeros(2 * T)
    beq[:T] = nl
    r = np.arange(T, 2 * T)
    Aeq[r, 4 * T + 1:4 * T + T + 1] += np.eye(T)   # + soc_{t+1}
    Aeq[r, 4 * T:4 * T + T] -= np.eye(T)           # - soc_t
    Aeq[r, 2 * T + idx] -= ec
    Aeq[r, 3 * T + idx] += 1.0 / ed

    A_ub, b_ub = None, None
    if c_dem > 0:
        A_ub = np.zeros((T, V))
        A_ub[idx, idx] = 1.0       # imp_t
        A_ub[idx, iD] = -1.0       # - D <= 0
        b_ub = np.zeros(T)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq,
                  bounds=list(zip(lb, ub)), method="highs")
    if not res.success and (LP_RETRY if retry is None else retry):
        # HiGHS intermittently reports this (always feasible) LP as infeasible;
        # a different algorithm on the unpresolved problem recovers it.
        for method, opts in (("highs-ds", {"presolve": False}),
                             ("highs-ipm", {"presolve": False})):
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq,
                          bounds=list(zip(lb, ub)), method=method, options=opts)
            if res.success:
                break
    if not res.success:
        return None
    imp, ch, dis = res.x[:T], res.x[2 * T:3 * T], res.x[3 * T:4 * T]
    if return_duals:
        lam = res.eqlin.marginals[:T]  # balance-constraint shadow prices
        return float(res.fun), ch, dis, imp, lam
    return float(res.fun), ch, dis, imp


def execute_open_loop(ch: np.ndarray, dis: np.ndarray,
                      nl_actual: np.ndarray, price: np.ndarray,
                      c_dem: float = 0.0) -> float:
    """Realised cost when the *planned* battery actions meet the actuals.

    The plan is executed without feedback: charging and discharging follow the
    schedule, and whatever mismatch is left is settled at the grid.  Surplus is
    curtailed, so exports earn nothing.
    """
    net = nl_actual + ch - dis
    imp = np.maximum(net, 0.0)
    cost = float(np.sum(price * imp))
    if c_dem > 0:
        cost += c_dem * float(imp.max())
    return cost


def realized_cost(plan, nl_actual: np.ndarray, price: np.ndarray,
                  c_dem: float = 0.0) -> float:
    """Convenience wrapper: realised cost of a plan returned by the LP."""
    return execute_open_loop(plan[1], plan[2], nl_actual, price, c_dem=c_dem)
