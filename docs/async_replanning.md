# Asynchronous Newton replanning and wall-clock budgets

## SUPER source comparison

Source: `hku-mars/SUPER`, `super_planner/src/super_core/super_planner.cpp`.

- `generateExpTraj` computes a future trajectory clock `replan_state_TT = replan_process_start_TT + cfg_.replan_forward_dt` (near line 419), evaluates the previous command trajectory at this forecast time and builds a collision-checked time-stamped guide.
- The reference rejects a newly optimized expansive trajectory if its planning latency exceeds `replan_forward_dt` (near line 758); `ReplanOnce` also checks its total latency against 90% of the forward interval (near line 234).
- The reference stitches an already-executing trajectory prefix to a new optimized suffix. It also implements a separate backup trajectory. Neither a backup planner nor its semantics are part of PathCover's source; this adaptation retains the last accepted trajectory instead.

## Implementation

`control/async_planner.py` creates one persistent **spawned CPU process** per flight. The main process alone owns Isaac Lab 3.0, Newton and CUDA. The worker owns a `Planner`, the 3-D point-cloud map, LiDAR raycasting, RISP and GCOPTER. Only state, prior polynomial coefficients, goals and completed plans cross the process pipe.

At each nominal 0.5 s **simulation-time** planning slot:

1. If the worker is busy or the previous candidate is awaiting activation, skip the slot rather than enqueueing an obsolete request. The old accepted trajectory continues to execute and Newton keeps advancing.
2. Estimate a future activation time from the minimum prediction horizon and the most recently measured simulator-clock/wall-clock ratio. The lead-time formula is `max(minimum_horizon, regular_wall_budget * observed_sim_per_wall)` simulated seconds. The previous 20% multiplier, 0.1 s addition, and forced ratio of at least 1 have been removed. Late candidates still fail the existing handoff deadline check.
3. Evaluate the **currently accepted** trajectory's position, velocity and acceleration at the future time; pass this hard start boundary to the independent worker. The current measured pose and quaternion are used to simulate the most recent LiDAR scan.
4. The worker checks an absolute monotonic deadline at planning stages and on every L-BFGS line-search evaluation. If the complete job, including LiDAR acquisition and trajectory diagnostics, exceeds the wall-clock budget, its result cannot replace the active plan. Original GCOPTER objective terms, analytic gradients and default cold-start optimizer settings are unchanged.
5. If the result arrives before the predicted activation time, keep executing the old trajectory until the target time. At handoff, require position error ≤0.2 m and speed-vector error ≤0.5 m/s relative to the new polynomial's start. Apply the pre-existing execution constraint check; reject late or inconsistent trajectories without changing the active reference. Record the predicted-state mismatch and actual reference jump separately.

The first route is planned **without stopping Newton**, while the quadrotor maintains initial hover. Its wall budget is separate from normal replanning. The planning worker's one-time initialization is logged independently of its first solve.

Default budgets: **4.0 s initial**, **1.5 s subsequent**; minimum handoff horizon **0.8 s simulation time**, adjusted using the measured clock speed. These timeouts are configurable rejection policies, not performance acceptance targets. Evaluate actual planning latency, effective handoff rate, and collision-free arrival independently. To change them:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m isaac_pathcover.demo \
  --scene both --seeds 0 1 2 3 4 --duration 35 \
  --solve-budget 1.5 --startup-budget 4.0 --forward-horizon 0.8 \
  --output results/async_validation
```

The original paused-clock baseline is still available with `--synchronous`.

## Acceptance and metrics

In asynchronous runs, `planning.json` records request time, target activation time, completion time, simulation progress during the wall-clock job, job/solver latency, timeout, optimizer status and handoff rejection reasons. `result.json` reports total physics steps while the worker was busy, skipped planning slots, worker timeouts, forecast mismatch, actual LiDAR fusion rate, and collision/arrival/tracking metrics. This distinguishes:

- **Nonblocking simulation**: physics advances while the planner is solving; test this by `physics_steps_while_worker_busy > 0` and `physics_progress_while_planning_sim_s > 0`.
- **Correct trajectory handoff**: the solver may succeed while the trajectory is refused because of delay, dynamic mismatch, or soft-constraint residual.
- **Wall-clock 2 Hz planning**: a complete solver decision would need ≤0.5 s wall time every cycle. The simulation real-time factor measures test throughput, not planner correctness or its scheduling deadline.

All debug serialization and offline 3-D HTML rendering occurs **after** the continuous stepping loop. The reported `wall_time_s` measures the flight loop rather than post-flight file writes or initial Isaac Lab boot.

Initial one-seed Newton acceptance (2026-11-15, 35 s maximum simulation horizon; output: `results/async_adaptive_20261115`):

| Scene | Arrival | Collisions | Simulation time | Flight wall time | Completed handoffs | Busy control steps |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| medium | yes | 0 | 13.41 s | 19.51 s | 5 | 647 |
| dense | yes | 0 | 15.79 s | 21.79 s | 7 | 798 |

These are two successful cases, not a statistical guarantee. A separate five-seed repeat uses the same source and the same execution criteria. The historical synchronous results remain in `docs/newton_validation.md` and are not relabeled as continuous-time evidence.

## Five-seed continuous-clock validation

Five fixed seeds per scene were rerun on 2026-11-15, preserving Newton physics at 400 Hz and control at 100 Hz. Each episode could use up to 35 s simulation time. Raw results are under `results/async_validation_20261115/{medium,dense}/seedNNN/`; the compact report is [`docs/data/async_validation.json`](data/async_validation.json).

| Metric | medium | dense |
| --- | ---: | ---: |
| Collision-free arrivals | **5/5** | **5/5** |
| Collisions | 0 | 0 |
| Out-of-bounds episodes | 0 | 0 |
| Completed planning jobs | 52 | 45 |
| Completed job wall time P50 / P95 | 1.376 / 2.330 s | 1.208 / 2.501 s |
| Regular jobs exceeding 1.5 s budget | 18/46 | 5/40 |
| Accepted trajectory handoffs | 25 | 34 |
| Late or predicted-state mismatch rejections | 0 | 0 |
| Physics control steps while planner busy | 5,385 | 4,717 |
| Pooled tracking RMSE | 1.36 cm | 1.50 cm |
| Simulator real-time factor (simulation/wall) | 0.822× | 0.830× |

There were 97 completed decisions and 10,102 physics control steps while the independent planner was busy. The five runs of each scene all arrived without collision, but **the simulator still ran slower than wall time** and GCOPTER still required more than 0.5 s for most decisions. A deadline keeps the old reference valid rather than making the optimizer itself complete on time. Scan fusion also runs at the actual accepted request frequency, not at an unconditional 2 Hz: requests are skipped while a job is in progress or a scheduled handoff is pending.

The medium five-seed run's source snapshot differs from the final repository **only in `demo.py`**: the changed lines report actual instead of nominal LiDAR fusion rate after the flight. No control, mapping, prediction or optimizer logic changed. The dense five-seed run's source snapshot matches the finalized Python source. The raw source snapshots and source hashes are preserved with the results.

The observed maximum body angular-rate norms were 4.04 rad/s (medium) and 2.96 rad/s (dense), versus the GCOPTER planned 3 rad/s upper bound. The work therefore establishes nonblocking, collision-free flight for these ten seeds, **not** guaranteed adherence to every actuator constraint or guaranteed real-time execution.

The five-seed results above precede the simplified forward-horizon policy. The complete new results must be evaluated separately; the old report remains an immutable baseline.

## Simplified lookahead and shared flight recorder

The latest implementation removes the fixed 20%/0.1 s lead-time margins and uses the last observed ratio of simulation-time progress to wall-clock solve time. Synchronous and asynchronous modes share `FlightTrace` for state sampling, Newton control steps, arrival criteria and flight records. The unused `Planner.last_result` field has been removed; the separate `GCopter.last_result` remains available for numerical oracle tests.

Five fixed seeds per scene were rerun with the shorter lookahead. Both versions use the same scenes, planning objective and collision criteria; CPU/GPU load was not isolated between runs. Compact measurements: [`data/simplified_validation.json`](data/simplified_validation.json).

| Metric | medium: prior → simplified | dense: prior → simplified |
| --- | ---: | ---: |
| Collision-free arrivals | 5/5 → **5/5** | 5/5 → **5/5** |
| Median forward horizon (sim s) | 1.900 → **1.259** | 1.900 → **1.033** |
| Decision wall-time P50 (s) | 1.376 → 1.294 | 1.208 → 1.197 |
| Accepted handoffs | 25 → 29 | 34 → 41 |
| Stale/mismatched handoffs | 0 → 0 | 0 → 0 |

The 1.5 s default is an **optional policy cutoff** for accepting completed trajectories. It is not the primary performance metric. Actual solver latency, valid handoff frequency and collision-free arrival remain the evaluation criteria.

The original `--synchronous` mode was also exercised for 2 s in medium. It completed 4 planning attempts with finite state and no collision. The first 49 recorded control samples (0.49 s) matched the historical synchronous reference bit-for-bit for time, physical state, trajectory reference, motor state, and requested wrench. The sample-by-sample comparison is recorded in [`data/simplified_validation.json`](data/simplified_validation.json).
