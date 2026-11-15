# Replanning engineering optimizations (2026-11-15)

## Source contracts

- PathCover `nanovoxmap.hpp` integrates rays into sparse occupancy blocks incrementally. Its `occupancy_inflation/src/point_cloud_inflater.cpp::finishIncremental` only accumulates **new** seeds; it cannot remove occupancy cleared by later rays. `finishOneShot` fully rebuilds on removals. The ROS node's `incremental` parameter is enabled by default.
- PathCover `trajectory_server/src/trajectory_server.cpp::replanImpl` seeds hard initial conditions from measured position, velocity, and acceleration, deliberately **not** from the last optimized trajectory. `gcopter_solver.hpp` constructs a new shortest-path and vertex-coordinate initialization for every solve; the repository does not carry optimized L-BFGS variables across replans.
- GCOPTER (`ZJU-FAST-Lab/GCOPTER`) implements MINCO, flatness and analytic-gradient L-BFGS optimization; it does not provide a map-update or cross-replan warm-start service.
- SUPER `super_planner/src/super_core/super_planner.cpp` constructs a collision-checked guide path with timestamps from the previous trajectory. `src/traj_opt/exp_traj_optimizer_s4.cpp::processCorridorWithGuideTraj` maps that guide onto the newly generated safe-flight corridor and initializes times/waypoints. Its replanning logic also checks planning delay against `replan_forward_dt`.

Upstream repositories: [PathCover](https://github.com/kunalnk123690/PathCover), [GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER), [SUPER](https://github.com/hku-mars/SUPER).

## Changes in this repository

1. `mapping/inflation.py`: maintain a count of covering ellipsoid kernels per output cell. A changed occupancy seed adds or removes exactly one kernel, so clearing rays no longer require a full dilation. Rendered solid/surface cells retain the upstream float32 lattice and morphology. Dense kernel counting incurs an **additional cold-start cost**; it is not hidden in runtime benchmarks. Integer keys replace repeated structured-record sorts for the 0.1 m inflation lattice.
2. `planning/pipeline.py`: when the previous DMP route ends at the same goal, project the measured start onto its nearest segment and retain the suffix **only if every sampled point is free in the current inflated grid**. Otherwise recompute JPS + DMP. Always generate a fresh corridor from the current occupancy points. This is an intentional replanning-policy change, not a claim that every reused path is the new shortest path.
3. `trajectory/gcopter.py`: sample the active trajectory at its current execution time; project its waypoint guide onto the **new** corridor's overlap polytopes, derive positive segment times, and use it as a candidate L-BFGS initial point. Choose the guide only when its value under the **original objective** is below the cold-start value. Keep the original GCOPTER objective, gradients, boundary measurements, optimizer tolerances, iteration cap and execution checks unchanged. Record `initialization` as `cold` or `guide`.

## Closed-loop Newton verification

Seed 0 for each scene. Same scene assets and Newton/L2F dynamics as the earlier validation, but not a paired isolated-CPU benchmark. Original validation: `results/validation_execution_20261114`; revised validation: `results/engineering_opt_20261115`. Timing is *wall clock*; the Newton simulation clock still pauses during planning.

| Scene | Execution | Plans | P50 decision / s | P95 decision / s | P50 optimizer / s | P50 inflation / s | Collision-free arrival |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| medium | prior | 20 | 2.279 | 9.271 | 0.601 | 0.245 | 1/1 |
| medium | optimized | 20 | **0.913** | 3.121 | **0.286** | **0.115** | 1/1 |
| dense | prior | 25 | 2.543 | 4.046 | 0.732 | 0.382 | 1/1 |
| dense | optimized | 22 | **1.216** | 2.532 | **0.437** | **0.177** | 1/1 |

- medium: 14/20 cached search suffixes, 9/20 selected trajectory guides; 19/20 optimizer successes and 19/20 accepted handoffs.
- dense: 19/22 cached search suffixes, 7/22 selected trajectory guides; 22/22 optimizer successes and 20/22 accepted handoffs.
- All **42/42** optimized complete decision cycles exceeded the requested 0.5 s interval. This is not real-time replanning, regardless of arrival outcome.
- The remaining long tail is mainly nonlinear GCOPTER optimization: e.g. medium at simulation t=1.5 s reached 1,000 L-BFGS iterations and spent 6.268 s optimizing; the candidate failed and the executing reference was retained. Lowering iteration caps or running planning asynchronously requires another closed-loop safety comparison.
- On the first static snapshot, initializing the exact incremental coverage map costs about 0.9 s in a standalone Python timing run; this cold-start cost needs separate budgeting.

## Numerical checks

The existing C++ reference oracles test the unchanged cold-start objective, gradient, MINCO coefficients, mapping, RISP and search. Added tests exercise exact inflation under additions and removals, previous-path reuse, and guide-based replanning. Full test command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

The one-seed runs validate implementation behavior, **not** the previously established five-seed arrival rate. All simulator output remains under the ignored `results/` directory.
