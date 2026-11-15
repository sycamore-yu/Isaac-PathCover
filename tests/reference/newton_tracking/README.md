# Fixed-command Newton tracking replay

Historical record from commit `8dcdab4`. The geometric tracker and its replay
scripts were replaced by the Ideal/TurboMPC controllers. Use that commit to
reproduce the commands below; all recorded results remain in `results/`.

This offline experiment tests the medium-flight failure recorded on 2026-11-15.
It keeps the recorded 100 Hz position, velocity, acceleration and yaw commands
unchanged. Both runs use `GeometricTracker`, its existing gains, `RotorActuator`,
the original motor equations, and the same Newton scene and solver settings.

`ReplayFlightEnv` overrides only `_apply_action`. At 100 Hz it applies the wrench
computed once per environment step. At 400 Hz it calls the same tracker before
each physical substep, with the same reference held for all four substeps.
No planner runs during replay. This experiment does not add a runtime controller
mode or alter the flight pipeline.

Run the two commands sequentially from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/reference/newton_tracking/replay.py --input-dir results/super_risp_development_20261115/newton_tracking_replay --rate 100 --output results/super_risp_development_20261115/newton_tracking_replay/hz100
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/reference/newton_tracking/replay.py --input-dir results/super_risp_development_20261115/newton_tracking_replay --rate 400 --output results/super_risp_development_20261115/newton_tracking_replay/hz400
.venv/bin/python tests/reference/newton_tracking/compare.py results/super_risp_development_20261115/newton_tracking_replay
```

The frozen input is `input_flight.npz`, SHA256
`1bc18000f31a1fe87e7d4f7b11e871522ed30fbe1ce53b90a3cbac88a8c2e3a8`.
It contains 2,642 command intervals, ending at 26.42 s. `input_result.json`
preserves the original scene/seed metadata. `input_source_hashes.json` preserves
the source snapshot when available. The source location is listed in
`input_manifest.json`; all replay commands read the frozen files.

Each result contains raw `replay.npz`, `tracking.csv`, `attitude.csv` and
`result.json`. Physics states and control diagnostics are sampled immediately
before integration at 400 Hz. The 100 Hz state is sampled after integration.
Geometry collision checks use the existing oriented-body intersection function.
Contact-force values use the unchanged Newton contact sensor. The force-z and
negative requested-thrust metrics measure the current feedback command before
rotor allocation. The desired-attitude increment is an angle between consecutive
400 Hz records; the 100 Hz experiment holds that attitude for four records.

The 100 Hz run must reproduce the saved state and wrench sequence before its
collision, which is checked explicitly in `result.json`. The 400 Hz result is a
fixed-command tracking comparison. Its command stream still contains decisions
made from the original 100 Hz flight's observations, so a successful replay
requires a separate live-planning flight before adoption in the runtime.

## Observed result

Both runs completed all 26.42 s with finite states on the same NVIDIA GeForce
RTX 4090. Torch used one CPU thread. Versions were NumPy 2.5.1, SciPy 1.17.0,
Torch 2.11.0+cu130, Isaac Lab package 28.0.0, Newton 1.6.1 and Warp 1.17.0.
The input, tracker, actuator, motor, environment and scene hashes match across
the pair. The 100 Hz replay reproduced every saved state and requested-wrench
component exactly, including the first collision.

| Measurement | 100 Hz wrench | 400 Hz wrench |
| --- | ---: | ---: |
| First body collision | 20.7350 s | 21.0575 s |
| First sensor contact | 20.7400 s | 21.0600 s |
| Maximum contact force, full recording | 203.542 N | 310.283 N |
| Tracking RMSE, 17.61–20.735 s | 0.94908 m | 0.86344 m |
| Maximum tracking error, same window | 3.11531 m | 2.79993 m |
| Maximum attitude error, same window | 3.14148 rad | 3.11935 rad |
| First negative requested thrust | 19.8200 s | 19.8200 s |
| Negative requested-thrust duration, same window | 0.3200 s | 0.4850 s |
| Tracking RMSE, full recording | 0.82773 m | 3.02868 m |
| Maximum tracking error, full recording | 3.86758 m | 9.14977 m |

The common comparison window starts when the original actual speed first exceeds
0.5 m/s and ends at its first collision. This excludes the long initial hover.
The 400 Hz update improves tracking RMSE by about 9.0% in this window, but both
runs develop an approximately pi-radian attitude error and collide. The change
delays the first collision by 0.3225 s and does not remove the negative-thrust
event. This experiment therefore does not support adopting the rate change as a
solution to the observed tracking failure. The runtime controller remains
unchanged.

Force-z is first negative at 17.88 s in the 100 Hz replay, while requested thrust
remains positive then. The later negative-thrust interval starts at pre-step
time 19.82 s; its row in the original post-step `flight.time` is labelled
19.83 s. The 2.20585-rad desired-attitude change occurs at this same interval.
Negative force-z alone is insufficient to identify the onset of failure.

Within the shared motion window, reference speed reaches 8.00018 m/s,
acceleration 15.11419 m/s² and jerk 140.16949 m/s³. With zero position and velocity
error, the existing tracker's desired attitude changes at up to 19.75845 rad/s,
measured by consecutive recorded attitudes divided by 0.01 s. The unchanged
source defaults disable jerk, angular-rate and thrust penalties. The existing
tracker includes yaw-rate feedforward, but its angular-velocity command does not
include the roll/pitch rates induced by reference jerk. These measurements
support a tracking-capability mismatch for this particular reference and plant.
They do not establish a general limitation of SUPER. `reference_demand.json`
preserves these calculations and their definition.

`comparison.json` retains the exact metrics and provenance. `comparison.html`
plots tracking error, attitude error, desired-attitude increment, feedback
force-z, requested thrust, actual body-rate norm and contact force. After the
original first collision, the fixed command stream still contains planning
decisions based on the original flight's diverged state. The full-recording
metrics quantify this replay, while the shared pre-collision window isolates
the initial tracking failure.
