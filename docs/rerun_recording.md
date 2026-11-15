# Rerun flight recording

Convert saved Newton flight data into an interactive Rerun recording without rerunning the physics simulator.

## Produce recordings

Requires Python 3.12, NumPy and `rerun-sdk` (validated with Rerun 0.38.1). Use a Python environment containing the SDK; this dependency is deliberately separate from the Isaac Lab simulation environment.

```bash
python scripts/record_rerun.py --scene both --seed 0 \
  --source results/simplified_validation_20261115 \
  --output results/rerun_20261115 --fps 12
rerun rrd verify results/rerun_20261115/medium_seed000.rrd
rerun rrd verify results/rerun_20261115/dense_seed000.rrd
rerun results/rerun_20261115/medium_seed000.rrd \
      results/rerun_20261115/dense_seed000.rrd
```

The current server already has a standalone `rerun-sdk` pipx environment. Its Python interpreter is `/home/tong/.local/share/pipx/venvs/rerun-sdk/bin/python`; the CLI is `/home/tong/.local/bin/rerun`.

Recordings include:

- Original obstacle boxes, start and goal, a body-oriented quadrotor and four rotor markers.
- Orange executed trajectory, blue reference trajectory, and purple currently accepted MINCO plan.
- Time-aligned commanded target, true/desired speed, positional tracking error and measured contact force.
- Quaternion conversion from Newton state `[w,x,y,z]` to Rerun `[x,y,z,w]`.

`results/rerun_20261115/summary.json` holds source-derived status and recorder metadata. `.rrd` and screenshots are generated assets, excluded from Git together with flight data.

## Recorded examples

| Recording | Simulated flight duration | Frames | Collisions | Destination reached |
| --- | ---: | ---: | ---: | --- |
| `medium_seed000.rrd` | 16.74 s | 211 | 0 | yes |
| `dense_seed000.rrd` | 19.34 s | 243 | 0 | yes |

Both recordings use the actual 2026-11-15 five-seed Newton validation data. The Rerun Viewer can select either recording under **Sources** and play its `sim_time` timeline. Offline Rerun screenshots were independently checked at simulated times 9 s and 11 s.
