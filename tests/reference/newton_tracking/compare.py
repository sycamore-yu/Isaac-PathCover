"""Summarize the fixed-command replay pair and save an interactive timeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from isaac_pathcover.control.tracking import GeometricTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = {hz: np.load(args.directory / f"hz{hz}" / "replay.npz") for hz in (100, 400)}
    reports = {hz: json.loads((args.directory / f"hz{hz}" / "result.json").read_text()) for hz in runs}
    for field in ("time", "references", "yaw_references", "command_time"):
        np.testing.assert_array_equal(runs[100][field], runs[400][field])
    np.testing.assert_array_equal(reports[100]["source_replay_max_state_error"], np.zeros(13))
    np.testing.assert_array_equal(reports[100]["source_replay_max_requested_wrench_error"], np.zeros(4))
    source_speed = np.linalg.norm(runs[100]["states"][:, 7:10], axis=1)
    onset = float(runs[100]["time"][np.flatnonzero(source_speed > .5)[0]])
    end = reports[100]["first_body_collision_s"]
    summary = {"same_reference_verified": True, "source_100hz_reproduced_exactly": True,
               "motion_to_source_collision_window_s": [onset, end], "runs": {}}
    reference = runs[100]["references"]
    mask = (runs[100]["time"] >= onset) & (runs[100]["time"] < end)
    tracker = GeometricTracker()
    ideal = []
    for ref, yaw in zip(reference, runs[100]["yaw_references"]):
        state = np.r_[ref[:, 0], [1., 0., 0., 0.], ref[:, 1], np.zeros(3)]
        ideal.append(tracker.command(state, ref[:, 0], ref[:, 1], ref[:, 2],
                                    yaw=yaw[0], yaw_rate=yaw[1])[1])
    ideal = np.asarray(ideal)
    ideal_rate = Rotation.from_matrix(ideal[:-1].transpose(0, 2, 1) @ ideal[1:]).magnitude() / .01
    demand = {"window_s": [onset, end],
              "reference_velocity_norm_max_m_s": float(np.linalg.norm(reference[mask, :, 1], axis=1).max()),
              "reference_acceleration_norm_max_m_s2": float(np.linalg.norm(reference[mask, :, 2], axis=1).max()),
              "reference_jerk_norm_max_m_s3": float(np.linalg.norm(reference[mask, :, 3], axis=1).max()),
              "ideal_desired_attitude_finite_difference_max_rad_s": float(ideal_rate[mask[1:]].max()),
              "ideal_desired_attitude_derivative_definition": "angle(R_ref[k-1]^T R_ref[k])/0.01 on held100Hz references, desired orientation from unchanged GeometricTracker with zero position/velocity error"}
    summary["reference_demand"] = demand
    (args.directory / "reference_demand.json").write_text(json.dumps(demand, indent=2) + "\n")
    figure = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=.025)
    fields = [("time", "tracking_error", "Tracking error (m)"),
              ("physics_time", "physics_attitude_error", "Attitude error (rad)"),
              ("physics_time", "physics_desired_increment", "Desired angle step (rad)"),
              ("physics_time", "physics_feedback_force", "Feedback force z (N)"),
              ("physics_time", "physics_requested_wrench", "Requested thrust (N)"),
              ("physics_time", "physics_states", "Body rate (rad/s)"),
              ("time", "contact_force", "Contact force (N)")]
    for hz, data in runs.items():
        mask = (data["time"] >= onset) & (data["time"] < end)
        physics_mask = (data["physics_time"] >= onset) & (data["physics_time"] < end)
        row = dict(reports[hz])
        row["motion_window_tracking_rmse_m"] = float(np.sqrt(np.mean(data["tracking_error"][mask] ** 2)))
        row["motion_window_tracking_max_m"] = float(np.max(data["tracking_error"][mask]))
        row["motion_window_attitude_error_max_rad"] = float(np.max(data["physics_attitude_error"][physics_mask]))
        row["motion_window_negative_thrust_s"] = float(np.count_nonzero(data["physics_requested_wrench"][physics_mask, 0] < 0) * .0025)
        row["motion_window_negative_force_z_s"] = float(np.count_nonzero(data["physics_feedback_force"][physics_mask, 2] < 0) * .0025)
        summary["runs"][hz] = row
        color = "#b84332" if hz == 100 else "#126aab"
        for index, (clock, name, label) in enumerate(fields, 1):
            values = data[name]
            if name == "physics_feedback_force":
                values = values[:, 2]
            elif name == "physics_requested_wrench":
                values = values[:, 0]
            elif name == "physics_states":
                values = np.linalg.norm(values[:, 10:13], axis=1)
            figure.add_trace(go.Scatter(x=data[clock], y=values, mode="lines", name=f"{hz} Hz",
                             legendgroup=str(hz), showlegend=index == 1, line=dict(color=color, width=1.3)),
                             row=index, col=1)
            figure.update_yaxes(title_text=label, row=index, col=1)
    figure.update_xaxes(range=[max(0., onset - .3), reports[100]["duration_sim_s"]])
    figure.update_xaxes(title_text="Simulation time (s)", row=7, col=1)
    figure.add_vline(x=end, line_dash="dash", line_color="#555")
    figure.update_layout(height=1500, width=1200, template="plotly_white",
                         title="Fixed 100 Hz reference: Newton wrench update at 100 vs 400 Hz")
    figure.write_html(args.directory / "comparison.html", include_plotlyjs=True)
    (args.directory / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"window_s": [onset, end], "runs": {
        hz: {key: value for key, value in row.items() if key.startswith("motion_window") or key.startswith("first_")}
        for hz, row in summary["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
