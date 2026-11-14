"""Self-contained Plotly output of the actual scenario, plans and Newton trace."""
import numpy as np
from scipy.spatial import ConvexHull

from isaac_pathcover.planning.geometry import vertices


def save_preview(path, scene, states, initial_corridor, initial_trajectory):
    import plotly.graph_objects as go
    offsets = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                          [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    coords = (scene.box_lower[:, None] + offsets[None] * (scene.box_upper - scene.box_lower)[:, None]).reshape(-1, 3)
    triangles = (triangles[None] + 8 * np.arange(len(scene.boxes))[:, None, None]).reshape(-1, 3)
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                           i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                           opacity=.25, color="#8794a7", name="原始障碍实体", showlegend=True))
    for index, (a, b) in enumerate(initial_corridor.polytopes):
        verts = vertices(a, b)
        faces = ConvexHull(verts).simplices
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                               i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=.13,
                               color="#20b8cd", name=f"初始走廊 {index + 1}", showlegend=True))
    reference = initial_trajectory.evaluate(np.linspace(0., initial_trajectory.duration, 500))
    fig.add_trace(go.Scatter3d(x=reference[:, 0], y=reference[:, 1], z=reference[:, 2], mode="lines",
                              line=dict(color="#26d7e8", width=5), name="初始 MINCO 轨迹"))
    if len(states):
        positions = states[:, :3].copy()
        positions[~np.isfinite(positions).all(axis=1)] = np.nan
        fig.add_trace(go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="lines",
                                  line=dict(color="#ff8c42", width=6), name="Newton 实际飞行"))
    for label, position, color in [("起点", scene.start, "#36dd90"), ("终点", scene.goal, "#ff578a")]:
        fig.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]], mode="markers+text",
                                  text=[label], marker=dict(color=color, size=6), name=label))
    fig.update_layout(title=f"Isaac-PathCover · {scene.name} · 实际地图与飞行结果", template="plotly_dark",
                      scene=dict(aspectmode="data", xaxis_title="x / m", yaxis_title="y / m", zaxis_title="z / m"),
                      margin=dict(l=0, r=0, t=55, b=0), height=850)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
