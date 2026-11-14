import numpy as np
from scipy.spatial.transform import Rotation

from isaac_pathcover.simulation.scene import Scene


def test_lidar_mount_follows_body_attitude():
    scene = Scene("medium")
    quaternion = np.array([np.sqrt(.5), 0., np.sqrt(.5), 0.])
    origin, points = scene.lidar(scene.start, quaternion, horizontal_rays=8, vertical_rays=2)
    np.testing.assert_allclose(origin, scene.start + [.25, 0., 0.], atol=1e-15)
    assert np.isfinite(points).all()
    assert scene.contains_obstacle(points).all()


def test_rotated_rays_match_rotated_scene():
    # Exact 90-degree rotation of a deliberately asymmetric voxel world.
    scene = object.__new__(Scene)
    scene.lower = np.array([-2., -2., -2.])
    scene.resolution = .1
    scene.occupied = np.zeros((40, 40, 40), dtype=bool)
    scene.occupied[30:33, 5:35, 5:35] = True
    scene.occupied[4:25, 30:34, 2:30] = True
    position = np.array([.15, .25, .05])
    origin, points = scene.lidar(position, [1., 0., 0., 0.], horizontal_rays=12, vertical_rays=4)
    scene.occupied = np.rot90(scene.occupied, axes=(0, 1))
    q = np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)])
    r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    rotated_origin, rotated_points = scene.lidar(r @ position, q, horizontal_rays=12, vertical_rays=4)
    np.testing.assert_allclose(rotated_origin, r @ origin, atol=1e-14)
    np.testing.assert_allclose(rotated_points, points @ r.T, atol=1e-12)
    assert scene.contains_obstacle(rotated_points).all()
