"""Drag-aware flatness map and analytic reverse derivatives.

The vector expressions are the normalization and chain rule in PathCover
8723a05d, trajectory_server/include/gcopter/flatness.hpp. NumPy batches
quadrature nodes so the original map is evaluated without a Python node loop.
Copyright (c) 2021 Zhepei Wang. Distributed under the upstream MIT license.
"""
import numpy as np


def _dot(a, b):
    return np.sum(a * b, axis=-1)


class FlatnessMap:
    def __init__(self, physical_parameters=(0.716, 9.81, 0.0, 0.0, 0.0, 0.001)):
        self.mass, self.grav, self.dh, self.dv, self.cp, self.veps = physical_parameters

    def forward(self, vel, acc, jer, psi=0.0, dpsi=0.0):
        self.vel, self.acc = np.asarray(vel, dtype=float), np.asarray(acc, dtype=float)
        jer = np.asarray(jer, dtype=float)
        self.speed = np.sqrt(_dot(self.vel, self.vel) + self.veps)
        self.wterm = 1 + self.cp * self.speed
        w = self.wterm[..., None] * self.vel
        u = self.acc + (self.dh / self.mass) * w
        u = u.copy()
        u[..., 2] += self.grav
        self.norm = np.sqrt(_dot(u, u))
        self.z = u / self.norm[..., None]
        self.v_dot_a = _dot(self.vel, self.acc)
        self.dwterm = self.cp * self.v_dot_a / self.speed
        dw = self.wterm[..., None] * self.acc + self.dwterm[..., None] * self.vel
        self.du = jer + (self.dh / self.mass) * dw
        self.z_dot_du = _dot(self.z, self.du)
        self.dz = (self.du - self.z * self.z_dot_du[..., None]) / self.norm[..., None]
        self.force = self.mass * self.acc + self.dv * w
        self.force[..., 2] += self.mass * self.grav
        thrust = _dot(self.z, self.force)
        self.tilt_den = np.sqrt(2 * (1 + self.z[..., 2]))
        tilt0 = 0.5 * self.tilt_den
        tilt1, tilt2 = -self.z[..., 1] / self.tilt_den, self.z[..., 0] / self.tilt_den
        self.ch, self.sh = np.cos(0.5 * psi), np.sin(0.5 * psi)
        quat = np.stack([tilt0 * self.ch, tilt1 * self.ch + tilt2 * self.sh,
                         tilt2 * self.ch - tilt1 * self.sh, tilt0 * self.sh], axis=-1)
        self.cy, self.sy = np.cos(psi), np.sin(psi)
        self.den = 1 + self.z[..., 2]
        self.ratio = self.dz[..., 2] / self.den
        h = self.dz[..., :2] - self.z[..., :2] * self.ratio[..., None]
        self.omega = np.stack([h[..., 0] * self.sy - h[..., 1] * self.cy,
                               h[..., 0] * self.cy + h[..., 1] * self.sy,
                               (self.z[..., 1] * self.dz[..., 0]
                                - self.z[..., 0] * self.dz[..., 1]) / self.den + dpsi], axis=-1)
        return thrust, quat, self.omega

    def backward(self, pos_grad, vel_grad, thr_grad, quat_grad, omg_grad):
        z, dz = self.z, self.dz
        gh = np.stack([self.sy * omg_grad[..., 0] + self.cy * omg_grad[..., 1],
                       -self.cy * omg_grad[..., 0] + self.sy * omg_grad[..., 1]], axis=-1)
        gz = np.zeros_like(z)
        gz[..., :2] = -self.ratio[..., None] * gh
        gdz = np.zeros_like(dz)
        gdz[..., :2] = gh
        gr = -_dot(gh, z[..., :2])
        gdz[..., 2] = gr / self.den
        gden = -gr * dz[..., 2] / self.den ** 2
        temp = omg_grad[..., 2] / self.den
        gz[..., 0] -= dz[..., 1] * temp
        gz[..., 1] += dz[..., 0] * temp
        gdz[..., 0] += z[..., 1] * temp
        gdz[..., 1] -= z[..., 0] * temp
        gden -= (z[..., 1] * dz[..., 0] - z[..., 0] * dz[..., 1]) * temp / self.den
        gz[..., 2] = gden
        gpsi = omg_grad[..., 0] * self.omega[..., 1] - omg_grad[..., 1] * self.omega[..., 0]
        gt0 = self.ch * quat_grad[..., 0] + self.sh * quat_grad[..., 3]
        gt1 = self.ch * quat_grad[..., 1] - self.sh * quat_grad[..., 2]
        gt2 = self.sh * quat_grad[..., 1] + self.ch * quat_grad[..., 2]
        gz[..., 0] += gt2 / self.tilt_den
        gz[..., 1] -= gt1 / self.tilt_den
        gd = 0.5 * gt0 + (z[..., 1] * gt1 - z[..., 0] * gt2) / self.tilt_den ** 2
        gz[..., 2] += gd / self.tilt_den
        t0, t1, t2 = 0.5 * self.tilt_den, -z[..., 1] / self.tilt_den, z[..., 0] / self.tilt_den
        gpsi += 0.5 * (-self.sh * t0 * quat_grad[..., 0]
                      + (-self.sh * t1 + self.ch * t2) * quat_grad[..., 1]
                      + (-self.sh * t2 - self.ch * t1) * quat_grad[..., 2]
                      + self.ch * t0 * quat_grad[..., 3])
        gz += np.asarray(thr_grad)[..., None] * self.force
        gf = np.asarray(thr_grad)[..., None] * z
        ga = self.mass * gf
        gw = self.dv * gf
        gz -= (gdz * self.z_dot_du[..., None] + self.du * _dot(gdz, z)[..., None]) / self.norm[..., None]
        gnorm = -_dot(gdz, dz) / self.norm
        gu = (gz - z * _dot(z, gz)[..., None]) / self.norm[..., None] + z * gnorm[..., None]
        gdu = (gdz - z * _dot(z, gdz)[..., None]) / self.norm[..., None]
        ga += gu
        gw += (self.dh / self.mass) * gu
        gdw = (self.dh / self.mass) * gdu
        ga += self.wterm[..., None] * gdw
        gwt = _dot(self.acc, gdw)
        gdwt = _dot(self.vel, gdw)
        gv = self.dwterm[..., None] * gdw
        gva = self.cp * gdwt / self.speed
        gv += self.acc * gva[..., None]
        ga += self.vel * gva[..., None]
        gs = -self.v_dot_a * gva / self.speed
        gv += self.wterm[..., None] * gw
        gwt += _dot(self.vel, gw)
        gs += self.cp * gwt
        gv += self.vel * (gs / self.speed)[..., None] + vel_grad
        return np.asarray(pos_grad).copy(), gv, ga, gdu, gpsi, omg_grad[..., 2].copy()


def flatness(vel, acc, jer, yaw=0.0, yaw_rate=0.0,
             physical_parameters=(0.716, 9.81, 0.0, 0.0, 0.0, 0.001)):
    return FlatnessMap(physical_parameters).forward(vel, acc, jer, yaw, yaw_rate)

