import numpy as np
from scipy.spatial.transform import Rotation
from se23 import SE23, imu_propagate


class IMUData:
    def __init__(self):
        self.t = None       # times кадров (N,)       — общий clock
        self.acc_t = None   # times IMU acc (M,)
        self.gyr_t = None   # times IMU gyr (M,)
        self.mag_t = None   # times mag   (K,)
        self.acc = None     # (M, 3) m/s²
        self.gyr = None     # (M, 3) rad/s
        self.mag = None     # (K, 3) uT
        self.g_world = np.array([0.0, 0.0, -9.81])

        # Плотности шума (σ/√Hz). ЗАГЛУШКИ: заменить на паспорт/Allan variance.
        self.sigma_a  = 2.0e-3   # acc,  m/s²/√Hz
        self.sigma_g  = 1.7e-4   # gyr,  rad/s/√Hz
        self.sigma_ba = 3.0e-3   # acc bias random walk
        self.sigma_bg = 1.9e-5   # gyr bias random walk

    @property
    def sigmas(self):
        return np.array([self.sigma_a, self.sigma_g, self.sigma_ba, self.sigma_bg])

    def segment(self, t0, t1):
        """Общая сетка времён + интерполированные acc/gyr на ней. Без состояния."""
        ts_acc, a_seg = self._slice(self.acc_t, self.acc, t0, t1)
        ts_gyr, w_seg = self._slice(self.gyr_t, self.gyr, t0, t1)
        ts = np.unique(np.concatenate([ts_acc, ts_gyr]))
        a_grid = np.stack([np.interp(ts, ts_acc, a_seg[:, i]) for i in range(3)], axis=1)
        w_grid = np.stack([np.interp(ts, ts_gyr, w_seg[:, i]) for i in range(3)], axis=1)
        return ts, a_grid, w_grid

    def load(self, t, acc, gyr, mag):
        self.t = np.loadtxt(t).ravel() * 1e-9
        self.acc, self.acc_t = np.split(np.loadtxt(acc, delimiter=','), [3], axis=1)
        self.gyr, self.gyr_t = np.split(np.loadtxt(gyr, delimiter=','), [3], axis=1)
        self.mag, self.mag_t = np.split(np.loadtxt(mag, delimiter=','), [3], axis=1)
        self.acc_t, self.gyr_t, self.mag_t = ((timestamp[:, 0] * 1e-9) for timestamp in [self.acc_t, self.gyr_t, self.mag_t])

    def mask_frames(self, idx):
        self.t = self.t[np.asarray(idx)]

    def init_orientation_from_gravity(self, n_first=50):
        # R0 такая, что R0 @ g_imu ≈ g_world. Для первой пары — обязательно.
        a0 = np.mean(self.acc[:n_first], axis=0)
        a0_n = a0 / (np.linalg.norm(a0) + 1e-12)
        g_n  = self.g_world / np.linalg.norm(self.g_world)
        R, _ = Rotation.align_vectors(-g_n[None, :], a0_n[None, :])

        # self check
        a = self.acc[:n_first] @ R.as_matrix().T + self.g_world
        assert np.median(np.linalg.norm(a, axis=1)) < 0.4, 'acceleration at inicialization is too high'

        return R.as_matrix()

    def _slice(self, ts, vals, t0, t1):
        assert t0 < t1, 'Reverse in time direction'
        assert len(ts) > 100, 'IMU data is not enough'

        # внутренние точки
        mask = (ts > t0) & (ts < t1)
        ts_in = ts[mask]
        v_in = vals[mask]

        # интерполяция на границах
        v0 = np.array([np.interp(t0, ts, vals[:, i]) for i in range(3)])
        v1 = np.array([np.interp(t1, ts, vals[:, i]) for i in range(3)])

        ts_out = np.concatenate([[t0], ts_in, [t1]])
        v_out  = np.concatenate([v0[None], v_in, v1[None]])
        return ts_out, v_out

    def preintegrate(self, t0, t1, R0, v0, bg=None, ba=None, g_world=None):
        """LEGACY-обёртка над той же пропагацией SE_2(3), что использует IEKF.

        Возвращает позу j в i и приращение скорости, всё в осях body(t0):
            dR: X_body(t1) -> X_body(t0)   (3,3) SO(3)
            dt: центр body(t1) в осях body(t0)  (3,)
            dv: (v(t1) - v(t0)) в осях body(t0) (3,)
        """
        bg = np.zeros(3) if bg is None else np.asarray(bg, float)
        ba = np.zeros(3) if ba is None else np.asarray(ba, float)
        g_world = self.g_world if g_world is None else g_world

        ts, a_grid, w_grid = self.segment(t0, t1)
        X = SE23(R0, v0, np.zeros(3))
        for k in range(len(ts) - 1):
            dt = float(ts[k + 1] - ts[k])
            if dt <= 0:
                continue
            X = imu_propagate(X, w_grid[k] - bg, a_grid[k] - ba, dt, g_world)

        return R0.T @ X.R, R0.T @ X.p, R0.T @ (X.v - np.asarray(v0, float))

    def build_H(self, R_delta, t_delta, K, n, d_bar):
        # Homography for LK-seed
        # H: cam_i -> cam_ip1, индуцированная плоскостью с нормалью n (в cam_i)
        # и средней глубиной d_bar. Если d_bar недостоверен — fallback на чистое вращение.
        K_inv = np.linalg.inv(K)
        R_delta = np.asarray(R_delta, dtype=float).reshape(3, 3)
        t_delta = np.asarray(t_delta, dtype=float).reshape(3, 1)

        if d_bar is None or not np.isfinite(d_bar) or d_bar <= 1e-6:
            return K @ R_delta @ K_inv

        n = np.asarray(n, dtype=float).reshape(3, 1)
        H = K @ (R_delta + t_delta @ n.T / d_bar) @ K_inv
        return H