import numpy as np
from scipy.spatial.transform import Rotation


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

    def load(self, t, acc, gyr, mag):
        self.t = np.loadtxt(t).ravel() * 1e-9
        self.acc, self.acc_t = np.split(np.loadtxt(acc, delimiter=','), [3], axis=1)
        self.gyr, self.gyr_t = np.split(np.loadtxt(gyr, delimiter=','), [3], axis=1)
        self.mag, self.mag_t = np.split(np.loadtxt(mag, delimiter=','), [3], axis=1)
        self.acc_t, self.gyr_t, self.mag_t = ((timestamp * 1e-9) for timestamp in [self.acc_t, self.gyr_t, self.mag_t])

    def mask_frames(self, idx):
        self.t = self.t[np.asarray(idx)]

    def init_orientation_from_gravity(self, n_first=50):
        """R0 такая, что R0 @ g_imu ≈ g_world. Для первой пары — обязательно."""
        a0 = np.mean(self.acc[:n_first], axis=0)
        a0_n = a0 / (np.linalg.norm(a0) + 1e-12)
        g_n  = self.g_world / np.linalg.norm(self.g_world)
        R, _ = Rotation.align_vectors(g_n[None, :], a0_n[None, :])
        return R.as_matrix()

    # ---------- срез с интерполяцией на границах ----------
    def _slice(self, ts, vals, t0, t1):
        if t1 <= t0:
            return np.array([t0]), vals[:1]
        if len(ts) < 2:
            return np.array([t0]), vals[:1]

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

    # ---------- предынтегрирование между двумя моментами ----------
    def preintegrate(self, t0, t1, R0=None, v0=None, bg=None, ba=None, g_world=None):
        """
        Возвращает относительное движение IMU-фрейма между t0 и t1:
            dR : X_body(t0) -> X_body(t1)   (3,3) SO(3)
            dt : сдвиг в системе body(t0)    (3,)
            dv : приращение скорости в body(t0) (3,)   (опционально)
        Интегрирование — в world-frame с явной гравитацией, потом проекция в body(t0).
        """
        if R0 is None:
            R0 = np.eye(3)
        if v0 is None:
            v0 = np.zeros(3)
        if bg is None:
            bg = np.zeros(3)
        if ba is None:
            ba = np.zeros(3)
        if g_world is None:
            g_world = self.g_world

        ts_acc, a_seg = self._slice(self.acc_t, self.acc, t0, t1)
        ts_gyr, w_seg = self._slice(self.gyr_t, self.gyr, t0, t1)

        # общая сетка: берём объединение времён
        ts = np.unique(np.concatenate([ts_acc, ts_gyr]))
        # интерполируем оба сигнала на эту сетку
        a_grid = np.stack([np.interp(ts, ts_acc, a_seg[:, i]) for i in range(3)], axis=1)
        w_grid = np.stack([np.interp(ts, ts_gyr, w_seg[:, i]) for i in range(3)], axis=1)

        R = R0.copy()
        v = v0.copy()
        p = np.zeros(3)

        for k in range(len(ts) - 1):
            dt = float(ts[k + 1] - ts[k])
            if dt <= 0:
                continue
            a_k = a_grid[k] - ba          # ускорение в body
            w_k = w_grid[k] - bg          # угловая скорость в body

            a_world = R @ a_k + g_world
            p = p + v * dt + 0.5 * a_world * dt * dt
            v = v + a_world * dt
            R = R @ Rotation.from_rotvec(w_k * dt).as_matrix()

        # проекция в body(t0)
        R0T = R0.T
        dR = R0T @ R
        dt_delta = R0T @ p
        dv = R0T @ (v - v0)
        return dR, dt_delta, dv

    # ---------- гомография для LK-seed ----------
    def build_H(self, R_delta, t_delta, K, n, d_bar):
        """
        H: cam_i -> cam_ip1, индуцированная плоскостью с нормалью n (в cam_i)
        и средней глубиной d_bar. Если d_bar недостоверен — fallback на чистое вращение.
        """
        K_inv = np.linalg.inv(K)
        R_delta = np.asarray(R_delta, dtype=float).reshape(3, 3)
        t_delta = np.asarray(t_delta, dtype=float).reshape(3, 1)

        if d_bar is None or not np.isfinite(d_bar) or d_bar <= 1e-6:
            return K @ R_delta @ K_inv

        n = np.asarray(n, dtype=float).reshape(3, 1)
        H = K @ (R_delta + t_delta @ n.T / d_bar) @ K_inv
        return H