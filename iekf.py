"""Error-state IEKF на SE_2(3). Этап 1: только predict (номинал + ковариация).

Состояние: X ∈ SE_2(3) (R body→world, v/p в world) + биасы ba, bg.
Ошибка (15): δx = [δp, δv, δθ, δba, δbg] — в осях тела, X_true = X̂.boxplus(δx[:9]).
Шум (12):    n = [n_a, n_g, n_ba, n_bg] — непрерывные плотности.

Непрерывная динамика ошибки (гравитация сокращается аналитически!):
    δṗ  = −[ω]ₓ δp + δv
    δv̇  = −[a]ₓ δθ − [ω]ₓ δv − δb_a − n_a
    δθ̇  = −[ω]ₓ δθ − δb_g − n_g
    δḃ   = n_b
F зависит только от показаний IMU и dt, но не от оценки — в этом весь смысл body-ошибки.
Дискретизация первым порядком (F = I + F_c dt) согласована с эйлеровским номиналом.
"""
import numpy as np

from se3 import _hat
from se23 import SE23, imu_propagate


class IEKF:
    N = 15   # размер ошибки
    M = 12   # размер шума

    def __init__(self, X, t0, ba=None, bg=None, P=None, sigmas=None,
                 g_world=np.array([0., 0., -9.81])):
        self.X = X
        self.t = float(t0)
        self.ba = np.zeros(3) if ba is None else np.asarray(ba, float).reshape(3)
        self.bg = np.zeros(3) if bg is None else np.asarray(bg, float).reshape(3)
        self.g = np.asarray(g_world, float).reshape(3)
        self.P = self.default_P() if P is None else np.asarray(P, float).copy()

        s = np.array([2.0e-3, 1.7e-4, 3.0e-3, 1.9e-5]) if sigmas is None \
            else np.asarray(sigmas, float).reshape(4)
        self.Qc = np.diag(np.repeat(s**2, 3))

        # G от сэмпла не зависит — строим один раз
        self.G = np.zeros((self.N, self.M))
        self.G[3:6, 0:3] = -np.eye(3)     # n_a → δv
        self.G[6:9, 3:6] = -np.eye(3)     # n_g → δθ
        self.G[9:12, 6:9] = np.eye(3)     # n_ba
        self.G[12:15, 9:12] = np.eye(3)   # n_bg
        self.GQGt = self.G @ self.Qc @ self.G.T

    @staticmethod
    def default_P():
        """p — начало world зафиксировано (0). roll/pitch выставлены по g, yaw ненаблюдаем."""
        d = np.zeros(15)
        d[0:3] = 0.0                       # p: gauge
        d[3:6] = 0.5**2                    # v
        d[6:8] = np.radians(2.0)**2        # roll/pitch
        d[8] = np.radians(30.0)**2         # yaw
        d[9:12] = 0.1**2                   # ba
        d[12:15] = 0.01**2                 # bg
        return np.diag(d)

    def F(self, w_hat, a_hat, dt):
        W, A = _hat(w_hat), _hat(a_hat)
        I3 = np.eye(3)
        F = np.eye(self.N)
        F[0:3, 0:3] = I3 - W * dt
        F[0:3, 3:6] = I3 * dt
        F[3:6, 3:6] = I3 - W * dt
        F[3:6, 6:9] = -A * dt
        F[3:6, 9:12] = -I3 * dt
        F[6:9, 6:9] = I3 - W * dt
        F[6:9, 12:15] = -I3 * dt
        return F

    def predict(self, ts, a_grid, w_grid):
        """Шаг predict: номинал на группе + P ← F P Fᵀ + G Q Gᵀ dt."""
        for k in range(len(ts) - 1):
            dt = float(ts[k + 1] - ts[k])
            if dt <= 0:
                continue
            a_hat = a_grid[k] - self.ba
            w_hat = w_grid[k] - self.bg
            F = self.F(w_hat, a_hat, dt)
            self.P = F @ self.P @ F.T + self.GQGt * dt
            self.X = imu_propagate(self.X, w_hat, a_hat, dt, self.g)
        self.P = .5 * (self.P + self.P.T)
        self.t = float(ts[-1])
        return self.X

    def predict_to(self, t1, imu):
        return self.predict(*imu.segment(self.t, t1))

    # TODO этап 2: update. Измерение — относительная поза T_ij в осях камеры,
    # нужен клон предыдущей позы в состоянии (MSCKF) либо перепроецирование.


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(1)

    X0 = SE23(Rotation.from_rotvec(rng.normal(size=3) * .3).as_matrix(),
              rng.normal(size=3), rng.normal(size=3))
    ba, bg = rng.normal(size=3) * .05, rng.normal(size=3) * .01
    w_raw, a_raw = rng.normal(size=3), rng.normal(size=3) * 3
    g = np.array([0., 0., -9.81])
    dt = 0.01

    f = IEKF(X0, 0.0, ba=ba, bg=bg, sigmas=[2e-3, 1.7e-4, 3e-3, 1.9e-5])
    F_an = f.F(w_raw - bg, a_raw - ba, dt)

    # численный якобиан пропагации в тех же координатах ошибки
    eps = 1e-6
    wh, ah = w_raw - bg, a_raw - ba
    Xe = imu_propagate(X0, wh, ah, dt, g)
    F_num = np.eye(15)
    for j in range(9):
        e = np.zeros(9); e[j] = eps
        Xp = imu_propagate(X0.boxplus(e), wh, ah, dt, g)
        F_num[:9, j] = (Xe.inverse() @ Xp).log() / eps
    for base, is_a in ((9, True), (12, False)):
        for c in range(3):
            db = np.zeros(3); db[c] = eps
            Xp = imu_propagate(X0, w_raw - (bg + (0 if is_a else db)),
                               a_raw - (ba + (db if is_a else 0)), dt, g)
            F_num[:9, base + c] = (Xe.inverse() @ Xp).log() / eps

    err = np.abs(F_an - F_num).max()
    print(f"F: max|analytic - numeric| = {err:.2e}")
    assert err < 2e-3   # остаток ~½dt²|a| — отброшенный Эйлером 2-й порядок, не ошибка

    # P растёт, симметрична, PSD
    ts = np.cumsum(np.r_[0., np.full(300, 0.005)])
    a_grid = np.tile(a_raw, (301, 1)) + rng.normal(scale=.05, size=(301, 3))
    w_grid = np.tile(w_raw, (301, 1)) + rng.normal(scale=.01, size=(301, 3))
    tr0 = np.trace(f.P)
    f.predict(ts, a_grid, w_grid)
    assert np.trace(f.P) > tr0
    assert np.allclose(f.P, f.P.T)
    assert np.linalg.eigvalsh(f.P).min() > -1e-12
    print("IEKF predict ok")