"""SE_2(3) — расширенная поза (R, v, p) для INS.

Группа SO(3) ⋉ R⁶ (R действует на оба вектора):
    X = [[R, v, p], [0,1,0], [0,0,1]]
    X1 @ X2     = (R1 R2, v1 + R1 v2, p1 + R1 p2)
    X.inverse() = (Rᵀ, −Rᵀ v, −Rᵀ p)

Поза: R = body → world, v и p — в world.
xi = [rho_p(3), rho_v(3), phi(3)];  perturbation правая: X.boxplus(xi) = X @ exp(xi).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from se3 import _hat, so3_left_jacobian, so3_left_jacobian_inv


class SE23:
    __slots__ = ("_X",)

    def __init__(self, R=None, v=None, p=None):
        X = np.zeros((5, 5))
        X[3, 3] = X[4, 4] = 1.0
        X[:3, :3] = np.eye(3) if R is None else np.asarray(R, float).reshape(3, 3)
        if v is not None:
            X[:3, 3] = np.asarray(v, float).reshape(3)
        if p is not None:
            X[:3, 4] = np.asarray(p, float).reshape(3)
        self._X = X

    @classmethod
    def from_matrix(self, M):
        M = np.asarray(M, float)
        return self(M[:3, :3], M[:3, 3], M[:3, 4])

    @property
    def R(self):      return self._X[:3, :3]
    @property
    def v(self):      return self._X[:3, 3]
    @property
    def p(self):      return self._X[:3, 4]
    @property
    def matrix(self): return self._X

    # --- группа ---
    def __matmul__(self, other):
        if isinstance(other, SE23):
            return SE23.from_matrix(self._X @ other._X)   # 5x5 произведение = операция группы
        return NotImplemented

    def inverse(self):
        Rt = self.R.T
        return SE23(Rt, -Rt @ self.v, -Rt @ self.p)

    # --- Ли ---
    @classmethod
    def exp(cls, xi):
        xi = np.asarray(xi, float).reshape(9)
        phi = xi[6:]
        V = so3_left_jacobian(phi)
        return cls(Rotation.from_rotvec(phi).as_matrix(), V @ xi[3:6], V @ xi[:3])

    def log(self):
        phi = Rotation.from_matrix(self.R).as_rotvec()
        Vi = so3_left_jacobian_inv(phi)
        return np.concatenate([Vi @ self.p, Vi @ self.v, phi])

    def Adjoint(self):
        """Ad(X)·xi == (X @ exp(xi) @ X.inverse()).log()"""
        A = np.zeros((9, 9))
        A[0:3, 0:3] = A[3:6, 3:6] = A[6:9, 6:9] = self.R
        A[0:3, 6:9] = _hat(self.p) @ self.R
        A[3:6, 6:9] = _hat(self.v) @ self.R
        return A

    def boxplus(self, xi):
        return self @ SE23.exp(xi)

    def boxminus(self, other):
        return (other.inverse() @ self).log()

    # --- кирпичики пропагации INS ---
    def shear(self, dt):
        """S_dt: p += v·dt. Автоморфизм группы, не элемент группы."""
        return SE23(self.R, self.v, self.p + self.v * dt)

    @classmethod
    def from_gravity(cls, g_world, dt):
        """Γ(dt) — гравитация, входит СЛЕВА (world)."""
        g = np.asarray(g_world, float).reshape(3)
        return cls(None, g * dt, .5 * g * dt * dt)

    @classmethod
    def from_imu(cls, w, a, dt):
        """U(ω,a,dt) — сэмпл IMU (уже без биасов), входит СПРАВА (body)."""
        return cls(Rotation.from_rotvec(np.asarray(w, float) * dt).as_matrix(),
                   np.asarray(a, float) * dt, .5 * np.asarray(a, float) * dt * dt)

    def __repr__(self):
        ang = np.degrees(np.linalg.norm(self.log()[6:]))
        return f"SE23(p={self.p.round(3)}, |v|={np.linalg.norm(self.v):.3f}, rot={ang:.2f}deg)"


def imu_propagate(X, w, a, dt, g_world):
    """Один шаг INS: X ← Γ(dt) · S_dt(X) · U(ω,a,dt).

    Численно это в точности эйлеровская схема из старого preintegrate:
        a_w = R a + g;  p += v dt + ½ a_w dt²;  v += a_w dt;  R = R Exp(ω dt)
    """
    return SE23.from_gravity(g_world, dt) @ X.shear(dt) @ SE23.from_imu(w, a, dt)


if __name__ == "__main__":
    from se3 import SE3
    rng = np.random.default_rng(0)
    rnd = lambda: SE23(Rotation.from_rotvec(rng.normal(size=3)).as_matrix(),
                       rng.normal(size=3), rng.normal(size=3))
    A, B = rnd(), rnd()
    assert np.allclose((A @ B).matrix, A.matrix @ B.matrix)
    assert np.allclose((A @ A.inverse()).matrix, np.eye(5))
    assert np.allclose(SE23.exp(A.log()).matrix, A.matrix, atol=1e-12)
    xi = rng.normal(size=9) * 1e-4
    assert np.allclose(A.Adjoint() @ xi, (A @ SE23.exp(xi) @ A.inverse()).log(), atol=1e-7)
    assert np.allclose(A.boxplus(xi).boxminus(A), xi, atol=1e-12)

    # SE(3) вкладывается как (R, 0, p) — согласованность с se3.py
    T = SE3(A.R, A.p)                      # A — SE23, здесь .p корректен
    assert np.allclose(SE23(T.R, None, T.t).log()[[0, 1, 2, 6, 7, 8]], T.log(), atol=1e-12)

    # групповая пропагация == старый цикл Эйлера
    R0 = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    v0, g = rng.normal(size=3), np.array([0., 0., -9.81])
    ts = np.cumsum(np.r_[0., rng.uniform(.002, .01, size=200)])
    a_grid, w_grid = rng.normal(size=(200, 3)) * 2, rng.normal(size=(200, 3)) * .5
    X = SE23(R0, v0, np.zeros(3))
    R, v, p = R0.copy(), v0.copy(), np.zeros(3)
    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]
        X = imu_propagate(X, w_grid[k], a_grid[k], dt, g)
        aw = R @ a_grid[k] + g
        p = p + v * dt + .5 * aw * dt * dt
        v = v + aw * dt
        R = R @ Rotation.from_rotvec(w_grid[k] * dt).as_matrix()
    assert np.allclose(X.R, R, atol=1e-12)
    assert np.allclose(X.v, v, atol=1e-12)
    assert np.allclose(X.p, p, atol=1e-12)
    print("SE23 ok")