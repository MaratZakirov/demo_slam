"""Минимальный SE(3) как элемент группы Ли.

Конвенция: T_a_b переводит координаты из b в a;  T_a_b @ T_b_c == T_a_c.
Хранение: 4x4 (для ~1e2 композиций дрейф ~1e-13; .normalized() на всякий случай).
Tangent xi = [rho(3), phi(3)] — сначала трансляция, потом вращение (как в Sophus).
Perturbation правая (body): X.boxplus(xi) = X @ exp(xi).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _hat(w):
    x, y, z = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

def so3_left_jacobian(phi):
    """V(φ): в exp_SE3 трансляция t = V·ρ. Ряд при малых углах."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    W = _hat(phi)
    if th < 1e-6:
        B, C = .5 - th**2 / 24., 1. / 6. - th**2 / 120.
    else:
        B = (1. - np.cos(th)) / th**2
        C = (th - np.sin(th)) / th**3
    return np.eye(3) + B * W + C * (W @ W)


def so3_left_jacobian_inv(phi):
    phi = np.asarray(phi, dtype=float).reshape(3)
    th = np.linalg.norm(phi)
    W = _hat(phi)
    c = 1. / 12. + th**2 / 720. if th < 1e-4 else \
        (1. - .5 * th / np.tan(.5 * th)) / th**2
    return np.eye(3) - .5 * W + c * (W @ W)

class SE3:
    __slots__ = ("_T",)

    def __init__(self, R=None, t=None):
        """SE3() = identity. t принимается и (3,), и (3,1) — reshape больше не нужен."""
        self._T = np.eye(4)
        if R is not None:
            self._T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
        if t is not None:
            self._T[:3, 3] = np.asarray(t, dtype=float).reshape(3)

    # --- конструкторы / доступ ---
    @classmethod
    def from_matrix(cls, T):
        T = np.asarray(T, dtype=float)
        return cls(T[:3, :3], T[:3, 3])

    @property
    def R(self):        return self._T[:3, :3]
    @property
    def t(self):        return self._T[:3, 3]          # (3,)
    @property
    def matrix(self):   return self._T                 # (4, 4)

    def as_Rt(self):
        """LEGACY-адаптер [R(3,3), t(3,1)] для pair/ba/debug. Убрать после их переезда."""
        return [self.R, self.t.reshape(3, 1)]

    # --- группа ---
    def __matmul__(self, other):
        if isinstance(other, SE3):
            return SE3.from_matrix(self._T @ other._T)
        return NotImplemented

    def inverse(self):
        return SE3(self.R.T, -self.R.T @ self.t)

    # --- действие на точки/векторы: (3,) и (N,3) ---
    def act(self, X):
        return np.asarray(X, dtype=float) @ self.R.T + self.t

    def rotate(self, v):
        """Только поворот, без трансляции — для скоростей, нормалей, приращений."""
        return np.asarray(v, dtype=float) @ self.R.T

    # --- Ли: нужно для IEKF ---
    @classmethod
    def exp(cls, xi):
        xi = np.asarray(xi, dtype=float).reshape(6)
        phi = xi[3:]
        return cls(Rotation.from_rotvec(phi).as_matrix(),
                   so3_left_jacobian(phi) @ xi[:3])

    def log(self):
        phi = Rotation.from_matrix(self.R).as_rotvec()
        return np.concatenate([so3_left_jacobian_inv(phi) @ self.t, phi])

    def Adjoint(self):
        """Ad(T) @ xi == (T @ exp(xi) @ T.inverse()).log()"""
        A = np.zeros((6, 6))
        A[:3, :3] = A[3:, 3:] = self.R
        A[:3, 3:] = _hat(self.t) @ self.R
        return A

    def boxplus(self, xi):
        return self @ SE3.exp(xi)

    def boxminus(self, other):
        return (other.inverse() @ self).log()

    # --- гигиена ---
    def normalized(self):
        U, _, Vt = np.linalg.svd(self.R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return SE3(R, self.t)

    def __repr__(self):
        ang = np.degrees(np.linalg.norm(self.log()[3:]))
        return f"SE3(t={self.t.round(4)}, rot={ang:.2f}deg)"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rnd = lambda: SE3(Rotation.from_rotvec(rng.normal(size=3)).as_matrix(),
                      rng.normal(size=3))
    T, T2, X = rnd(), rnd(), rng.normal(size=(5, 3))
    xi = rng.normal(size=6) * .1
    assert np.allclose((T @ T2).act(X), T.act(T2.act(X)))
    assert np.allclose(T.inverse().act(T.act(X)), X)
    assert np.allclose(SE3.exp(T.log()).matrix, T.matrix)
    assert np.allclose(T.boxplus(xi).boxminus(T), xi)
    assert np.allclose(T.Adjoint() @ xi, (T @ SE3.exp(xi) @ T.inverse()).log())
    assert np.allclose(SE3.exp([1, 2, 3, 0, 0, 0]).t, [1, 2, 3])
    print("SE3 ok")