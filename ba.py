import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from skimage.data import camera


def bundle_adjust(tm, cam_poses, K,
                  loss='huber', f_scale=1.0,
                  fix_camera0=True, verbose=1, max_nfev=10000):
    """
    Оптимизирует позы камер (кроме камеры 0) и 3D-точки треков,
    минимизируя ошибку репроекции всех наблюдений из tm.

    Конвенция позы:
        X_world = R_ito0 @ X_cam + t_ito0.
    Проекция:
        X_cam = R_ito0.T @ X_world - R_ito0.T @ t_ito0, затем π(K, ·).
    """
    # ---- 1. Собираем наблюдения ----
    obs_cam = []
    obs_uv  = []
    obs_pt  = []
    X_init  = []
    tid_to_idx = {}

    for tid, observations in tm.obs.items():
        if len(observations) < 2:
            continue
        X = tm.X.get(tid)
        if X is None:
            continue
        X = np.asarray(X).ravel()
        if not np.all(np.isfinite(X)):
            continue
        tid_to_idx[tid] = len(X_init)
        X_init.append(X)
        for cam, u, v in observations:
            obs_cam.append(int(cam))
            obs_uv.append((float(u), float(v)))
            obs_pt.append(tid_to_idx[tid])

    obs_cam = np.asarray(obs_cam, dtype=int)
    obs_uv  = np.asarray(obs_uv, dtype=float)
    obs_pt  = np.asarray(obs_pt, dtype=int)
    X_init  = np.asarray(X_init, dtype=float)

    n_obs  = len(obs_cam)
    n_pts  = len(X_init)
    n_cams = len(cam_poses)

    if n_obs == 0 or n_pts == 0:
        print("[BA] Нет наблюдений или точек. Ничего не делаем.")
        return cam_poses, tm, None

    if verbose:
        print(f"[BA] cams={n_cams}  points={n_pts}  obs={n_obs}  "
              f"obs/pt={n_obs / n_pts:.2f}")

    # ---- 2. Какие камеры оптимизируем ----
    if fix_camera0:
        cam_opt = list(range(1, n_cams))
    else:
        cam_opt = list(range(n_cams))
    cam_idx = {c: i for i, c in enumerate(cam_opt)}
    n_cam_params = 6 * len(cam_opt)

    # ---- 3. x0 ----
    x0 = np.zeros(n_cam_params + 3 * n_pts)
    for c in cam_opt:
        i = cam_idx[c]
        R_i, t_i = cam_poses[c]
        x0[6*i : 6*i+3] = Rotation.from_matrix(R_i).as_rotvec()
        x0[6*i+3 : 6*i+6] = np.asarray(t_i).ravel()
    x0[n_cam_params:] = X_init.ravel()

    # ---- 4. unpack ----
    def unpack(x):
        cams = {}
        for c in cam_opt:
            i = cam_idx[c]
            rv = x[6*i : 6*i+3]
            tv = x[6*i+3 : 6*i+6]
            R = Rotation.from_rotvec(rv).as_matrix()
            cams[c] = (R, tv.reshape(3, 1))
        if 0 not in cams:
            cams[0] = (np.asarray(cam_poses[0][0]),
                       np.asarray(cam_poses[0][1]).reshape(3, 1))
        pts = x[n_cam_params:].reshape(n_pts, 3)
        return cams, pts

    # ---- 5. Residuals + аналитический якобиан ----
    K = np.asarray(K, dtype=float)
    n_res  = 2 * n_obs
    n_vars = n_cam_params + 3 * n_pts

    def residuals_and_jac(x):
        cams, pts = unpack(x)

        # Позы в world->cam
        R_wc = np.empty((n_cams, 3, 3))
        t_wc = np.empty((n_cams, 3))
        for c in range(n_cams):
            R_v, t_v = cams[c]
            R_v = np.asarray(R_v)
            t_v = np.asarray(t_v).reshape(3,)
            R_wc[c] = R_v.T
            t_wc[c] = -R_v.T @ t_v

        R_o = R_wc[obs_cam]          # (M, 3, 3)
        t_o = t_wc[obs_cam]          # (M, 3)
        X_o = pts[obs_pt]            # (M, 3)

        Xc = np.einsum('mij,mj->mi', R_o, X_o) + t_o      # (M, 3)
        u_h = Xc @ K.T                                     # (M, 3)
        u_hat = u_h[:, :2] / u_h[:, 2:3]
        r = (u_hat - obs_uv).ravel()

        # ----- Jacobian -----
        X_, Y_, Z_ = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        # защита от деления на ~0 (точка за/на камере)
        Z_safe = np.where(np.abs(Z_) > 1e-9, Z_, 1e-9)
        fx, fy = K[0, 0], K[1, 1]
        invZ  = 1.0 / Z_safe
        invZ2 = invZ * invZ

        # d(π)/d(Xc): (M, 2, 3)
        dpi = np.zeros((n_obs, 2, 3))
        dpi[:, 0, 0] =  fx * invZ
        dpi[:, 0, 2] = -fx * X_ * invZ2
        dpi[:, 1, 1] =  fy * invZ
        dpi[:, 1, 2] = -fy * Y_ * invZ2

        # d(Xc)/d(X_world) = R_wc  →  J_pt = dπ @ R_wc
        J_pt = np.einsum('mij,mjk->mik', dpi, R_o)         # (M, 2, 3)

        # d(Xc)/d(t_v) = -R_wc
        J_t  = -J_pt

        # d(Xc)/d(δ_rotvec) = [Xc]_×
        skew = np.zeros((n_obs, 3, 3))
        skew[:, 0, 1] = -Z_
        skew[:, 0, 2] =  Y_
        skew[:, 1, 0] =  Z_
        skew[:, 1, 2] = -X_
        skew[:, 2, 0] = -Y_
        skew[:, 2, 1] =  X_
        J_rot = np.einsum('mij,mjk->mik', dpi, skew)       # (M, 2, 3)

        # ---- Сборка разреженной матрицы ----
        rows_u = 2 * np.arange(n_obs)
        rows_v = rows_u + 1

        cols_p = n_cam_params + 3 * obs_pt[:, None] + np.arange(3)[None, :]

        has_cam = np.array([c in cam_idx for c in obs_cam])
        idx_c   = np.where(has_cam)[0]
        K_c     = len(idx_c)

        rows_list = []
        cols_list = []
        vals_list = []

        # Вклад точки
        rows_list.append(np.repeat(rows_u, 3))
        cols_list.append(cols_p.ravel())
        vals_list.append(J_pt[:, 0, :].ravel())

        rows_list.append(np.repeat(rows_v, 3))
        cols_list.append(cols_p.ravel())
        vals_list.append(J_pt[:, 1, :].ravel())

        # Вклад камеры
        if K_c > 0:
            cam_off = np.array([cam_idx[obs_cam[i]] for i in idx_c])
            cols_c_rot = 6 * cam_off[:, None] + np.arange(3)[None, :]
            cols_c_t   = 6 * cam_off[:, None] + 3 + np.arange(3)[None, :]

            J_rot_c = J_rot[idx_c]
            J_t_c   = J_t[idx_c]
            rows_c_u = 2 * idx_c
            rows_c_v = rows_c_u + 1

            rows_list.append(np.repeat(rows_c_u, 3))
            cols_list.append(cols_c_rot.ravel())
            vals_list.append(J_rot_c[:, 0, :].ravel())

            rows_list.append(np.repeat(rows_c_v, 3))
            cols_list.append(cols_c_rot.ravel())
            vals_list.append(J_rot_c[:, 1, :].ravel())

            rows_list.append(np.repeat(rows_c_u, 3))
            cols_list.append(cols_c_t.ravel())
            vals_list.append(J_t_c[:, 0, :].ravel())

            rows_list.append(np.repeat(rows_c_v, 3))
            cols_list.append(cols_c_t.ravel())
            vals_list.append(J_t_c[:, 1, :].ravel())

        rows_all = np.concatenate(rows_list)
        cols_all = np.concatenate(cols_list)
        vals_all = np.concatenate(vals_list)

        J = csr_matrix((vals_all, (rows_all, cols_all)),
                       shape=(n_res, n_vars))
        return r, J

    # ---- 6. Кэш для r/J, чтобы scipy не считал дважды ----
    _cache = {'x': None, 'r': None, 'J': None}

    def _compute(x):
        if _cache['x'] is None or not np.array_equal(x, _cache['x']):
            r, J = residuals_and_jac(x)
            _cache['x'] = x.copy()
            _cache['r'] = r
            _cache['J'] = J
        return _cache['r'], _cache['J']

    def fun(x):
        return _compute(x)[0]

    def jac(x):
        return _compute(x)[1]

    # ---- 7. Запуск ----
    r0 = residuals_and_jac(x0)[0]
    rms0 = float(np.sqrt(np.mean(r0 ** 2)))
    if verbose:
        print(f"[BA] initial RMS = {rms0:.3f} px")

    result = least_squares(
        fun, x0,
        jac=jac,
        loss=loss, f_scale=f_scale,
        method='trf',
        verbose=verbose,
        x_scale='jac',
        max_nfev=max_nfev,
    )

    r1 = residuals_and_jac(result.x)[0]
    rms1 = float(np.sqrt(np.mean(r1 ** 2)))
    if verbose:
        print(f"[BA] final   RMS = {rms1:.3f} px   "
              f"(improvement: {rms0 - rms1:+.3f})")

    # ---- 8. Запись результата обратно ----
    cams_opt_dict, pts_opt = unpack(result.x)

    cam_poses_new = []
    for c in range(n_cams):
        R, t = cams_opt_dict[c]
        cam_poses_new.append((R, np.asarray(t).reshape(3, 1)))

    idx_to_tid = {v: k for k, v in tid_to_idx.items()}
    for idx, tid in idx_to_tid.items():
        tm.X[int(tid)] = pts_opt[idx]

    return cam_poses_new, tm, result