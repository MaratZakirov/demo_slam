import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from skimage.data import camera


def bundle_adjust(tm, cam_poses, K,
                  loss='huber', f_scale=1.0,
                  fix_camera0=True, verbose=1, max_nfev=6000):
    """
    Оптимизирует позы камер (кроме камеры 0) и 3D-точки треков,
    минимизируя ошибку репроекции всех наблюдений из tm.

    Конвенция позы:
        X_world = R_ito0 @ X_cam + t_ito0.
    Проекция:
        X_cam = R_ito0.T @ X_world - R_ito0.T @ t_ito0, затем π(K, ·).

    Parameters
    ----------
    tm : TrackManager
        Треки: наблюдения в tm.obs, начальные 3D в tm.X.
    cam_poses : list of (R_ito0, t_ito0)
        Позы камер в системе кадра 0.
    K : (3,3)
    loss : 'linear' | 'huber' | 'cauchy' | 'soft_l1'
    f_scale : float
        Масштаб ошибки для робастной функции (в пикселях).
    fix_camera0 : bool
        Фиксировать позу камеры 0 (gauge freedom монокуляра).
    verbose : int
    max_nfev : int

    Returns
    -------
    cam_poses_new : list of (R_i, t_i)
    tm : TrackManager (обновлён in-place — tm.X)
    result : OptimizeResult
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

    # ---- 4. Распаковка x -> (R_w2c, t_w2c) для каждой камеры и точек ----
    def unpack(x):
        cams = {}
        for c in cam_opt:
            i = cam_idx[c]
            rv = x[6*i : 6*i+3]
            tv = x[6*i+3 : 6*i+6]
            R = Rotation.from_rotvec(rv).as_matrix()
            cams[c] = (R, tv.reshape(3, 1))
        # камера 0 — из начальных данных, если фиксирована
        if 0 not in cams:
            cams[0] = (np.asarray(cam_poses[0][0]),
                       np.asarray(cam_poses[0][1]).reshape(3, 1))
        pts = x[n_cam_params:].reshape(n_pts, 3)
        return cams, pts

    # ---- 5. Residuals (векторизованные) ----
    K = np.asarray(K, dtype=float)

    def residuals(x):
        cams, pts = unpack(x)

        # world->cam для каждой камеры: R_wc = R_w2c.T, t_wc = -R_wc @ t_w2c
        R_wc = np.empty((n_cams, 3, 3))
        t_wc = np.empty((n_cams, 3))
        for c in range(n_cams):
            R_w2c, t_w2c = cams[c]
            R_wc[c] = R_w2c.T
            t_wc[c] = (-R_w2c.T @ np.asarray(t_w2c).reshape(3, 1)).ravel()

        # Проекция всех наблюдений
        R_o = R_wc[obs_cam]          # (M, 3, 3)
        t_o = t_wc[obs_cam]          # (M, 3)
        X_o = pts[obs_pt]            # (M, 3)

        Xc = np.einsum('nij,nj->ni', R_o, X_o) + t_o   # (M, 3)
        u_h = Xc @ K.T                                  # (M, 3)
        u_hat = u_h[:, :2] / u_h[:, 2:3]

        return (u_hat - obs_uv).ravel()

    # ---- 6. Разреженность якобиана ----
    n_res = 2 * n_obs
    n_vars = n_cam_params + 3 * n_pts
    sp = lil_matrix((n_res, n_vars), dtype=int)

    for k in range(n_obs):
        cam = obs_cam[k]
        pt  = obs_pt[k]
        if cam in cam_idx:
            col = 6 * cam_idx[cam]
            sp[2*k    , col:col+6] = 1
            sp[2*k + 1, col:col+6] = 1
        col_p = n_cam_params + 3 * pt
        sp[2*k    , col_p:col_p+3] = 1
        sp[2*k + 1, col_p:col_p+3] = 1

    # ---- 7. Запуск ----
    r0 = residuals(x0)
    rms0 = float(np.sqrt(np.mean(r0**2)))
    if verbose:
        print(f"[BA] initial RMS = {rms0:.3f} px")

    result = least_squares(
        residuals, x0,
        jac_sparsity=sp.tocsr(),
        loss=loss, f_scale=f_scale,
        method='trf',
        verbose=verbose,
        x_scale='jac',
        max_nfev=max_nfev,
    )

    r1 = residuals(result.x)
    rms1 = float(np.sqrt(np.mean(r1**2)))
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