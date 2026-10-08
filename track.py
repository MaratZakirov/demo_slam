import numpy as np

class TrackManager:
    # Хранит треки: id -> [(cam, u, v), ...] и id -> X_world.
    # Живёт сквозь все кадры.
    def __init__(self):
        self.obs = {}            # tid -> list[(cam, u, v)]
        self.X   = {}            # tid -> np.ndarray(3,) in world (frame 0)
        self.next_id = 0

    # Создаёт новые треки с наблюдением на кадре `cam`
    def birth(self, cam, pts_2d):
        assert np.isfinite(pts_2d.sum()), 'Not finite!'
        ids = []
        for u, v in pts_2d:
            tid = self.next_id
            self.next_id += 1
            self.obs[tid] = [(int(cam), float(u), float(v))]
            ids.append(tid)
        return np.asarray(ids, dtype=int)

    # Добавляет наблюдение существующим трекам.
    def extend(self, tids, cam, pts_2d):
        assert np.isfinite(pts_2d.sum()), 'Not finite!'
        for tid, (u, v) in zip(tids, pts_2d):
            self.obs[int(tid)].append((int(cam), float(u), float(v)))

    def set_X(self, tids, X_arr):
        assert np.isfinite(X_arr.sum()), 'Not finite!'
        for tid, X in zip(tids, X_arr):
            self.X[int(tid)] = np.asarray(X, dtype=float).ravel()

    # X_world для каждого tid; NaN где ещё нет 3D.
    def get_X(self, tids):
        out = np.full((len(tids), 3), np.nan, dtype=float)
        for k, tid in enumerate(tids):
            X = self.X.get(int(tid))
            if X is not None:
                out[k] = X
        return out

    def active_on(self, cam):
        # Треки, чьё последнее наблюдение — на кадре `cam`.
        # Возвращает (tids, pts_2d, X_world).
        tids, pts2d, X = [], [], []
        for tid, observations in self.obs.items():
            if observations[-1][0] == cam:
                tids.append(tid)
                pts2d.append((observations[-1][1], observations[-1][2]))
                X.append(self.X.get(tid))

        tids = np.asarray(tids, dtype=int)
        pts2d = np.asarray(pts2d, dtype=float) if pts2d else np.zeros((0, 2))
        X = np.asarray(
            [x if x is not None else [np.nan, np.nan, np.nan] for x in X],
            dtype=float
        ) if X else np.zeros((0, 3))
        return tids, pts2d, X

    def stats(self, cam_i=None):
        n = len(self.obs)
        lengths = [len(v) for v in self.obs.values()]
        if not lengths:
            return "tm: empty"

        tids_on_cam_i = 0
        for tid in self.obs:
            for cam_j, u, v in self.obs[tid]:
                if cam_j == cam_i:
                    tids_on_cam_i += 1
                    break

        return (f"tm: tracks={n}  "
                f"obs/track min={min(lengths)} med={int(np.median(lengths))} max={max(lengths)}  "
                f"with3d={len(self.X)} tids_on_cam_i={tids_on_cam_i} ")