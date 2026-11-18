from utils import *
from debug import *
from imu import *
import matplotlib.pyplot as plt
from pair import process_stereo_pair
from track import TrackManager
from ba import bundle_adjust
from pair import process_dense_pair
import json
import csv

if __name__ == '__main__':
    # load configs
    main_config = json.load(open('main.json'))
    camera_config = json.load(open(main_config['camera_json']))

    # load IMU data
    imu = IMUData()
    imu.load(main_config['imu']['t'],
             main_config['imu']['acc'],
             main_config['imu']['gyr'],
             main_config['imu']['mag'])

    # load and trim camera parameters
    dist = np.array(camera_config['D'])
    K = main_config['scale_factor'] * np.array(camera_config['K'])
    K[2, 2] = 1

    # load frames
    frames = load_frames(main_config['video'], scale_factor=main_config['scale_factor'], quiet=True)[:150]
    if abs(dist[0]) > 0.01:
        frames = undistort_frames(frames, dist, K)
    frames, selected_idx = select_key_frames(frames, min_survival=0.65, drop=main_config['drop'])
    imu.mask_frames(selected_idx)

    if 0:
        for i, frame in enumerate(frames):
            plt.figure()
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f'Кадр {i} из {len(frames)}')
            plt.show()

    # Begin pairwise loop
    R_ito0_imu = imu.init_orientation_from_gravity()
    v_0_imu = np.zeros(3) # fair assumption

    tm = TrackManager()
    cam_poses = []    # [R_ito0, t_ito0] — позы cam_i → cam_0

    # Накопитель: world (frame 0) → cam_i
    R_0toi = np.eye(3)
    t_0toi = np.zeros((3, 1))

    n = np.array([0.0, 0.0, 1.0])

    for i in range(len(frames) - 1):
        # consider next frame
        j = i + 1

        R_ji_imu, t_ji_imu, v_ji_imu = imu.preintegrate(imu.t[i], imu.t[j], R_ito0_imu, v_0_imu)
        v_0_imu    = v_0_imu + R_ito0_imu @ v_ji_imu   # R ещё старая — dv в осях body(i)
        R_ito0_imu = R_ito0_imu @ R_ji_imu

        # pose cam_i -> cam_j для H: X_j = R_ij @ X_i + t_ij
        R_ij_imu = R_ji_imu.T
        t_ij_imu = -R_ji_imu.T @ t_ji_imu

        # средняя глубина в cam_i
        if len(tm.X) >= 20:
            X_w = np.array(list(tm.X.values()))
            X_cam_i = X_w @ R_0toi.T + t_0toi.ravel()
            d_bar = float(np.median(X_cam_i[:, 2]))
        else:
            d_bar = None

        H_ij = imu.build_H(R_ij_imu, t_ij_imu, K, n, d_bar)

        R_ij, t_ij = process_stereo_pair(
            frames[i], frames[j], tm, world_pose=(R_0toi.T, -R_0toi.T @ t_0toi),
            K=K, cam_i=i, cam_j=j, H_ij=H_ij, t_ij_imu=t_ij_imu, verbose=0)

        cam_poses.append([R_0toi.T, -R_0toi.T @ t_0toi])
        R_0toi = R_ij @ R_0toi
        t_0toi = R_ij @ t_0toi + t_ij

        print(f"[iter {i}] {tm.stats(i)}  d_bar={d_bar}")

    # last camera
    cam_poses.append([R_0toi.T, -R_0toi.T @ t_0toi])

    cam_poses, tm, result = bundle_adjust(tm, cam_poses, K)

    visualize_slam_scene(tm, cam_poses, K=K, color_mode='pixel')
    visualize_slam_scene(tm, cam_poses, K=K, color_mode='camera')