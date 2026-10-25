from utils import *
from debug import *
from imu import *
import matplotlib.pyplot as plt
from pair import process_stereo_pair_i
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
    frames = load_frames(main_config['video'], scale_factor=main_config['scale_factor'], quiet=True)
    frames, selected_idx = select_key_frames(frames, drop=main_config['drop'])
    imu.mask_frames(selected_idx)
    frames = undistort_frames(frames, dist, K)

    if 0:
        for i, frame in enumerate(frames):
            plt.figure()
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f'Кадр {i + 1} из {len(frames)}')
            plt.show()

    # Begin pairwise loop
    R_0_imu = imu.init_orientation_from_gravity()
    v_0_imu = np.zeros(3) # fair assumption

    tm = TrackManager()
    cam_poses = []    # [R_i, t_i] — позы cam_i → cam_{i+1}

    # Накопитель: cam_left → world (frame 0)
    R_0toi = np.eye(3)
    t_0toi = np.zeros((3, 1))

    n = np.array([0.0, 0.0, 1.0])

    for i in range(len(frames) - 1):
        R_delta_imu, t_delta_imu, v_delta_imu = imu.preintegrate(
            imu.t[i], imu.t[i + 1], R_0_imu, v_0_imu)

        R_old = R_0_imu.copy()
        R_0_imu = R_0_imu @ R_delta_imu
        v_0_imu = v_0_imu + R_old @ v_delta_imu

        # pose cam_i -> cam_ip1 для H: X_ip1 = R_ij @ X_i + t_ij
        R_ij_imu = R_delta_imu.T
        t_ij_imu = -R_delta_imu.T @ t_delta_imu

        # средняя глубина в cam_i
        if len(tm.X) >= 20:
            X_w = np.array(list(tm.X.values()))
            R_ito0 = R_0toi.T
            t_ito0 = -R_0toi.T @ t_0toi
            X_cam_i = (R_ito0.T @ (X_w - t_ito0.ravel()).T).T
            d_bar = float(np.median(X_cam_i[:, 2]))
        else:
            d_bar = None

        H_iip1 = imu.build_H(R_ij_imu, t_ij_imu, K, n, d_bar)

        R_i, t_i = process_stereo_pair_i(
            frames[i], frames[i + 1], tm,
            world_pose=(R_0toi.T, -R_0toi.T @ t_0toi),
            cam_i=i, cam_ip1=i + 1, H_iip1=H_iip1, verbose=0)

        cam_poses.append([R_0toi.T, -R_0toi.T @ t_0toi])
        R_0toi = R_i @ R_0toi
        t_0toi = R_i @ t_0toi + t_i

        print(f"[iter {i}] {tm.stats(i)}  d_bar={d_bar}")