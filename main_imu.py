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

    for i in range(len(frames) - 1):
        # IMU data part for initial poses for process_stereo_pair_i
        R_delta_imu, t_delta_imu, v_delta_imu = imu.preintegrate(imu.t[i], imu.t[i+1], R_0_imu, v_0_imu)
        R_old = R_0_imu.copy()  # до обновления!
        R_0_imu = R_0_imu @ R_delta_imu
        v_0_imu = v_0_imu + R_old @ v_delta_imu  # body(t0) -> world

        # предвариельные замеры для process_stereo_pair_i
        R_ij_imu = R_delta_imu.T  # body(t0) -> body(t1)
        t_ij_imu = R_delta_imu.T @ t_delta_imu  # сдвиг в body(t1)... проверить на тесте

        H_iip1 = imu.build_H(R_delta, t_delta, K, n, d_bar)

        R_i, t_i = process_stereo_pair_i(
            frames[i], frames[i+1], tm,
            world_pose=(R_0toi.T, -R_0toi.T @ t_0toi), # R_ito0 t_it0
            cam_i=i, cam_ip1=i + 1, H_iip1=H_iip1, verbose=0)

        # instantly obtaining camera pose in camera 0 system
        cam_poses.append([R_0toi.T, -R_0toi.T @ t_0toi])

        # Обновляем: 0→(i+1) = (i→i+1) ∘ (0→i)
        R_0toi = R_i @ R_0toi
        t_0toi = R_i @ t_0toi + t_i

        print(f"[iter {i}] {tm.stats(i)}")

    # last camera
    cam_poses.append([R_0toi.T, -R_0toi.T @ t_0toi])
