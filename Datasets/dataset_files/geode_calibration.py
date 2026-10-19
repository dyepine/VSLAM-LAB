"""Pure NumPy/OpenCV GEODE calibration, shared with the isolated ROS1 extractor."""
import cv2
import numpy as np
import yaml


def matrix(record):
    T = np.asarray(record['data'], dtype=float).reshape(record['rows'], record['cols'])
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Invalid rigid calibration transform')
    return T


def quaternion_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion) / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_calibration(path):
    # The source is three concatenated YAML sections with duplicate camera keys
    # and a duplicated gyro_std. Read sections separately; never last-key-wins.
    text = path.read_text()
    left, rest = text.split('# Right Camera Parameters.', 1)
    right, rest = rest.split('# IMU noise', 1)
    noise, transforms = rest.split('T_IMU_CamL:', 1)
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:opencv-matrix',
                           lambda loader, node: loader.construct_mapping(node, deep=True))
    gyro = [float(line.split(':', 1)[1]) for line in noise.splitlines()
            if line.startswith('gyro_std:')]
    if len(gyro) != 2:
        raise ValueError('Expected the two documented gyro_std entries')
    scalar = yaml.safe_load(noise)
    return {
        'cameras': [yaml.safe_load(left), yaml.safe_load(right)],
        'noise': dict(sigma_g_c=gyro[0], sigma_gw_c=gyro[1],
                      sigma_a_c=scalar['accel_std'], sigma_aw_c=scalar['accel_rw']),
        'transforms': yaml.load('T_IMU_CamL:' + transforms, Loader=Loader),
    }


def body_transforms(calibration):
    """Publisher beta<-gamma LiDAR and beta<-Xsens transforms, independent of image processing."""
    source = calibration['transforms']
    # Publisher gamma2GT_gnss.py: T_world_beta = T_world_gamma @ inv(T).
    # The gamma device frame is the LiDAR frame used by the publisher's LIOs.
    # Pin the numerical transform from that script rather than fitting to GT.
    T_BG = np.eye(4)
    T_BG[:3, :3] = quaternion_matrix([-0.0057758, 0.0022253, 0.0140019, 0.9998828])
    T_BG[:3, 3] = [0.0305, -0.5959, 0.0902]
    T_BI = T_BG @ np.linalg.inv(matrix(source['T_IMU_LiDAR']))
    return T_BG, T_BI


def geometry(calibration):
    """T_A_B maps B into A. Use the consistent two camera-IMU calibrations."""
    source = calibration['transforms']
    T_IC0 = matrix(source['T_IMU_CamL'])
    T_IC1 = matrix(source['T_IMU_CamR'])
    T_C1C0 = np.linalg.inv(T_IC1) @ T_IC0
    T_BG, T_BI = body_transforms(calibration)
    cameras = calibration['cameras']
    size = (int(cameras[0]['image_width']), int(cameras[0]['image_height']))
    Ks, Ds = [], []
    for camera in cameras:
        if (camera['image_width'], camera['image_height']) != size:
            raise ValueError('Stereo camera resolutions differ')
        p, d = camera['projection_parameters'], camera['distortion_parameters']
        Ks.append(np.array([[p['fx'], 0, p['cx']], [0, p['fy'], p['cy']], [0, 0, 1.]], float))
        Ds.append(np.array([d[k] for k in ('k1', 'k2', 'p1', 'p2', 'k3')]))
    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
        Ks[0], Ds[0], Ks[1], Ds[1], size, T_C1C0[:3, :3].copy(), T_C1C0[:3, 3].reshape(3, 1).copy(),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    extrinsics, maps = [], []
    for K, D, R, P, T_IC in zip(Ks, Ds, (R0, R1), (P0, P1), (T_IC0, T_IC1)):
        T_Crect = np.eye(4)
        T_Crect[:3, :3] = R.T
        extrinsics.append(T_BI @ T_IC @ T_Crect)
        maps.append(cv2.initUndistortRectifyMap(K, D, R, P[:3, :3], size, cv2.CV_32FC1))
    discrepancy = matrix(source['T_CamR_CamL'])
    metadata = {
        'inter_rig_transform_source': 'https://github.com/PengYu-Team/GEODE_dataset/blob/c6e930623d4fed450d7fc50e16e3ffe0288b692b/script/gamma2GT_gnss.py',
        'body_frame': 'publisher beta reference frame',
        'T_beta_gamma_lidar': T_BG.tolist(),
        'T_beta_xsens': T_BI.tolist(),
        'camera_chain': 'T_IMU_CamL and T_IMU_CamR; stereo = inverse(T_IMU_CamR) @ T_IMU_CamL',
        'unused_stereo_translation_difference_m': float(np.linalg.norm(discrepancy[:3, 3] - T_C1C0[:3, 3])),
        'unused_stereo_rotation_difference_deg': float(np.linalg.norm(cv2.Rodrigues(discrepancy[:3, :3] @ T_C1C0[:3, :3].T)[0]) * 180 / np.pi),
        'rectification_version': 2,
        'rectification': 'OpenCV stereoRectify, alpha=0, CALIB_ZERO_DISPARITY; full five-coefficient source model',
        'gyro_source_typo': 'first gyro_std is white noise; second is random walk, according to adjacent comments',
        'noise_units': 'imu_utils continuous noise convention; no per-sample rate scaling',
        'calibration_status': 'published starting calibration; independent chains disagree; not a validated precision benchmark',
        'publisher_calibration_issue': 'https://github.com/PengYu-Team/GEODE_dataset/issues/19',
    }
    return size, maps, (P0, P1), extrinsics, T_BI, metadata
