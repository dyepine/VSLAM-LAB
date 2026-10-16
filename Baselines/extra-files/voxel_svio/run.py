"""File inputs/pose representation for the pinned author implementation of Voxel-SVIO."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ADAPTER = Path(__file__).resolve().parent
SOURCE = ADAPTER.parent.parent / 'Voxel-SVIO'
sys.path.insert(0, str(ADAPTER.parent))
from stereo_vi_inputs import read_yaml, transform, write_manifest


def native_settings(calibration, settings, output):
    result = copy.deepcopy(settings)
    imu = next(i for i in calibration['imus'] if i['imu_name'] == 'imu_0')
    T_BI = transform(imu['T_BS'])
    cameras = [next(c for c in calibration['cameras'] if c['cam_name'] == n)
               for n in ['rgb_0', 'rgb_1']]
    for camera, side in zip(cameras, ['left', 'right']):
        distortion = camera['distortion_type']
        if camera['cam_model'] != 'pinhole' or distortion not in ('radtan4', 'equid4', 'pinhole'):
            raise ValueError(f'Voxel-SVIO requires radtan4/equid4/pinhole, got {distortion}')
        coefficients = [0.] * 4 if distortion == 'pinhole' else camera['distortion_coefficients']
        if len(coefficients) != 4:
            raise ValueError('Voxel-SVIO requires four distortion coefficients')
        result['camera_parameter'].update({
            f'distortion_model_{side}': 'equidistant' if distortion == 'equid4' else 'radtan',
            f'intrinsics_{side}': camera['focal_length'] + camera['principal_point'],
            f'distortion_coeffs_{side}': coefficients,
            f'resolution_{side}': camera['image_dimension'],
            f'T_imu_cam_{side}': (np.linalg.inv(T_BI) @ transform(camera['T_BS'])).ravel().tolist(),
            f'timeshift_cam_imu_{side}': 0.0})
    for source, destination in [('sigma_g_c', 'gyroscope_noise_density'),
                                ('sigma_gw_c', 'gyroscope_random_walk'),
                                ('sigma_a_c', 'accelerometer_noise_density'),
                                ('sigma_aw_c', 'accelerometer_random_walk')]:
        value = float(imu[source])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid IMU noise: {source}')
        result['imu_parameter'][destination] = value
    result['initializer_parameter']['gravity_mag'] = float(imu['g'])
    if any(imu.get('a0', [0., 0., 0.])) or any(imu.get('g0', [0., 0., 0.])):
        raise ValueError('Nonzero initial sensor biases must be explicitly adapted')
    # CSV data is already calibrated. Keep the same fixed intrinsics/extrinsics for all methods.
    for key in ['calib_cam_extrinsics', 'calib_cam_intrinsics', 'calib_cam_timeoffset',
                'calib_imu_intrinsics', 'calib_imu_g_sensitivity']:
        if result['state_parameter'][key]:
            raise ValueError(f'Fixed-calibration comparison requires {key}: false')
    for key in ['Tw', 'Ta', 'R_acc_imu', 'R_gyr_imu']:
        result['imu_parameter'][key] = np.eye(3).ravel().tolist()
    result['imu_parameter']['Tg'] = [0.] * 9
    if result['odometry_parameter']['use_mask']:
        raise ValueError('Masks are not supported by this file adapter')
    result['output_path'] = str(output)
    return result, T_BI


def imu_to_body(values, T_BI):
    result = values.copy()
    T_IB = np.linalg.inv(T_BI)
    rotations = Rotation.from_quat(values[:, 3:]).as_matrix()
    result[:, :3] += np.einsum('nij,j->ni', rotations, T_IB[:3, 3])
    result[:, 3:] = Rotation.from_matrix(rotations @ T_IB[:3, :3]).as_quat()
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ['sequence_path', 'calibration_yaml', 'rgb_csv', 'exp_folder', 'settings_yaml']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--exp_it', type=int, required=True)
    parser.add_argument('--mode', choices=['stereo-vi'], default='stereo-vi')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    native, T_BI = native_settings(read_yaml(args.calibration_yaml), read_yaml(args.settings_yaml), args.exp_folder)
    native_path = args.exp_folder / 'voxel_svio_native.yaml'
    native_path.write_text(yaml.safe_dump(native, sort_keys=False))
    manifest = args.exp_folder / 'voxel_svio_images.tsv'
    pairs = write_manifest(args.sequence_path, args.rgb_csv, manifest, ['rgb_0', 'rgb_1'], .005)
    imu_csv = args.sequence_path / 'imu_0.csv'
    with imu_csv.open() as stream:
        reader = csv.reader(stream)
        next(reader)
        origin_ns = int(next(reader)[0]) - 1000000000
    raw = args.exp_folder / f'{args.exp_it:05d}_voxel_svio_imu.csv.tmp'
    env = dict(os.environ)
    for key in ['LD_LIBRARY_PATH', 'LD_PRELOAD']:
        env.pop(key, None)
    subprocess.run([str(SOURCE / 'build/vslamlab_voxel_svio_stereo_vi'), str(native_path),
                    str(manifest), str(imu_csv), str(raw), str(origin_ns)], env=env, check=True)
    with raw.open() as stream:
        rows = list(csv.reader(stream))[1:]
    stamps = np.array([int(row[0]) for row in rows], dtype=np.int64)
    values = np.array([[float(v) for v in row[1:]] for row in rows])
    if (len(stamps) < 3 or np.any(np.diff(stamps) <= 0) or not np.isfinite(values).all() or
        np.any(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1) > .005)):
        raise RuntimeError('Voxel-SVIO exported invalid/insufficient initialized poses')
    values = imu_to_body(values, T_BI)
    trajectory = args.exp_folder / f'{args.exp_it:05d}_KeyFrameTrajectory.csv'
    temporary = trajectory.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows([str(stamp), *row] for stamp, row in zip(stamps, values))
    temporary.replace(trajectory)
    (args.exp_folder / 'voxel_svio_provenance.json').write_text(json.dumps({
        'repository': 'https://github.com/ZikangYuan/voxel_svio',
        'revision': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'patch_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ADAPTER.glob('*.patch')},
        'timestamp_origin_ns': origin_ns, 'stereo_pairs': pairs, 'initialized_poses': len(stamps),
        'estimate_frame': 'body', 'native_settings': str(native_path),
        'orientation': 'Hamilton R_WI from transpose of native JPL R_GtoI; T_WB=T_WI*inverse(T_BI)'}, indent=2))


if __name__ == '__main__':
    main()
