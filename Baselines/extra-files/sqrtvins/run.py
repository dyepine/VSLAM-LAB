"""Calibrated stereo/IMU file adapter for the pinned author sqrtVINS implementation."""
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
SOURCE = ADAPTER.parent.parent / 'SqrtVINS'
sys.path.insert(0, str(ADAPTER.parent))
from stereo_vi_inputs import read_yaml, transform, write_manifest


def native_settings(calibration, settings):
    settings = copy.deepcopy(settings)
    imu = next(i for i in calibration['imus'] if i['imu_name'] == 'imu_0')
    T_BI = transform(imu['T_BS'])
    cameras = {}
    for i in (0, 1):
        camera = next(c for c in calibration['cameras'] if c['cam_name'] == f'rgb_{i}')
        distortion = camera['distortion_type']
        if camera['cam_model'] != 'pinhole' or distortion not in ('radtan4', 'equid4', 'pinhole'):
            raise ValueError(f'sqrtVINS requires pinhole+radtan4/equid4, got {distortion}')
        D = [0.] * 4 if distortion == 'pinhole' else camera['distortion_coefficients']
        if len(D) != 4 or not np.isfinite(D).all():
            raise ValueError('sqrtVINS requires four finite distortion coefficients')
        cameras[f'cam{i}'] = dict(T_imu_cam=(np.linalg.inv(T_BI) @ transform(camera['T_BS'])).tolist(),
            cam_overlaps=[1-i], camera_model='pinhole', distortion_coeffs=D,
            distortion_model='equidistant' if distortion == 'equid4' else 'radtan',
            intrinsics=camera['focal_length'] + camera['principal_point'],
            resolution=camera['image_dimension'], timeshift_cam_imu=0., rostopic=f'/cam{i}/image_raw')
    noise = {}
    for source, target in [('sigma_g_c', 'gyroscope_noise_density'), ('sigma_gw_c', 'gyroscope_random_walk'),
                           ('sigma_a_c', 'accelerometer_noise_density'), ('sigma_aw_c', 'accelerometer_random_walk')]:
        value = float(imu[source])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid IMU noise: {source}')
        noise[target] = value
    if any(imu.get('a0', [0., 0., 0.])) or any(imu.get('g0', [0., 0., 0.])):
        raise ValueError('Nonzero initial sensor biases require explicit adaptation')
    for key in ('calib_cam_extrinsics', 'calib_cam_intrinsics', 'calib_cam_timeoffset',
                'calib_imu_intrinsics', 'calib_imu_g_sensitivity'):
        if settings.get(key):
            raise ValueError(f'Fixed-calibration comparison requires {key}: false')
    if settings.get('use_mask') or settings.get('max_cameras') != 2 or not settings.get('use_stereo'):
        raise ValueError('sqrtVINS file adapter requires stereo without external masks')
    settings['gravity_mag'] = float(imu['g'])
    noise.update(update_rate=float(imu['fps']), rostopic='/imu/data')
    return settings, cameras, dict(imu0=noise), T_BI


def write_opencv_yaml(path, value):
    class Dumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)
    path.write_text('%YAML:1.0\n---\n' + yaml.dump(value, Dumper=Dumper, sort_keys=False))


def imu_to_body(values, T_BI):
    values = values.copy()
    T_IB = np.linalg.inv(T_BI)
    rotations = Rotation.from_quat(values[:, 3:]).as_matrix()
    values[:, :3] += np.einsum('nij,j->ni', rotations, T_IB[:3, 3])
    values[:, 3:] = Rotation.from_matrix(rotations @ T_IB[:3, :3]).as_quat()
    return values


def main():
    parser = argparse.ArgumentParser()
    for name in ('sequence_path', 'calibration_yaml', 'rgb_csv', 'exp_folder', 'settings_yaml'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--exp_it', type=int, required=True)
    parser.add_argument('--mode', choices=['stereo-vi'], default='stereo-vi')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    folder = args.exp_folder
    settings, cameras, imu, T_BI = native_settings(read_yaml(args.calibration_yaml), read_yaml(args.settings_yaml))
    camera_path = folder / 'sqrtvins_cameras.yaml'; imu_path = folder / 'sqrtvins_imu.yaml'
    # Upstream YamlParser always prefixes these paths with the main config directory.
    settings.update(relative_config_imucam=camera_path.name, relative_config_imu=imu_path.name)
    config = folder / 'sqrtvins_native.yaml'
    for path, data in ((camera_path, cameras), (imu_path, imu), (config, settings)):
        write_opencv_yaml(path, data)
    manifest = folder / 'sqrtvins_images.tsv'
    pairs = write_manifest(args.sequence_path, args.rgb_csv, manifest, ['rgb_0', 'rgb_1'], .005)
    imu_csv = args.sequence_path / 'imu_0.csv'
    with imu_csv.open() as stream:
        reader = csv.reader(stream); next(reader)
        origin_ns = int(next(reader)[0]) - 1_000_000_000
    raw = folder / f'{args.exp_it:05d}_sqrtvins_imu.csv.tmp'
    env = dict(os.environ)
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    binary = SOURCE / 'build/vslamlab_sqrtvins_stereo_vi'
    provenance = dict(repository='https://github.com/rpng/sqrtVINS',
        revision=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        precision='float32 (upstream USE_FLOAT=1)', timestamp_origin_ns=origin_ns, stereo_pairs=pairs,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (args.calibration_yaml, args.settings_yaml, config, camera_path, imu_path, imu_csv, manifest)},
        adapter_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), ADAPTER/'runner.cpp', ADAPTER/'CMakeLists.txt')},
        native_config=str(config), estimate_frame='body',
        pose_conversion='Hamilton R_WI from transpose of native JPL R_GtoI; T_WB=T_WI*inverse(T_BI)', status='running')
    provenance_path = folder / f'sqrtvins_provenance_{args.exp_it:05d}.json'
    provenance_path.write_text(json.dumps(provenance, indent=2))
    subprocess.run([str(binary), str(config), str(manifest), str(imu_csv), str(raw), str(origin_ns)],
                   check=True, env=env)
    with raw.open() as stream:
        rows = list(csv.reader(stream))[1:]
    stamps = np.array([int(row[0]) for row in rows], dtype=np.int64)
    values = np.array([[float(v) for v in row[1:]] for row in rows])
    if (len(stamps) < 3 or np.any(np.diff(stamps) <= 0) or not np.isfinite(values).all()
            or np.any(np.abs(np.linalg.norm(values[:, 3:], axis=1)-1) > .005)):
        raise RuntimeError('Invalid/insufficient sqrtVINS trajectory')
    values = imu_to_body(values, T_BI)
    trajectory = folder / f'{args.exp_it:05d}_KeyFrameTrajectory.csv'
    temporary = trajectory.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows([int(stamp), *row] for stamp, row in zip(stamps, values))
    temporary.replace(trajectory)
    provenance.update(status='completed', initialized_poses=len(stamps))
    provenance_path.write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
