"""Translate VSLAM-LAB inputs; run the pinned, unmodified OKVIS 1 estimator."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ADAPTER = Path(__file__).resolve().parent
SOURCE = ADAPTER.parent.parent / 'OKVIS1'
sys.path.insert(0, str(ADAPTER.parent))
from stereo_vi_inputs import read_yaml, transform, write_manifest


class OpenCVDumper(yaml.SafeDumper):
    pass


OpenCVDumper.add_representer(list, lambda dumper, values: dumper.represent_sequence(
    'tag:yaml.org,2002:seq', values, flow_style=True))


def native_settings(calibration, settings, imu_overrides):
    """Native camera T_SC = inverse(T_BI) @ T_BC; IMU T_BS remains T_BI."""
    result = copy.deepcopy(settings)
    names = result.pop('cam_stereo')
    if not isinstance(names, list) or len(names) != 2 or names[0] == names[1]:
        raise ValueError('cam_stereo must select two distinct cameras')
    imu_name = result.pop('imu')
    imu = copy.deepcopy(next(item for item in calibration['imus'] if item['imu_name'] == imu_name))
    allowed = {'sigma_bg', 'sigma_ba', 'a_max', 'g_max', 'a0', 'tau'}
    if not isinstance(imu_overrides, dict) or set(imu_overrides) - allowed:
        raise ValueError('Unsupported OKVIS 1 imu_overrides')
    imu.update(imu_overrides)
    T_BI = transform(imu['T_BS'])
    # Nonzero g0 is not an input supported by OKVIS 1's parameter reader.
    if any(imu.get('g0', [0., 0., 0.])):
        raise ValueError('OKVIS 1 does not support nonzero initial gyro bias g0')
    result['cameras'] = []
    models = {'equid4': ('equidistant', 4), 'radtan4': ('radialtangential', 4),
              'radtan5': ('radialtangential8', 5),
              'radtan8': ('radialtangential8', 8), 'pinhole': ('radialtangential', 4)}
    for name in names:
        camera = next(item for item in calibration['cameras'] if item['cam_name'] == name)
        if camera['cam_model'] != 'pinhole' or camera['distortion_type'] not in models:
            raise ValueError(f'Unsupported OKVIS 1 camera model: {name}')
        model, size = models[camera['distortion_type']]
        coefficients = ([0.] * 4 if camera['distortion_type'] == 'pinhole'
                        else camera['distortion_coefficients'])
        if len(coefficients) != size:
            raise ValueError(f'{name}: distortion coefficient count mismatch')
        if camera['distortion_type'] == 'radtan5':
            # Exact Brown polynomial as a rational model with unit denominator.
            # Native order is [k1,k2,p1,p2,k3,k4,k5,k6].
            coefficients = list(coefficients) + [0., 0., 0.]
        result['cameras'].append({
            'T_SC': (np.linalg.inv(T_BI) @ transform(camera['T_BS'])).ravel().tolist(),
            'image_dimension': camera['image_dimension'],
            'distortion_coefficients': coefficients, 'distortion_type': model,
            'focal_length': camera['focal_length'], 'principal_point': camera['principal_point']})
    cameras = [next(c for c in calibration['cameras'] if c['cam_name'] == n) for n in names]
    if cameras[0]['fps'] != cameras[1]['fps']:
        raise ValueError('Stereo camera rates differ')
    result['camera_params']['camera_rate'] = int(cameras[0]['fps'])
    fields = ['a_max', 'g_max', 'sigma_g_c', 'sigma_a_c', 'sigma_bg', 'sigma_ba',
              'sigma_gw_c', 'sigma_aw_c', 'g', 'tau']
    native_imu = {}
    for key in fields:
        value = float(imu[key])  # Missing priors must be explicitly configured.
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'OKVIS 1 requires positive {key}; set imu_overrides for priors/limits')
        native_imu[key] = value
    a0 = np.asarray(imu['a0'], dtype=float)
    if a0.shape != (3,) or not np.isfinite(a0).all():
        raise ValueError('Expected finite accelerometer bias a0')
    native_imu.update(a0=a0.tolist(), imu_rate=int(imu['fps']), T_BS=T_BI.ravel().tolist())
    result['imu_params'] = native_imu
    if result['imageDelay'] != 0:
        raise ValueError('This adapter requires timestamp-corrected inputs and imageDelay: 0.0')
    result['displayImages'] = False
    result['useDriver'] = False
    result['publishing_options']['publishImuPropagatedState'] = False
    return result, names, imu_name


def main():
    parser = argparse.ArgumentParser()
    for key in ['sequence_path', 'calibration_yaml', 'rgb_csv', 'exp_folder', 'settings_yaml']:
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--exp_it', type=int, required=True)
    parser.add_argument('--mode', choices=['stereo-vi'], default='stereo-vi')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--imu_overrides', type=json.loads, default={})
    args = parser.parse_args()
    settings, names, imu = native_settings(read_yaml(args.calibration_yaml),
                                          read_yaml(args.settings_yaml), args.imu_overrides)
    native = args.exp_folder / 'okvis1_native.yaml'
    native.write_text('%YAML:1.0\n---\n' + yaml.dump(settings, Dumper=OpenCVDumper, sort_keys=False))
    manifest = args.exp_folder / 'okvis1_images.tsv'
    pairs = write_manifest(args.sequence_path, args.rgb_csv, manifest, names,
                           settings['camera_params']['timestamp_tolerance'])
    trajectory = args.exp_folder / f'{args.exp_it:05d}_KeyFrameTrajectory.csv'
    temporary = trajectory.with_suffix('.csv.tmp')
    env = dict(os.environ)
    for key in ['LD_LIBRARY_PATH', 'LD_PRELOAD']:
        env.pop(key, None)
    command = [str(SOURCE / 'build/vslamlab_okvis1_stereo_vi'), str(native),
               str(manifest), str(args.sequence_path / f'{imu}.csv'), str(temporary)]
    tracking = {'initialized': False, 'tracking_failure_messages': 0}
    # The pinned Frontend.cpp logs its initialization transition and tracking failures.
    # A trajectory file also exists before visual initialization; it cannot prove VIO works.
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            if ' Frontend.cpp:' in line:
                tracking['initialized'] |= '] Initialized!' in line
                tracking['tracking_failure_messages'] += 'Tracking failure.' in line
        returncode = process.wait()
    (args.exp_folder / 'okvis1_tracking.json').write_text(json.dumps(
        dict(tracking, returncode=returncode), indent=2))
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    if not tracking['initialized']:
        raise RuntimeError('OKVIS 1 never visually initialized; see okvis1_tracking.json and native log')
    with temporary.open() as stream:
        poses = list(csv.DictReader(stream))
    stamps = [int(row['ts (ns)']) for row in poses]
    values = np.array([[float(v) for k, v in row.items() if k != 'ts (ns)'] for row in poses])
    if (len(stamps) < 3 or any(b <= a for a, b in zip(stamps, stamps[1:])) or
        not np.isfinite(values).all() or
        np.any(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1) > 0.005)):
        raise RuntimeError('OKVIS 1 produced invalid or insufficient optimized poses')
    temporary.replace(trajectory)
    (args.exp_folder / 'okvis1_provenance.json').write_text(json.dumps({
        'repository': 'https://github.com/ethz-asl/okvis',
        'revision': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'estimate_frame': 'body', 'stereo_pairs': pairs, 'optimized_poses': len(poses),
        'tracking': tracking,
        'native_settings': str(native), 'imu_csv': str(args.sequence_path / f'{imu}.csv')}, indent=2))


if __name__ == '__main__':
    main()
