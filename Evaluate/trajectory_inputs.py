"""Validate VSLAM-LAB CSVs and adapt their representation to evo."""
import csv
from pathlib import Path

import numpy as np
import yaml
from evo.core.trajectory import PoseTrajectory3D


def read_pose_csv(path: Path):
    with path.open() as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if [s.strip() for s in header] != ['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw']:
            raise ValueError(f'Expected standard VSLAM-LAB pose CSV: {path}')
        rows = list(reader)
    if len(rows) < 3 or any(len(row) != 8 for row in rows):
        raise ValueError(f'Need at least three complete poses: {path}')
    stamps = np.array([int(row[0]) for row in rows], dtype=np.int64)
    values = np.array([[float(x) for x in row[1:]] for row in rows])
    if np.any(np.diff(stamps) <= 0) or not np.isfinite(values).all():
        raise ValueError(f'Non-monotonic timestamps or non-finite poses: {path}')
    norms = np.linalg.norm(values[:, 3:], axis=1)
    if np.any(np.abs(norms - 1) > 0.005):
        raise ValueError(f'Invalid unit quaternions: {path}')
    values[:, 3:] /= norms[:, None]
    return stamps, values


def to_evo(stamps, values, origin_ns):
    # Subtract a shared integer origin BEFORE converting ns to float seconds.
    # VSLAM-LAB uses xyzw, evo uses wxyz (not EuRoC CSV's ordering).
    return PoseTrajectory3D(values[:, :3], np.roll(values[:, 3:], 1, axis=1),
                            (stamps - origin_ns).astype(float) / 1e9)


def camera_to_body(trajectory, calibration: Path, camera_name='rgb_0'):
    cameras = yaml.safe_load(calibration.read_text())['cameras']
    camera = next(cam for cam in cameras if cam['cam_name'] == camera_name)
    transform = np.array(camera['T_BS'], dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    if (not np.isfinite(transform).all() or
        not np.allclose(transform[3], [0, 0, 0, 1]) or
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
        not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError(f'Invalid {camera_name} T_BS calibration')
    trajectory.transform(np.linalg.inv(transform), right_mul=True)
    return trajectory
