"""Sensor representation checks shared by offline stereo-inertial adapters."""
import csv
from pathlib import Path

import numpy as np
import yaml


def read_yaml(path):
    text = Path(path).read_text()
    if text.startswith('%YAML:'):
        text = text.split('\n', 1)[1]
    return yaml.safe_load(text)

def transform(values):
    matrix = np.asarray(values, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]) or
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
        not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError('Invalid rigid sensor T_BS')
    return matrix

def write_manifest(sequence, rgb_csv, destination, names, tolerance):
    previous = [-1, -1]
    count = 0
    with Path(rgb_csv).open(newline='') as stream, destination.open('w') as out:
        for row in csv.DictReader(stream):
            timestamps = [int(row[f'ts_{name} (ns)']) for name in names]
            if any(t <= p for t, p in zip(timestamps, previous)):
                raise ValueError('Non-monotonic stereo timestamps')
            if abs(timestamps[0] - timestamps[1]) >= tolerance * 1e9 and timestamps[0] != timestamps[1]:
                raise ValueError('Stereo timestamps exceed OKVIS synchronization tolerance')
            paths = [(Path(sequence) / row[f'path_{name}']).resolve() for name in names]
            if any(not path.is_file() or '\t' in str(path) or '\n' in str(path) for path in paths):
                raise ValueError('Missing image or invalid path in stereo input')
            out.write(f'{timestamps[0]}\t{paths[0]}\t{timestamps[1]}\t{paths[1]}\n')
            previous = timestamps
            count += 1
    if count < 3:
        raise ValueError('At least three stereo pairs required')
    return count
