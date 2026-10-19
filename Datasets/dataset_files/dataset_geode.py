"""GEODE gamma stereo/Xsens adapter. License: GPLv3.

The body frame is the publisher's beta reference frame. Source bags and GT
are retained. See docs/GEODE.md in vio_research for calibration conventions.
"""
import csv
from decimal import Decimal
import errno
import json
import os
from pathlib import Path
import subprocess
import shutil

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from Datasets.dataset_files.dataset_botanicgarden import asset_checksum, verified, stereo_pairs
from Datasets.dataset_files.geode_calibration import read_calibration, body_transforms, matrix
from path_constants import VSLAM_LAB_DIR
from utilities import downloadFile, write_csv_rows


GT_ORIENTATION_VERSION = 1


def reference_orientations(quaternions):
    """Apply the explicit GNSS orientation convention, without Euler singularities.

    R' = Rz(90deg) Rx(180deg) R Rx(180deg), equivalent to extrinsic xyz
    roll'=roll, pitch'=-pitch, yaw'=90deg-yaw. No fitted bias is applied.
    """
    flip = Rotation.from_euler('x', 180, degrees=True)
    left = Rotation.from_euler('z', 90, degrees=True) * flip
    return (left * Rotation.from_quat(quaternions) * flip).as_quat()



def reference_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError('Expected TUM timestamp, xyz, xyzw')
        timestamp = Decimal(fields[0]) * 1_000_000_000
        if not timestamp.is_finite() or timestamp != timestamp.to_integral_value():
            raise ValueError('Reference timestamp is not an integer number of nanoseconds')
        rows.append([int(timestamp), *map(float, fields[1:])])
    stamps = np.array([r[0] for r in rows], dtype=np.int64)
    values = np.array([r[1:] for r in rows])
    if (len(rows) < 3 or np.any(stamps <= 0) or not np.isfinite(values).all()
            or np.any(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1) > .005)):
        raise ValueError('Invalid 6DoF reference')
    if len(np.unique(stamps)) != len(stamps):
        raise ValueError('Duplicate GT timestamps; refusing to choose a pose silently')
    inversions = int(np.count_nonzero(np.diff(stamps) < 0))
    # Explicit serialization policy, not a claim that source timing was repaired.
    rows.sort(key=lambda row: row[0])
    return rows, {'source_adjacent_time_inversions': inversions, 'pose_count': len(rows),
                  'timestamp_policy': 'stable ascending timestamp sort; no pose removal, interpolation or timestamp changes',
                  'timing_status': 'publisher timestamp anomalies remain a benchmark limitation'}


class GeodeGammaDataset(DatasetVSLAMLAB):
    def __init__(self):
        super().__init__('geode-gamma')

    def _raw(self):
        return Path(os.environ.get('GEODE_SOURCE_DIR', str(self.dataset_path / 'raw')))

    def assets_for(self, sequence_name):
        if not self.contains_sequence(sequence_name):
            raise ValueError(f'Unsupported GEODE sequence: {sequence_name}')
        return {**self.cfg['assets'],
                **self.cfg.get('assets_by_sequence', {}).get(sequence_name, {})}

    def _asset(self, kind, sequence_name='offroad1'):
        return self._raw() / self.assets_for(sequence_name)[kind]['name']

    def _old_rectification(self, sequence_name):
        folder = self.sequence_path(sequence_name)
        recorded = folder / 'rectification.json'
        return ((folder / 'import_complete.json').is_file() and recorded.is_file()
                and json.loads(recorded.read_text())['metadata'].get('rectification_version', 1) < 2)

    def download_sequence(self, sequence_name):
        if self._old_rectification(sequence_name):
            folder = self.sequence_path(sequence_name)
            backup = folder.with_name(folder.name + '.rectification-v1')
            if backup.exists():
                raise FileExistsError(f'Preserved GEODE v1 data already exists: {backup}; refusing to overwrite it')
            folder.rename(backup)
            print(f'GEODE: preserved old rectification in {backup}; regenerating from retained bag', flush=True)
        # Migrate datasets prepared by the first adapter version without reading
        # the bag or touching pixels. The recorded rectification stays canonical.
        if self.check_sequence_availability(sequence_name, verbose=False) == 'available':
            cameras = yaml.safe_load(self.calibration_yaml_path(sequence_name).read_text())['cameras']
            if any(c['distortion_type'] == 'pinhole' for c in cameras):
                print('GEODE: updating rectified distortion encoding; reusing images', flush=True)
                self.create_calibration_yaml(sequence_name)
            reference = self.sequence_path(sequence_name) / 'reference_metadata.json'
            reference_metadata = json.loads(reference.read_text()) if reference.is_file() else {}
            if (self.groundtruth_csv_path(sequence_name).is_file()
                    and (reference_metadata.get('local_origin_version') != 1
                         or reference_metadata.get('orientation_convention_version') != GT_ORIENTATION_VERSION)):
                print('GEODE: updating local GT origin and orientation convention; reusing images and IMU', flush=True)
                self.create_groundtruth_csv(sequence_name)
        super().download_sequence(sequence_name)

    def _ros(self, script, *args):
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        subprocess.run(['pixi', 'run', '--frozen', '-e', 'ros1', 'python',
                        str(VSLAM_LAB_DIR / 'Datasets/extra-files' / script), *map(str, args)],
                       check=True, cwd=VSLAM_LAB_DIR, env=env)

    def check_sequence_integrity(self, sequence_name, verbose=False):
        return (super().check_sequence_integrity(sequence_name, verbose)
                and self.groundtruth_csv_path(sequence_name).is_file()
                and (self.sequence_path(sequence_name) / 'import_complete.json').is_file()
                and not self._old_rectification(sequence_name))

    def check_sequence_availability(self, sequence_name, verbose=True):
        folder = self.sequence_path(sequence_name)
        if (folder / '.import_pending').exists() or self._old_rectification(sequence_name):
            return 'non-available'
        return super().check_sequence_availability(sequence_name, verbose)

    def download_sequence_data(self, sequence_name):
        if not self.contains_sequence(sequence_name):
            raise ValueError(f'Unsupported GEODE sequence: {sequence_name}')
        folder = self.sequence_path(sequence_name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / '.import_pending').touch()
        self._raw().mkdir(parents=True, exist_ok=True)
        for asset in self.assets_for(sequence_name).values():
            path = self._raw() / asset['name']
            # Reuse local files only after full checksum validation. HTTP download
            # uses .part and atomically renames after size + checksum verification.
            if not verified(path, asset):
                url = ('https://drive.usercontent.google.com/download?export=download&confirm=t&id='
                       + asset['drive_id'])
                downloadFile(url, str(self._raw()), asset['size'], filename=asset['name'],
                             checksum=asset_checksum(asset))
        (folder / 'sources.json').write_text(json.dumps(self.assets_for(sequence_name), indent=2))

    def create_rgb_folder(self, sequence_name):
        if sequence_name == 'offroad1_raw_radtan5':
            # Both raw variants have exactly the same grayscale pixels. Reuse
            # already extracted images, without applying the calibration fit.
            source = self.sequence_path('offroad1_raw_radtan4')
            if self.check_sequence_availability('offroad1_raw_radtan4', verbose=False) == 'available':
                geometry = json.loads((source / 'image_geometry.json').read_text())
                if geometry['source_calibration_sha256'] != self.assets_for(sequence_name)['calibration']['sha256']:
                    raise ValueError('Raw image reuse requires the same source calibration')
                folder = self.sequence_path(sequence_name)
                use_hardlinks = True
                links = copies = 0
                for i in (0, 1):
                    (folder / f'rgb_{i}').mkdir(exist_ok=True)
                    index = source / f'rgb_{i}.csv'
                    with index.open() as stream:
                        for row in list(csv.reader(stream))[1:]:
                            original, target = source / row[1], folder / row[1]
                            if target.exists():
                                if os.path.samefile(original, target):
                                    links += 1
                                elif original.read_bytes() == target.read_bytes():
                                    copies += 1
                                else:
                                    raise FileExistsError(f'Refusing to replace image: {target}')
                            else:
                                if use_hardlinks:
                                    try:
                                        os.link(original, target)
                                        links += 1
                                        continue
                                    except OSError as error:
                                        if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV):
                                            raise
                                        use_hardlinks = False  # e.g. exFAT on the data SSD.
                                shutil.copyfile(original, target)
                                copies += 1
                    (folder / index.name).write_bytes(index.read_bytes())
                geometry['image_reuse'] = dict(method='hard links or exact byte copies',
                    source_sequence='offroad1_raw_radtan4', linked_images=links, copied_images=copies)
                (folder / 'image_geometry.json').write_text(json.dumps(geometry, indent=2))
                print('GEODE: reused native-geometry grayscale images; original radtan5 calibration is separate', flush=True)
                return
        self._ros('extract_geode_stereo.py', '--bag', self._asset('bag', sequence_name),
                  '--calibration', self._asset('calibration', sequence_name), '--output', self.sequence_path(sequence_name),
                  *(['--unrectified'] if sequence_name in ('offroad1_raw_radtan4', 'offroad1_raw_radtan5') else []))

    def create_rgb_csv(self, sequence_name):
        streams = []
        folder = self.sequence_path(sequence_name)
        for i in (0, 1):
            with (folder / f'rgb_{i}.csv').open() as stream:
                streams.append(list(csv.reader(stream))[1:])
        rows = stereo_pairs(*streams, tolerance_ns=self.cfg['stereo_tolerance_ns'])
        write_csv_rows(self.rgb_csv_path(sequence_name),
                       ['ts_rgb_0 (ns)', 'path_rgb_0', 'ts_rgb_1 (ns)', 'path_rgb_1'], rows)
        (folder / 'stereo_report.json').write_text(json.dumps({
            'input_counts': list(map(len, streams)), 'paired_count': len(rows),
            'unpaired_counts': [len(s) - len(rows) for s in streams],
            'tolerance_ns': self.cfg['stereo_tolerance_ns'],
            'max_pair_offset_ns': max(abs(int(r[0]) - int(r[2])) for r in rows)}, indent=2))

    def create_imu_csv(self, sequence_name):
        self._ros('extract_rosbag_imu.py', '--rosbag_path', self._asset('bag', sequence_name),
                  '--sequence_path', self.sequence_path(sequence_name), '--imu_topic', '/imu/data', '--strict')

    def create_calibration_yaml(self, sequence_name):
        if sequence_name == 'offroad1_raw_radtan5':
            return self.create_original_calibration(sequence_name)
        if sequence_name == 'offroad1_raw_radtan4':
            return self.create_unrectified_calibration(sequence_name)
        calibration = read_calibration(self._asset('calibration', sequence_name))
        rectification = json.loads((self.sequence_path(sequence_name) / 'rectification.json').read_text())
        if rectification['source_calibration_sha256'] != self.assets_for(sequence_name)['calibration']['sha256']:
            raise ValueError('Extracted images use a different source calibration')
        projections = [np.array(P) for P in rectification['projections']]
        transforms = [np.array(T) for T in rectification['camera_T_BS']]
        T_BI = np.array(rectification['imu_T_BS'])
        metadata = rectification['metadata']
        metadata['rectification_opencv_version'] = rectification['opencv_version']
        metadata['output_distortion_representation'] = 'radtan4 with four zero coefficients; rectified pinhole geometry'
        cams = [dict(cam_name=f'rgb_{i}', cam_type='gray', cam_model='pinhole',
                     # The packaged ORB-SLAM3 reader rejects distortion_type=pinhole.
                     # Zero radtan coefficients encode exactly the same rectified rays.
                     distortion_type='radtan4', distortion_coefficients=[0., 0., 0., 0.],
                     focal_length=[float(P[0, 0]), float(P[1, 1])],
                     principal_point=[float(P[0, 2]), float(P[1, 2])], fps=self.rgb_hz, T_BS=T)
                for i, (P, T) in enumerate(zip(projections, transforms))]
        imu = dict(imu_name='imu_0', fps=100., g=9.80665, T_BS=T_BI, **calibration['noise'])
        self.write_calibration_yaml(sequence_name, rgb=cams, imu=[imu])
        (self.sequence_path(sequence_name) / 'calibration_metadata.json').write_text(json.dumps(metadata, indent=2))

    def create_original_calibration(self, sequence_name):
        folder = self.sequence_path(sequence_name)
        extraction = json.loads((folder / 'image_geometry.json').read_text())
        if extraction['source_calibration_sha256'] != self.assets_for(sequence_name)['calibration']['sha256']:
            raise ValueError('Extracted images use a different source calibration')
        source = read_calibration(self._asset('calibration', sequence_name))
        _, T_BI = body_transforms(source)
        cameras = []
        for i, camera in enumerate(source['cameras']):
            p, d = camera['projection_parameters'], camera['distortion_parameters']
            cameras.append(dict(cam_name=f'rgb_{i}', cam_type='gray', cam_model='pinhole',
                distortion_type='radtan5',
                distortion_coefficients=[float(d[k]) for k in ('k1', 'k2', 'p1', 'p2', 'k3')],
                focal_length=[float(p['fx']), float(p['fy'])],
                principal_point=[float(p['cx']), float(p['cy'])], fps=self.rgb_hz,
                T_BS=T_BI @ matrix(source['transforms']['T_IMU_CamL' if i == 0 else 'T_IMU_CamR'])))
        self.write_calibration_yaml(sequence_name, rgb=cameras,
            imu=[dict(imu_name='imu_0', fps=100., g=9.80665, T_BS=T_BI, **source['noise'])])
        (folder / 'calibration_metadata.json').write_text(json.dumps(dict(
            body_frame='publisher beta reference frame',
            camera_chain='Original T_IMU_CamL and T_IMU_CamR; no rectification rotations',
            image_processing=extraction['image_processing'],
            distortion_model='Original published radtan5 [k1,k2,p1,p2,k3]; no fitting or truncation',
            source_calibration_sha256=extraction['source_calibration_sha256'],
            calibration_status='Publisher calibration; GT/inter-rig calibration limitations remain'), indent=2))

    def create_unrectified_calibration(self, sequence_name):
        from Datasets.dataset_files.geode_radtan_fit import fit_camera
        folder = self.sequence_path(sequence_name)
        extraction = json.loads((folder / 'image_geometry.json').read_text())
        if extraction['source_calibration_sha256'] != self.assets_for(sequence_name)['calibration']['sha256']:
            raise ValueError('Extracted images use a different source calibration')
        source = read_calibration(self._asset('calibration', sequence_name))
        _, T_BI = body_transforms(source)
        report = dict(source_calibration_sha256=extraction['source_calibration_sha256'],
                      image_processing=extraction['image_processing'],
                      approximation='Model approximation to published radtan5, not a new physical camera calibration',
                      cameras=[fit_camera(camera) for camera in source['cameras']])
        cams = []
        for i, fitted in enumerate(report['cameras']):
            selected = fitted['candidates'][fitted['selected']]
            K = np.array(selected['K'])
            cams.append(dict(cam_name=f'rgb_{i}', cam_type='gray', cam_model='pinhole',
                distortion_type='radtan4', distortion_coefficients=selected['D'],
                focal_length=[float(K[0, 0]), float(K[1, 1])],
                principal_point=[float(K[0, 2]), float(K[1, 2])], fps=self.rgb_hz,
                T_BS=T_BI @ matrix(source['transforms']['T_IMU_CamL' if i == 0 else 'T_IMU_CamR'])))
        self.write_calibration_yaml(sequence_name, rgb=cams,
            imu=[dict(imu_name='imu_0', fps=100., g=9.80665, T_BS=T_BI, **source['noise'])])
        (folder / 'radtan4_fit.json').write_text(json.dumps(report, indent=2))
        (folder / 'calibration_metadata.json').write_text(json.dumps({
            'body_frame': 'publisher beta reference frame',
            'camera_chain': 'Original T_IMU_CamL and T_IMU_CamR; no rectification rotations',
            'image_processing': report['image_processing'],
            'distortion_model': 'Refitted radtan4; full error report in radtan4_fit.json',
            'calibration_status': 'Experimental approximation; publisher GT/inter-rig calibration limitations remain',
        }, indent=2))

    def create_groundtruth_csv(self, sequence_name):
        rows, metadata = reference_rows(self._asset('reference', sequence_name))
        # Always derive both transformations from the retained source, never
        # from prepared GT: repeated preparation must not apply them twice.
        origin = np.array(rows[0][1:4])
        origin_timestamp = rows[0][0]
        orientations = reference_orientations(np.array([row[4:] for row in rows]))
        rows = [[row[0], *(np.array(row[1:4]) - origin).tolist(), *q.tolist()]
                for row, q in zip(rows, orientations)]
        write_csv_rows(self.groundtruth_csv_path(sequence_name),
                       ['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw'], rows)
        metadata.update(source=f"published {self.assets_for(sequence_name)['reference']['name']}, GNSS/INS-RTK",
                        source_checksum=asset_checksum(self.assets_for(sequence_name)['reference']),
                        body_frame='publisher beta reference frame',
                        pose_transform='p_local = p_source - origin_xyz_m; orientation convention applied separately; position axes unchanged',
                        local_origin_version=1,
                        orientation_convention_version=GT_ORIENTATION_VERSION,
                        orientation_transform='R_new = Rz(90deg) Rx(180deg) R_source Rx(180deg)',
                        orientation_euler_equivalent='extrinsic xyz: roll_new=roll, pitch_new=-pitch, yaw_new=90deg-yaw',
                        orientation_convention_status='Pipeline convention supported by gyro and saved-trajectory diagnostics; original GNSS convention not independently confirmed',
                        fitted_orientation_bias_applied=False,
                        origin_timestamp_ns=origin_timestamp,
                        origin_xyz_m=origin.tolist(),
                        independent_of_visual_frontend=True,
                        world_frame='local origin at earliest timestamp-sorted GT pose; publisher axis directions and metre scale preserved; source CRS unspecified',
                        publisher_timestamp_issue='https://github.com/PengYu-Team/GEODE_dataset/issues/5',
                        evaluation_status='provisional: source timing and inter-rig calibration require validation')
        (self.sequence_path(sequence_name) / 'reference_metadata.json').write_text(json.dumps(metadata, indent=2))

    def remove_unused_files(self, sequence_name):
        folder = self.sequence_path(sequence_name)
        (folder / 'import_complete.json').write_text(json.dumps({'format_version': 1}))
        (folder / '.import_pending').unlink()
