"""BotanicGarden VLIO stereo/Xsens adapter. License: GPLv3.

Implemented for vio_research with assistance from Codex, 2026-10-13.
Native streams and corrected VLP16 reference are converted to an Xsens body frame.
"""

import csv
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from Datasets.DatasetVSLAMLAB import DatasetVSLAMLAB
from path_constants import VSLAM_LAB_DIR
from utilities import (downloadFile, download_onedrive_file,
                       run_rosbag_frame_extraction, write_csv_rows)


def read_calibration(path):
    class OpenCVLoader(yaml.SafeLoader):
        pass

    OpenCVLoader.add_constructor(
        "tag:yaml.org,2002:opencv-matrix", lambda loader, node: loader.construct_mapping(node, deep=True)
    )
    return yaml.load(path.read_text().replace("%YAML:1.0", ""), Loader=OpenCVLoader)


def matrix(record):
    value = np.array(record["data"], dtype=float).reshape(record["rows"], record["cols"])
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1])
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Invalid rigid calibration transform")
    return value


def stereo_pairs(left, right, tolerance_ns):
    # Ordered, one-to-one nearest matches; never shift pairs by row after a dropped frame.
    lt, rt = [int(r[0]) for r in left], [int(r[0]) for r in right]
    for stamps in (lt, rt):
        if not stamps or stamps[0] <= 0 or np.any(np.diff(stamps) <= 0):
            raise ValueError("Camera stamps must increase")
    rows, j = [], 0
    for row, t in zip(left, lt):
        while j + 1 < len(rt) and abs(rt[j + 1] - t) < abs(rt[j] - t):
            j += 1
        if j < len(rt) and abs(rt[j] - t) <= tolerance_ns:
            rows.append([t, row[1], rt[j], right[j][1]])
            j += 1
        if j == len(rt):
            break
    if not rows:
        raise ValueError("No synchronized stereo pairs")
    return rows


def verified(path, asset):
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return False
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != asset["md5"]:
        raise ValueError(f"Checksum mismatch: {path}; file preserved")
    return True


class BotanicgardenDataset(DatasetVSLAMLAB):
    def __init__(self, dataset_name="botanicgarden"):
        super().__init__(dataset_name)

    def _raw(self):
        return self.dataset_path / "raw"

    def _ros(self, script, *args):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                "pixi",
                "run",
                "--frozen",
                "-e",
                "ros1",
                "python",
                str(VSLAM_LAB_DIR / "Datasets/extra-files" / script),
                *map(str, args),
            ],
            check=True,
            cwd=VSLAM_LAB_DIR,
            env=env,
        )

    def check_sequence_integrity(self, sequence_name, verbose=False):
        return (
            super().check_sequence_integrity(sequence_name, verbose)
            and self.groundtruth_csv_path(sequence_name).is_file()
            and (self.sequence_path(sequence_name) / "import_complete.json").is_file()
        )

    def check_sequence_availability(self, sequence_name, verbose=True):
        status = super().check_sequence_availability(sequence_name, verbose=False)
        folder = self.sequence_path(sequence_name)
        if (
            status == "corrupted"
            and (folder / "sources.json").is_file()
            and not (folder / "import_complete.json").exists()
        ):
            return "non-available"
        return status

    def download_sequence_data(self, sequence_name):
        if not self.contains_sequence(sequence_name):
            raise ValueError(f"Unsupported sequence: {sequence_name}")
        raw = self._raw()
        raw.mkdir(parents=True, exist_ok=True)
        self.sequence_path(sequence_name).mkdir(parents=True, exist_ok=True)
        # An explicit local source directory allows reuse without copying multi-GB bags.
        local = Path(os.environ.get("BOTANICGARDEN_SOURCE_DIR", str(raw)))
        assets = {
            f"{sequence_name}_VLIO.bag": self.cfg["bags"][sequence_name],
            f"gt_{sequence_name}_qnew.bag": self.cfg["groundtruth_bags"][sequence_name],
        }
        paths = {}
        providers = {}
        # Own the partial import before network IO so a failed download can be retried.
        sources = self.sequence_path(sequence_name) / "sources.json"
        if not sources.exists():
            sources.write_text(json.dumps({"files": {}, "checksums": assets}))
        for name, asset in assets.items():
            # Resume an explicitly configured local partial in place rather than
            # requiring manual downloading or creating a second multi-GB copy.
            if (local / name).exists():
                destination = local
            elif (raw / name).exists():
                destination = raw
            else:
                def partial_size(directory):
                    partial = directory / (name + ".part")
                    return partial.stat().st_size if partial.exists() else 0
                destination = max([raw, local], key=partial_size)
            path = destination / name
            if not verified(path, asset):
                if path.exists():
                    raise ValueError(f"Incomplete source: {path}; resume download before importing")
                providers[name] = "onedrive"
                download_onedrive_file(asset["onedrive"], str(destination), asset["size"],
                                       filename=name, checksum="md5:" + asset["md5"])
            else:
                providers[name] = "local_verified"
            paths[name] = str(path.resolve())
        calib = raw / "calib"
        for relative in self.cfg["calibration_files"]:
            path = calib / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                downloadFile(self.cfg["calibration_root"] + "/" + relative, str(path.parent))
            expected = self.cfg["calibration_files"][relative]
            content = path.read_bytes()
            blob = b"blob " + str(len(content)).encode() + b"\0" + content
            if len(content) != expected["size"] or hashlib.sha1(blob).hexdigest() != expected["git_sha1"]:
                raise ValueError(f"Calibration source checksum mismatch: {path}; preserved")
        (self.sequence_path(sequence_name) / "sources.json").write_text(
            json.dumps(
                {
                    "files": paths,
                    "download_providers": providers,
                    "calibration_root": self.cfg["calibration_root"],
                    "checksums": assets,
                },
                indent=2,
            )
        )

    def _source(self, sequence_name, name):
        return Path(json.loads((self.sequence_path(sequence_name) / "sources.json").read_text())["files"][name])

    def create_rgb_folder(self, sequence_name):
        for cam, topic in enumerate(["/dalsa_gray/left/image_raw", "/dalsa_gray/right/image_raw"]):
            run_rosbag_frame_extraction(
                "ros1",
                self._source(sequence_name, f"{sequence_name}_VLIO.bag"),
                self.sequence_path(sequence_name),
                topic,
                cam,
                header_stamp=True,
                strict=True,
                separate_csv=True,
            )

    def create_rgb_csv(self, sequence_name):
        streams = []
        for cam in (0, 1):
            with (self.sequence_path(sequence_name) / f"rgb_{cam}.csv").open() as f:
                reader = csv.reader(f)
                next(reader)
                streams.append(list(reader))
        # Author documentation: hardware-synchronized stereo. Require exact header stamps;
        # a real offset needs investigation rather than borrowing GrandTour's 2 ms tolerance.
        rows = stereo_pairs(*streams, tolerance_ns=0)
        write_csv_rows(
            self.rgb_csv_path(sequence_name), ["ts_rgb_0 (ns)", "path_rgb_0", "ts_rgb_1 (ns)", "path_rgb_1"], rows
        )
        (self.sequence_path(sequence_name) / "stereo_report.json").write_text(
            json.dumps(
                {
                    "input_counts": list(map(len, streams)),
                    "paired_count": len(rows),
                    "unpaired_counts": [len(s) - len(rows) for s in streams],
                    "tolerance_ns": 0,
                },
                indent=2,
            )
        )

    def create_imu_csv(self, sequence_name):
        self._ros(
            "extract_rosbag_imu.py",
            "--rosbag_path",
            self._source(sequence_name, f"{sequence_name}_VLIO.bag"),
            "--sequence_path",
            self.sequence_path(sequence_name),
            "--imu_topic",
            "/imu/data",
            "--strict",
        )

    def create_calibration_yaml(self, sequence_name):
        root = self._raw() / "calib"
        chain = read_calibration(root / "extrinsics/calib_chain.yaml")
        transforms = [matrix(chain["T_xsens_gray0"])]
        transforms.append(transforms[0] @ matrix(chain["T_gray0_gray1"]))
        cams = []
        for cam in (0, 1):
            c = read_calibration(root / f"camera_intrinsics/dalsa_gray{cam}_down.yaml")
            p, d = c["projection_parameters"], c["distortion_parameters"]
            from PIL import Image

            with Image.open(next((self.sequence_path(sequence_name) / f"rgb_{cam}").glob("*.png"))) as image:
                if image.size != (c["image_width"], c["image_height"]):
                    raise ValueError("Calibration resolution mismatch")
            cams.append(
                {
                    "cam_name": f"rgb_{cam}",
                    "cam_type": "gray",
                    "cam_model": "pinhole",
                    "distortion_type": "radtan4",
                    "focal_length": [float(p["fx"]), float(p["fy"])],
                    "principal_point": [float(p["cx"]), float(p["cy"])],
                    "distortion_coefficients": [float(d[k]) for k in ["k1", "k2", "p1", "p2"]],
                    "fps": self.rgb_hz,
                    "T_BS": transforms[cam],
                }
            )
        noise = read_calibration(root / "imu_intrinsics/xsens_imu_param.yaml")
        imu = {
            "imu_name": "imu_0",
            "sigma_g_c": noise["gyroscope_noise_density"],
            "sigma_gw_c": noise["gyroscope_random_walk"],
            "sigma_a_c": noise["accelerometer_noise_density"],
            "sigma_aw_c": noise["accelerometer_random_walk"],
            "fps": float(noise["update_rate"]),
            "g": 9.80665,
            "T_BS": np.eye(4),
        }
        self.write_calibration_yaml(sequence_name, rgb=cams, imu=[imu])

    def create_groundtruth_csv(self, sequence_name):
        folder = self.sequence_path(sequence_name)
        bag = self._raw() / f"gt_{sequence_name}_qnew.bag"
        if not bag.exists():
            sources = json.loads((folder / "sources.json").read_text())["files"]
            if bag.name in sources:
                bag = Path(sources[bag.name])
            else:
                # Imports started before direct OneDrive delivery used the corrected ZIP.
                with zipfile.ZipFile(self._source(sequence_name, "GT_pose_qnew.zip")) as z:
                    content = z.read(f"GT_pose_qnew/gt_{sequence_name}_qnew.bag")
                    temporary = bag.with_suffix(".bag.tmp")
                    temporary.write_bytes(content)
                    temporary.replace(bag)
        raw_csv = folder / "groundtruth_vlp16.csv"
        self._ros("extract_rosbag_poses.py", "--rosbag_path", bag, "--output", raw_csv, "--pose_topic", "/gt_poses")
        with raw_csv.open() as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        values = np.array([[float(v) for v in row[1:]] for row in rows])
        finite = np.isfinite(values).all(axis=1)
        rejected_stamps = [int(row[0]) for row, keep in zip(rows, finite) if not keep]
        rows = [row for row, keep in zip(rows, finite) if keep]
        values = values[finite]
        if len(rows) < 3 or np.any(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1) > 0.005):
            raise ValueError("Invalid reference quaternion/position")
        chain = read_calibration(self._raw() / "calib/extrinsics/calib_chain.yaml")
        T = np.linalg.inv(matrix(chain["T_xsens_vlp16"]))
        rotation = Rotation.from_quat(values[:, 3:])
        positions = values[:, :3] + rotation.apply(T[:3, 3])
        quaternions = (rotation * Rotation.from_matrix(T[:3, :3])).as_quat()
        write_csv_rows(
            self.groundtruth_csv_path(sequence_name),
            ["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"],
            [[int(row[0]), *p, *q] for row, p, q in zip(rows, positions, quaternions)],
        )
        (folder / "reference_metadata.json").write_text(
            json.dumps(
                {
                    "source": f"corrected gt_{sequence_name}_qnew.bag /gt_poses",
                    "rejected_nonfinite_pose_timestamps_ns": rejected_stamps,
                    "body_frame": "Xsens",
                    "source_tracked_frame": "VLP16",
                    "transform": "T_world_xsens = T_world_vlp16 @ inverse(T_xsens_vlp16)",
                    "T_xsens_vlp16": matrix(chain["T_xsens_vlp16"]).tolist(),
                    "independent_of_visual_frontend": False,
                    "world_frame": "sequence-local surveyed map",
                    "imu_noise": "Published Kalibr densities/random walks; no invented bias priors; g=9.80665 convention",
                },
                indent=2,
            )
        )

    def remove_unused_files(self, sequence_name):
        # Preserve source bags, calibration and the original reference, regardless of retention.
        (self.sequence_path(sequence_name) / "import_complete.json").write_text(json.dumps({"format_version": 1}))
