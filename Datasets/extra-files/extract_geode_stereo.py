"""Extract grayscale GEODE gamma stereo, with optional rectification, in one bag pass."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import rosbag
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Datasets.dataset_files.geode_calibration import read_calibration, geometry

TOPICS = {'/left_camera/image/compressed': 0, '/right_camera/image/compressed': 1}


def extract(bag_path, calibration, folder, unrectified=False):
    if unrectified:
        camera = read_calibration(calibration)['cameras'][0]
        size = (int(camera['image_width']), int(camera['image_height']))
    else:
        size, maps, projections, transforms, T_BI, metadata = geometry(read_calibration(calibration))
    folder.mkdir(parents=True, exist_ok=True)
    for i in (0, 1):
        (folder / f'rgb_{i}').mkdir(exist_ok=True)
    rows = [[], []]
    frames = [set(), set()]
    with rosbag.Bag(str(bag_path), 'r') as bag:
        count = bag.get_message_count(topic_filters=list(TOPICS))
        print(f'GEODE: extracting {count} images from {bag_path} to {folder}', flush=True)
        for number, (topic, msg, _) in enumerate(tqdm(
                bag.read_messages(topics=list(TOPICS)), total=count,
                desc='GEODE stereo extraction', disable=not sys.stderr.isatty()), 1):
            cam = TOPICS[topic]
            if msg._type != 'sensor_msgs/CompressedImage':
                raise ValueError(f'Unexpected image type on {topic}: {msg._type}')
            timestamp = msg.header.stamp.to_nsec()
            if timestamp <= 0 or (rows[cam] and timestamp <= rows[cam][-1][0]):
                raise ValueError(f'Image header stamps must increase: {topic}')
            frames[cam].add(msg.header.frame_id)
            if len(frames[cam]) > 1:
                raise ValueError(f'Camera frame_id changes: {topic}')
            # Decode BGR first, then OpenCV luminance conversion. JPEG decoder
            # grayscale mode can differ from the documented RGB-to-gray path.
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[1::-1] != size:
                raise ValueError(f'Invalid source image/dimensions: {topic} {timestamp}')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output_image = gray if unrectified else cv2.remap(gray, *maps[cam], cv2.INTER_LINEAR)
            relative = f'rgb_{cam}/{timestamp}.png'
            path = folder / relative
            temporary = path.with_name(path.stem + '.tmp.png')
            if not cv2.imwrite(str(temporary), output_image):
                raise OSError(f'Could not write {temporary}')
            temporary.replace(path)
            rows[cam].append([timestamp, relative])
            if number % 500 == 0 or number == count:
                print(f'GEODE: extracted {number}/{count} images', flush=True)
    for cam, stream in enumerate(rows):
        if not stream:
            raise ValueError(f'No images for camera {cam}')
        target = folder / f'rgb_{cam}.csv'
        temporary = target.with_suffix('.csv.tmp')
        with temporary.open('w') as output:
            writer = csv.writer(output)
            writer.writerow([f'ts_rgb_{cam} (ns)', f'path_rgb_{cam}'])
            writer.writerows(stream)
        temporary.replace(target)
    # Persist the geometry actually used for the pixels. The ROS1 and main
    # environments can have different OpenCV versions; never recompute it there.
    target = folder / ('image_geometry.json' if unrectified else 'rectification.json')
    temporary = target.with_suffix('.json.tmp')
    report = {
        'opencv_version': cv2.__version__,
        'source_calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
        'image_processing': 'BGR to grayscale PNG; no warp, crop or resize' if unrectified else 'grayscale stereo rectification',
    }
    if not unrectified:
        report.update(projections=[P.tolist() for P in projections],
                      camera_T_BS=[T.tolist() for T in transforms],
                      imu_T_BS=T_BI.tolist(), metadata=metadata)
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--unrectified', action='store_true', help='Preserve native pixel geometry; only convert to grayscale PNG')
    args = parser.parse_args()
    extract(args.bag, args.calibration, args.output, args.unrectified)
