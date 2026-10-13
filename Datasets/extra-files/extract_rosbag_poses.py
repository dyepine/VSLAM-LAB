"""Strict header-stamped PoseStamped extraction for dataset references."""
import argparse
import csv
from pathlib import Path
import rosbag

p = argparse.ArgumentParser()
p.add_argument('--rosbag_path', required=True)
p.add_argument('--output', required=True)
p.add_argument('--pose_topic', required=True)
a = p.parse_args()
rows = []
frames = set()
with rosbag.Bag(a.rosbag_path) as bag:
    for _, msg, _ in bag.read_messages(topics=[a.pose_topic]):
        t = msg.header.stamp.to_nsec()
        if t <= 0 or (rows and t <= rows[-1][0]):
            raise ValueError('Reference stamps must increase')
        frames.add(msg.header.frame_id)
        pos, quat = msg.pose.position, msg.pose.orientation
        rows.append([t, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w])
if not rows or len(frames) != 1:
    raise ValueError('Missing poses or inconsistent reference frame')
path = Path(a.output)
with path.with_suffix('.tmp').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['ts (ns)', 'tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw'])
    writer.writerows(rows)
path.with_suffix('.tmp').replace(path)
