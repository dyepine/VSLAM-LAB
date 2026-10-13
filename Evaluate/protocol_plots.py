"""Static comparison figures from published evo results and saved pose files."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evo.tools import file_interface
from Evaluate.trajectory_inputs import read_pose_csv, to_evo, camera_to_body

METRICS = [('ape_position_m', 'APE position [m]'), ('rpe_position_m', 'RPE position [m]'),
           ('ape_orientation_deg', 'APE orientation [deg]'), ('rpe_orientation_deg', 'RPE orientation [deg]')]


def full_paths(summary):
    """Apply the saved alignment to observed poses; never refit or interpolate."""
    origin = summary['time_origin_ns']
    trajectories = []
    for key in ['trajectory', 'groundtruth']:
        path = Path(summary[key])
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary[key + '_sha256']:
            raise ValueError(f'Plot input changed since evaluation: {path}')
        timestamps, values = read_pose_csv(path)
        intervals = summary['protocol']['gt_valid_intervals_ns']
        if intervals:
            mask = np.zeros(len(timestamps), dtype=bool)
            for start, end in intervals:
                mask |= (timestamps >= start) & (timestamps <= end)
            timestamps, values = timestamps[mask], values[mask]
        trajectories.append(to_evo(timestamps, values, origin))
    estimate, reference = trajectories
    if summary['protocol']['estimate_frame'] == 'camera':
        calibration = Path(summary['calibration'])
        if hashlib.sha256(calibration.read_bytes()).hexdigest() != summary['calibration_sha256']:
            raise ValueError('Plot calibration changed since evaluation')
        camera_to_body(estimate, calibration, summary['protocol']['camera'])
    estimate.transform(np.asarray(summary['alignment_transform_gt_from_estimate']))
    # Show measured GT over the saved estimate's time span, not the rest of a long recording.
    ids = np.flatnonzero((reference.timestamps >= estimate.timestamps[0]) &
                         (reference.timestamps <= estimate.timestamps[-1]))
    reference.reduce_to_ids(ids)
    return estimate, reference


def segments(times, summary):
    gap = summary['protocol']['max_pose_gap_s']
    breaks = set((np.flatnonzero(np.diff(times) > gap) + 1).tolist())
    for interval in (summary['protocol']['gt_valid_intervals_ns'] or [])[1:]:
        boundary = (interval[0] - summary['time_origin_ns']) / 1e9
        index = int(np.searchsorted(times, boundary))
        if 0 < index < len(times): breaks.add(index)
    return np.split(np.arange(len(times)), sorted(breaks))


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=160)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def plot_comparison(table):
    table = Path(table)
    with table.open() as stream:
        records = list(csv.DictReader(stream))
    output = table.parent / 'figures'
    output.mkdir(exist_ok=True)
    groups = {}
    for row in records:
        groups.setdefault((row['dataset'], row['sequence']), []).append(row)
    manifest = []
    for (dataset, sequence), rows in groups.items():
        tag = re.sub(r'[^A-Za-z0-9_-]', '_', f'{dataset}_{sequence}')
        available = []
        for row in rows:
            if row['status'] != 'evaluated': continue
            directory = Path(row['evaluation_directory'])
            summary = json.loads((directory / 'summary.json').read_text())
            label = f"{row['experiment']} / repeat {int(row['iteration']) + 1}"
            metrics = {name: file_interface.load_res_file(directory / (name + '.zip'))
                       for name, _ in METRICS if (directory / (name + '.zip')).is_file()}
            estimate, reference = full_paths(summary)
            available.append((label, summary, metrics, estimate, reference))
        if not available:
            manifest.append(dict(dataset=dataset, sequence=sequence, plots=[], statuses=rows))
            continue
        colors = plt.get_cmap('tab10').colors
        color_by_label = {f"{row['experiment']} / repeat {int(row['iteration']) + 1}": colors[i % 10]
                          for i, row in enumerate(rows)}
        # Each method keeps the alignment fitted in its own evaluation.
        fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
        # One measured GT over the union of the methods' saved-pose time spans.
        if len({summary['groundtruth_sha256'] for _, summary, *_ in available}) != 1:
            raise ValueError('Cannot draw a shared GT for different reference trajectories')
        gt_origin = min(summary['time_origin_ns'] for _, summary, *_ in available)
        gt_times = np.concatenate([np.rint(reference.timestamps * 1e9).astype(np.int64)
                                   + summary['time_origin_ns'] - gt_origin
                                   for _, summary, _, _, reference in available])
        gt_positions = np.concatenate([reference.positions_xyz for *_, reference in available])
        times, indices = np.unique(gt_times, return_index=True)
        gt_positions = gt_positions[indices]
        gt_summary = dict(available[0][1], time_origin_ns=gt_origin)
        for part, ids in enumerate(segments(times / 1e9, gt_summary)):
            ax.plot(gt_positions[ids, 0], gt_positions[ids, 1], color='0.3', linestyle='--',
                    linewidth=1.5, label='GT' if part==0 else None, zorder=3)
        for label, summary, _, estimate, _ in available:
            for part, ids in enumerate(segments(estimate.timestamps, summary)):
                ax.plot(estimate.positions_xyz[ids, 0], estimate.positions_xyz[ids, 1],
                        color=color_by_label[label], linewidth=1.2,
                        label=label if part==0 else None)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
        ax.set_title(f'{dataset} / {sequence}: XY trajectory')
        save_figure(fig, output / (tag + '_trajectories'))

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
        common_origin = min(item[1]['time_origin_ns'] for item in available)
        for ax, (metric, title) in zip(axes.flat, METRICS):
            count = 0
            for index, (label, summary, results, _, _) in enumerate(available):
                if metric not in results: continue
                result = results[metric]
                relative_time = np.asarray(result.np_arrays['timestamps'])
                offset = (summary['time_origin_ns'] - common_origin) / 1e9
                if metric.startswith('rpe_'):
                    # all_pairs can produce several errors at the same end timestamp.
                    ax.scatter(relative_time + offset, result.np_arrays['error_array'], s=10,
                               color=color_by_label[label],
                               label=f"{label}, GT delta={summary['protocol']['rpe_delta_m']:g} m")
                else:
                    for part, ids in enumerate(segments(relative_time, summary)):
                        ax.plot(relative_time[ids] + offset, result.np_arrays['error_array'][ids],
                                color=color_by_label[label], marker='.', markersize=3, linewidth=.8,
                                label=label if part==0 else None)
                count += 1
            ax.set_title(title)
            ax.set_xlabel('Time from common evaluation origin [s]')
            ax.set_ylabel(title)
            ax.grid(alpha=.25)
            if count: ax.legend(fontsize=8)
            else: ax.text(.5, .5, 'Metric unavailable', ha='center', transform=ax.transAxes)
        fig.suptitle(f'{dataset} / {sequence}: evo errors on associated saved poses')
        save_figure(fig, output / (tag + '_errors'))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
        labels = [f"{row['experiment']}\nr{int(row['iteration'])+1}" for row in rows]
        for ax, metric, title in [(axes[0], 'ape_position_m_rmse', 'APE position RMSE [m]'),
                                  (axes[1], 'rpe_position_m_rmse', 'RPE position RMSE [m]'),
                                  (axes[2], 'matched_fraction_of_saved_poses', 'Matched saved poses [%]')]:
            for i, row in enumerate(rows):
                value = row.get(metric, '')
                if row['status']=='evaluated' and value:
                    height = float(value) * (100 if metric.startswith('matched') else 1)
                    ax.bar(i, height, color=colors[i % 10])
                    ax.annotate(f'{height:.3g}', (i, height), xytext=(0, 4), textcoords='offset points', ha='center')
                else:
                    ax.text(i, .02, row['status'] if row['status']!='evaluated' else 'unavailable',
                            rotation=90, ha='center', transform=ax.get_xaxis_transform(), fontsize=8)
            ax.set_xticks(range(len(rows)), labels, rotation=25, ha='right', fontsize=8)
            ax.set_title(title); ax.grid(axis='y', alpha=.25)
            if metric.startswith('matched'): ax.set_ylim(0, 110)
        protocols = sorted({f"{s['protocol']['align']}, scale={s['protocol']['correct_scale']}, dt={s['protocol']['max_time_difference_s']:g}s, RPE delta={s['protocol']['rpe_delta_m']:g}m" for _, s, *_ in available})
        fig.suptitle(f'{dataset} / {sequence}: ' + '; '.join(protocols))
        save_figure(fig, output / (tag + '_summary'))
        manifest.append(dict(dataset=dataset, sequence=sequence,
                             plots=[tag + suffix + '.png' for suffix in ['_trajectories','_errors','_summary']],
                             evaluations=[str(Path(row['evaluation_directory'])) for row in rows if row['status']=='evaluated']))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return output
