#!/usr/bin/env python3
"""Evaluate a saved VSLAM-LAB run with evo, independently of the viewer."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import evo
import numpy as np
import yaml
from evo.core import metrics, sync, filters
from evo.core.units import Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from Evaluate.artifacts import source_directory, create_run_directory
from Evaluate.trajectory_inputs import read_pose_csv, to_evo, camera_to_body

DEFAULT_PROTOCOL = dict(align='SE3', correct_scale=False, max_time_difference_s=0.01,
                        estimate_frame='camera', camera='rgb_0', evaluate_orientation=True,
                        max_pose_gap_s=1.0, rpe_delta_m=1.0, rpe_relative_tolerance=0.1,
                        gt_valid_intervals_ns=None)


def validate_protocol(settings):
    if not isinstance(settings, dict):
        raise ValueError('Evaluation protocol must be a mapping')
    unknown = set(settings) - set(DEFAULT_PROTOCOL)
    if unknown:
        raise ValueError(f'Unknown Evaluation settings: {sorted(unknown)}')
    protocol = dict(DEFAULT_PROTOCOL, **settings)
    if protocol['align'] not in ('SE3', 'none') or protocol['correct_scale'] is not False:
        raise ValueError('Use align: SE3/none and correct_scale: false for metric VIO')
    if protocol['estimate_frame'] not in ('camera', 'body') or not isinstance(protocol['camera'], str):
        raise ValueError('Invalid estimate frame/camera')
    if type(protocol['evaluate_orientation']) is not bool:
        raise ValueError('evaluate_orientation must be boolean')
    for key in ('max_time_difference_s', 'max_pose_gap_s', 'rpe_delta_m', 'rpe_relative_tolerance'):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    intervals = protocol['gt_valid_intervals_ns']
    if intervals is not None:
        if not isinstance(intervals, list) or not intervals:
            raise ValueError('GT intervals must be a non-empty list')
        previous_end = None
        for interval in intervals:
            if (not isinstance(interval, list) or len(interval) != 2 or
                any(type(x) is not int for x in interval) or interval[0] >= interval[1] or
                (previous_end is not None and interval[0] <= previous_end)):
                raise ValueError('GT intervals must be ordered, disjoint integer-ns ranges')
            previous_end = interval[1]
    return protocol


def segments(reference, estimate, gap):
    # Do not bridge gaps in either associated trajectory.
    breaks = np.flatnonzero((np.diff(reference.timestamps) > gap) |
                           (np.diff(estimate.timestamps) > gap)) + 1
    return np.split(np.arange(estimate.num_poses), breaks)


def calculate(estimate, reference, protocol, interval_boundaries=()):
    # Associate separately in quality intervals so nearest matching cannot cross their boundaries.
    bounds = [-np.inf, *interval_boundaries, np.inf]
    paired = []
    for start, end in zip(bounds, bounds[1:]):
        ref, est = copy.deepcopy(reference), copy.deepcopy(estimate)
        ref_ids = np.flatnonzero((ref.timestamps >= start) & (ref.timestamps < end))
        if not len(ref_ids):
            continue
        ref.reduce_to_ids(ref_ids)
        est_ids = np.flatnonzero((est.timestamps >= max(start, ref.timestamps[0])) &
                                (est.timestamps <= ref.timestamps[-1]) & (est.timestamps < end))
        if not len(est_ids):
            continue
        est.reduce_to_ids(est_ids)
        try:
            paired.append(sync.associate_trajectories(ref, est, max_diff=protocol['max_time_difference_s']))
        except sync.SyncException as error:
            if 'found no matching timestamps' not in str(error):
                raise
    if not paired:
        raise ValueError('Fewer than three time-matched poses')
    reference, estimate = [PoseTrajectory3D(
        np.concatenate([pair[i].positions_xyz for pair in paired]),
        np.concatenate([pair[i].orientations_quat_wxyz for pair in paired]),
        np.concatenate([pair[i].timestamps for pair in paired])) for i in (0, 1)]
    if estimate.num_poses < 3:
        raise ValueError('Fewer than three time-matched poses')
    transform = np.eye(4)
    if protocol['align'] == 'SE3':
        r, t, _ = estimate.align(reference, correct_scale=False)
        transform[:3, :3], transform[:3, 3] = r, t
    groups = segments(reference, estimate, protocol['max_pose_gap_s'])
    for boundary in interval_boundaries:
        groups = [part for group in groups for part in
                  (group[estimate.timestamps[group] < boundary], group[estimate.timestamps[group] >= boundary])
                  if len(part)]
    results, unavailable = {}, {}
    relations = {'position_m': metrics.PoseRelation.translation_part}
    if protocol['evaluate_orientation']:
        relations['orientation_deg'] = metrics.PoseRelation.rotation_angle_deg
    for name, relation in relations.items():
        ape = metrics.APE(relation)
        ape.process_data((reference, estimate))
        result = ape.get_result()
        result.add_np_array('timestamps', estimate.timestamps)
        result.add_trajectory('reference', reference)
        result.add_trajectory('estimate', estimate)
        results['ape_' + name] = result
        # RPE pairs are selected by travelled GT distance, within each continuous segment.
        errors, times = [], []
        rpe = metrics.RPE(relation, delta=protocol['rpe_delta_m'], delta_unit=Unit.meters,
                          rel_delta_tol=protocol['rpe_relative_tolerance'], all_pairs=True,
                          pairs_from_reference=True)
        for ids in groups:
            if len(ids) < 2:
                continue
            ref, est = copy.deepcopy(reference), copy.deepcopy(estimate)
            ref.reduce_to_ids(ids)
            est.reduce_to_ids(ids)
            try:
                rpe.process_data((ref, est))
            except filters.FilterException as error:
                if 'produced an empty index list' not in str(error):
                    raise
                continue
            errors.append(rpe.error.copy())
            times.append(est.timestamps[rpe.delta_ids])
        if errors:
            rpe.error = np.concatenate(errors)
            result = rpe.get_result()
            result.add_np_array('timestamps', np.concatenate(times))
            results['rpe_' + name] = result
        else:
            unavailable['rpe_' + name] = 'No distance pairs within continuous matched segments'
    return results, {'alignment_transform_gt_from_estimate': transform.tolist(),
                     'matched_poses': estimate.num_poses,
                     'matched_pairs': estimate.num_poses,
                     'unique_matched_estimate_poses': len(np.unique(estimate.timestamps)),
                     'unique_matched_gt_poses': len(np.unique(reference.timestamps)),
                     'max_matched_time_difference_s': float(np.max(np.abs(reference.timestamps - estimate.timestamps))),
                     'matched_segments': [{'start_s': float(estimate.timestamps[g[0]]),
                                           'end_s': float(estimate.timestamps[g[-1]]), 'poses': len(g)} for g in groups],
                     'unavailable_metrics': unavailable}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_files(trajectory, gt, calibration, evaluation_root, protocol, provenance,
                   execution=None, native_log=None):
    """Evaluate explicit files; all frames, units and association rules are recorded."""
    protocol = validate_protocol(protocol)
    run = evaluation_root.parent
    output = trajectory.parent
    est_ns, est_values = read_pose_csv(trajectory)
    gt_ns, gt_values = read_pose_csv(gt)
    if protocol['evaluate_orientation'] and np.allclose(gt_values[:, 3:], [0, 0, 0, 1]):
        raise ValueError('Identity-only GT orientations: verify reference or disable orientation evaluation explicitly')
    if protocol['gt_valid_intervals_ns'] is not None:
        mask = np.zeros(len(gt_ns), dtype=bool)
        for start, end in protocol['gt_valid_intervals_ns']:
            mask |= (gt_ns >= start) & (gt_ns <= end)
        gt_ns, gt_values = gt_ns[mask], gt_values[mask]
        # Also filter estimates; no matching across excluded GT quality intervals.
        mask = np.zeros(len(est_ns), dtype=bool)
        for start, end in protocol['gt_valid_intervals_ns']:
            mask |= (est_ns >= start) & (est_ns <= end)
        selected_ns, est_values = est_ns[mask], est_values[mask]
    else:
        selected_ns = est_ns
    if min(len(selected_ns), len(gt_ns)) < 3:
        raise ValueError('Fewer than three poses in allowed GT intervals')
    origin_ns = int(min(selected_ns[0], gt_ns[0]))
    estimate, reference = to_evo(selected_ns, est_values, origin_ns), to_evo(gt_ns, gt_values, origin_ns)
    if protocol['estimate_frame'] == 'camera':
        if provenance.get('calibration_sha256') and digest(calibration) != provenance['calibration_sha256']:
            raise ValueError('Run calibration changed since execution')
        estimate = camera_to_body(estimate, calibration, protocol['camera'])
    native_log = native_log or output / 'system_output_00000.txt'
    text = native_log.read_text(errors='replace') if native_log.exists() else ''
    if text.count('Creation of new map with id:') > 1:
        raise ValueError('Multiple independent maps: evaluate a single map explicitly')
    warnings = ['GT quality is not certified; errors describe saved keyframes, not all input images.']
    if not execution or not execution.get('success') or ('exit_code' in execution and execution['exit_code'] != 0):
        warnings.append('Successful completion is not confirmed; partial result.')
    elif 'exit_code' not in execution:
        warnings.append('Completion follows native STATUS/SUCCESS; process exit code was not recorded.')
    if not native_log.exists():
        warnings.append('Native log missing; resets and map changes cannot be checked.')
    for marker in ('Fail to track local map!', 'Not enough motion for initializing', 'Reset map'):
        if marker in text:
            warnings.append(f'{marker}: {text.count(marker)} occurrences')
    boundaries = [] if protocol['gt_valid_intervals_ns'] is None else [
        (start - origin_ns) / 1e9 for start, _ in protocol['gt_valid_intervals_ns'][1:]]
    results, summary = calculate(estimate, reference, protocol, boundaries)
    gaps = np.diff(est_ns).astype(float) / 1e9
    summary.update(protocol=protocol, evo_version=evo.__version__, time_origin_ns=origin_ns,
                   estimate_input_frame=protocol['estimate_frame'], compared_pose_frame='body',
                   evaluation_code_sha256=digest(Path(__file__)),
                   input_adapter_sha256=digest(Path(__file__).with_name('trajectory_inputs.py')),
                   run_directory=str(run), trajectory=str(trajectory), trajectory_sha256=digest(trajectory),
                   groundtruth=str(gt), groundtruth_sha256=digest(gt),
                   estimate_poses=len(est_ns), matched_fraction_of_saved_poses=summary['unique_matched_estimate_poses'] / len(est_ns),
                   saved_pose_span_s=float((est_ns[-1] - est_ns[0]) / 1e9),
                   max_saved_pose_gap_s=float(gaps.max()),
                   saved_pose_gaps=[{'start_ns': int(est_ns[i]), 'end_ns': int(est_ns[i+1])}
                                    for i in np.flatnonzero(gaps > protocol['max_pose_gap_s'])],
                   execution=execution, warnings=warnings,
                   selected_input_images=provenance.get('input_audit', {}).get('images_selected'),
                   metrics={name: result.stats for name, result in results.items()},
                   metric_sample_counts={name: len(result.np_arrays['error_array']) for name, result in results.items()},
                   association='evo associate_trajectories; shorter trajectory queries longer; pose reuse allowed; no boundary extrapolation')
    if protocol['estimate_frame'] == 'camera':
        summary.update(calibration=str(calibration), calibration_sha256=digest(calibration))
    # Publish complete evaluations only; every invocation receives its own immutable directory.
    root = evaluation_root
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=root) as temporary:
        pending = Path(temporary)
        destination = create_run_directory(root)
        published = False
        try:
            for name, result in results.items():
                result.add_info({'title': name, 'est_name': '/'.join([provenance.get('module', 'estimate'), run.name, destination.name]),
                                 'ref_name': 'reference', 'protocol': protocol, 'evo_version': evo.__version__})
                file_interface.save_res_file(pending / (name + '.zip'), result)
            (pending / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            (pending / 'protocol.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
            # Rename the complete directory over the reserved empty directory atomically.
            pending.rename(destination)
            published = True
        finally:
            if not published:
                destination.rmdir()
    return destination


def update_evaluation_log(log_path, dataset, sequence, iteration, destination, summary, error=None):
    """Serialize read/modify/write so evaluating separate runs cannot lose log rows."""
    import fcntl
    if not log_path.exists():
        return
    with log_path.with_suffix('.evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with log_path.open() as stream:
            reader = csv.DictReader(stream)
            rows, headers = list(reader), list(reader.fieldnames)
        for field in ('num_evaluated_frames', 'evaluation_directory', 'evaluation_error'):
            if field not in headers:
                headers.append(field)
        for row in rows:
            if (row['dataset_name'] == dataset and row['sequence_name'] == sequence and
                    int(row['exp_it']) == iteration):
                if error is None:
                    row.update(EVALUATION='evo', num_evaluated_frames=summary['unique_matched_estimate_poses'],
                               evaluation_directory=str(destination), evaluation_error='')
                else:
                    row.update(EVALUATION='failed', num_evaluated_frames=0,
                               evaluation_directory='', evaluation_error=error)
        with tempfile.NamedTemporaryFile(mode='w', dir=log_path.parent, delete=False) as stream:
            temporary_log = Path(stream.name)
            try:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            except BaseException:
                temporary_log.unlink()
                raise
        try:
            temporary_log.replace(log_path)
        finally:
            temporary_log.unlink(missing_ok=True)


def evaluate_run(run, protocol_override=None, gt_override=None):
    provenance = json.loads((run / 'run_provenance.json').read_text())
    if protocol_override:
        settings = yaml.safe_load(Path(protocol_override).read_text())
    else:
        settings = provenance.get('evaluation_protocol')
        if settings is None:
            raise ValueError('Legacy run has no saved Evaluation protocol; supply --protocol YAML explicitly')
    protocol = validate_protocol(settings)
    output = Path(provenance['output_directory'])
    trajectory = output / '00000_KeyFrameTrajectory.csv'
    gt = Path(gt_override).expanduser().resolve() if gt_override else source_directory(provenance) / 'groundtruth.csv'
    calibration = output / 'calibration_exp.yaml'
    execution_path = run / 'execution.json'
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    destination = evaluate_files(trajectory, gt, calibration, run / 'evaluations', protocol,
                                 provenance, execution=execution)
    summary = json.loads((destination / 'summary.json').read_text())
    update_evaluation_log(output.parent.parent / 'vslamlab_exp_log.csv',
                          provenance.get('dataset_name'), provenance.get('sequence_name'),
                          0, destination, summary)
    return destination
