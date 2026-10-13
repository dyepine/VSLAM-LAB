"""Protocol evaluation integrated with native experiment rows and repetitions."""
import csv
import json
import yaml
from pathlib import Path
from Evaluate.protocol import evaluate_files, update_evaluation_log, validate_protocol
from utilities import read_csv


def evaluate_sequence(exp, dataset, sequence):
    protocol = validate_protocol(exp.evaluation)
    output = exp.folder / dataset.dataset_folder / str(sequence)
    log = read_csv(exp.log_csv) if exp.log_csv.is_file() else None
    rows = {} if log is None else {int(row['exp_it']): row for _, row in log.iterrows()
        if row['dataset_name'] == dataset.dataset_name and str(row['sequence_name']) == str(sequence)}
    records = []
    for iteration in range(exp.num_runs):
        row = rows.get(iteration)
        record = dict(experiment=exp.name, module=exp.module, dataset=dataset.dataset_name,
                      sequence=str(sequence), iteration=iteration, status='run_failed', error='',
                      evaluation_directory='', estimate_frame=protocol['estimate_frame'],
                      align=protocol['align'], correct_scale=protocol['correct_scale'],
                      max_time_difference_s=protocol['max_time_difference_s'])
        if row is None:
            record['error'] = 'Native execution log row missing'
            records.append(record)
            continue
        if str(row['STATUS']) != 'completed' or str(row['SUCCESS']).lower() != 'true':
            record['error'] = 'Native run did not complete successfully'
            records.append(record)
            continue
        try:
            metadata = dict(module=exp.module, dataset_name=dataset.dataset_name,
                            sequence_name=str(sequence), output_directory=str(output),
                            input_directory=str(dataset.sequence_path(sequence)), iteration=iteration,
                            native_experiment=exp.name, evaluation_protocol=protocol)
            rgb_csv = output / 'rgb_exp.csv'
            if rgb_csv.is_file():
                with rgb_csv.open() as stream:
                    metadata['input_audit'] = {'images_selected': sum(1 for _ in stream) - 1}
            execution = dict(success=True, confirmation='native STATUS/SUCCESS log',
                             time_s=float(row['TIME']))
            destination = evaluate_files(
                output / f'{iteration:05d}_KeyFrameTrajectory.csv', output / 'groundtruth.csv',
                output / 'calibration_exp.yaml', output / 'evaluations' / f'{iteration:05d}',
                protocol, metadata, execution=execution,
                native_log=output / f'system_output_{iteration:05d}.txt')
            summary = json.loads((destination / 'summary.json').read_text())
            update_evaluation_log(exp.log_csv, dataset.dataset_name, str(sequence), iteration,
                                  destination, summary)
            record.update(status='evaluated', evaluation_directory=str(destination),
                          matched_fraction_of_saved_poses=summary['matched_fraction_of_saved_poses'],
                          estimate_poses=summary['estimate_poses'], matched_pairs=summary['matched_pairs'],
                          unique_matched_estimate_poses=summary['unique_matched_estimate_poses'],
                          max_matched_time_difference_s=summary['max_matched_time_difference_s'])
            for metric, stats in summary['metrics'].items():
                record[metric + '_rmse'] = stats['rmse']
                record[metric + '_samples'] = summary['metric_sample_counts'][metric]
            record['unavailable_metrics'] = json.dumps(summary['unavailable_metrics'])
        except Exception as error:
            # Preserve failures as comparison rows and continue through all repetitions.
            record.update(status='evaluation_failed', error=f'{type(error).__name__}: {error}')
            update_evaluation_log(exp.log_csv, dataset.dataset_name, str(sequence), iteration,
                                  None, None, error=record['error'])
        records.append(record)
    return records


def write_comparison(records, destination):
    """One row per run, including failed runs; no averaging across different sequences."""
    headers = list(dict.fromkeys(key for row in records for key in row))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
    return destination


def compare_saved(experiments, destination):
    """Compare the latest published evaluation for every native run, preserving failures."""
    records = []
    for exp in experiments.values():
        log = read_csv(exp.log_csv) if exp.log_csv.is_file() else None
        logged = {} if log is None else {(r['dataset_name'], str(r['sequence_name']), int(r['exp_it'])): r
                                        for _, r in log.iterrows()}
        selection = yaml.safe_load(exp.config_yaml.read_text())
        expected = [(dataset, str(sequence), iteration) for dataset, sequences in selection.items()
                    for sequence in sequences for iteration in range(exp.num_runs)]
        for dataset, sequence, iteration in expected:
            row = logged.get((dataset, sequence, iteration),
                             dict(dataset_name=dataset, sequence_name=sequence, exp_it=iteration,
                                  STATUS='missing', SUCCESS=False))
            record = dict(experiment=exp.name, module=exp.module, dataset=row['dataset_name'],
                          sequence=str(row['sequence_name']), iteration=int(row['exp_it']),
                          status='not_evaluated', error='', evaluation_directory='')
            if str(row['STATUS']) != 'completed' or str(row['SUCCESS']).lower() != 'true':
                record.update(status='run_failed', error='Native run did not complete successfully')
            elif str(row['EVALUATION']) == 'failed':
                record.update(status='evaluation_failed', error=str(row.get('evaluation_error', '')))
            elif str(row['EVALUATION']) == 'evo':
                directory = Path(str(row['evaluation_directory']))
                summary = json.loads((directory / 'summary.json').read_text())
                protocol = summary['protocol']
                record.update(status='evaluated', evaluation_directory=str(directory),
                              align=protocol['align'], correct_scale=protocol['correct_scale'],
                              estimate_frame=protocol['estimate_frame'],
                              max_time_difference_s=protocol['max_time_difference_s'],
                              matched_fraction_of_saved_poses=summary['matched_fraction_of_saved_poses'],
                              matched_pairs=summary['matched_pairs'], estimate_poses=summary['estimate_poses'])
                for metric, stats in summary['metrics'].items():
                    record[metric + '_rmse'] = stats['rmse']
                    record[metric + '_samples'] = summary['metric_sample_counts'][metric]
                record['unavailable_metrics'] = json.dumps(summary['unavailable_metrics'])
            records.append(record)
    return write_comparison(records, destination)
