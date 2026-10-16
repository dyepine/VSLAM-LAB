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
from matplotlib.lines import Line2D
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


MAX_RUNS_PER_PAGE = 4


def run_label(row):
    name = re.sub(r'^comparison_[^_]+_[^_]+_', '', row['experiment'])
    name = re.sub(r'_\d+$', '', name)
    name = {'orbslam3_default': 'ORB-SLAM3', 'orbslam3_no_loop': 'ORB-SLAM3 без LC'}.get(name, name)
    return f"{name} / r{int(row['iteration']) + 1}"


def page(fig, title, subtitle, handles):
    outer = fig.add_gridspec(2, 1, height_ratios=[.22, 1])
    header = outer[0].subgridspec(3, 1)
    axes = [fig.add_subplot(header[i]) for i in range(3)]
    for ax in axes: ax.axis('off')
    axes[0].text(.5, .5, title, ha='center', va='center', fontsize=16, weight='bold')
    axes[1].text(.5, .5, subtitle, ha='center', va='center', fontsize=10, color='.35')
    if handles: axes[2].legend(handles=handles, loc='center', ncol=min(3, len(handles)), frameon=False, fontsize=10)
    return outer[1]


def save_figure(fig, path):
    # Freeze the computed layout before switching between PNG and PDF renderers.
    fig.canvas.draw()
    fig.set_layout_engine('none')
    fig.savefig(path.with_suffix('.png'), dpi=150)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def metrics_table(ax, rows):
    ax.axis('off')
    cells = []
    for row in rows:
        values = [run_label(row)]
        for metric, _ in METRICS:
            value = row.get(metric + '_rmse', '')
            values.append(f'{float(value):.3g}' if value else '—')
        coverage = row.get('matched_fraction_of_saved_poses', '')
        values.extend([f'{float(coverage)*100:.1f}%' if coverage else '—',
                       f"{row.get('unique_matched_estimate_poses', '—')} / {row.get('estimate_poses', '—')}",
                       'OK' if row['status']=='evaluated' else row['status']])
        cells.append(values)
    table = ax.table(cellText=cells, colLabels=['Метод / повтор', 'APE, м', 'RPE, м', 'APE, °', 'RPE, °', 'Покрытие', 'Позы', 'Статус'],
                     colWidths=[.25,.1,.1,.1,.1,.1,.1,.15], cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white'); cell.set_facecolor('#e8eef3' if r==0 else '#f4f6f8')
        if r==0: cell.set_text_props(weight='bold')
    ax.text(0, -.04, 'RMSE; покрытие и позы: сопоставленные / сохранённые, не доля входных кадров.',
            fontsize=9, color='.35', transform=ax.transAxes)


def plot_comparison(table):
    table = Path(table)
    with table.open() as stream: records = list(csv.DictReader(stream))
    output = table.parent / 'figures'
    output.mkdir(exist_ok=True)
    groups = {}
    for row in records: groups.setdefault((row['dataset'], row['sequence']), []).append(row)
    manifest = []
    for (dataset, sequence), rows in groups.items():
        tag = re.sub(r'[^A-Za-z0-9_-]', '_', f'{dataset}_{sequence}')
        cached = {}
        palette = plt.get_cmap('tab20').colors
        palette = palette[::2] + palette[1::2]
        colors = {i:palette[i % len(palette)] for i in range(len(rows))}
        for i, row in enumerate(rows):
            if row['status']!='evaluated': continue
            directory = Path(row['evaluation_directory'])
            summary = json.loads((directory/'summary.json').read_text())
            metrics = {name:file_interface.load_res_file(directory/(name+'.zip')) for name, _ in METRICS
                       if (directory/(name+'.zip')).is_file()}
            estimate, reference = full_paths(summary)
            cached[i] = (summary, metrics, estimate, reference)
        if len({s['groundtruth_sha256'] for s, *_ in cached.values()}) > 1:
            raise ValueError('Cannot draw a shared GT for different reference trajectories')
        full_gt = None
        if cached:
            first = next(iter(cached.values()))[0]
            gt_ns, gt_values = read_pose_csv(Path(first['groundtruth']))
            full_gt = to_evo(gt_ns, gt_values, first['time_origin_ns'])
        origin = min((s['time_origin_ns'] for s, *_ in cached.values()), default=0)
        # Common limits across pages make visual comparisons between pages meaningful.
        limits = {}
        for metric, _ in METRICS:
            values = [r[metric].np_arrays['error_array'] for _, r, *_ in cached.values() if metric in r]
            times = [r[metric].np_arrays['timestamps'] + (s['time_origin_ns']-origin)/1e9
                     for s, r, *_ in cached.values() if metric in r]
            if values:
                limits[metric] = (max(float(np.max(v)) for v in values),
                                  min(float(np.min(t)) for t in times), max(float(np.max(t)) for t in times))
        route_positions = [e.positions_xyz[:,:2] for _, _, e, _ in cached.values()]
        if full_gt is not None: route_positions.append(full_gt.positions_xyz[:,:2])
        bounds = None
        if route_positions:
            points = np.concatenate(route_positions); low=points.min(axis=0); high=points.max(axis=0)
            margin=np.maximum((high-low)*.05, .1); bounds=(low-margin, high+margin)
        files = []
        total = (len(rows)+MAX_RUNS_PER_PAGE-1)//MAX_RUNS_PER_PAGE
        for number, start in enumerate(range(0, len(rows), MAX_RUNS_PER_PAGE), 1):
            indices = list(range(start, min(start+MAX_RUNS_PER_PAGE, len(rows))))
            selected = [rows[i] for i in indices]
            suffix = '' if total==1 else f'_p{number:02d}'
            handles = [Line2D([0],[0],color=colors[i], lw=1.5, label=run_label(rows[i])) for i in indices]
            summaries = [cached[i][0] for i in indices if i in cached]
            protocols = sorted({f"{s['protocol']['align']}; scale={s['protocol']['correct_scale']}; dt={s['protocol']['max_time_difference_s']:g}s; RPE Δ={s['protocol']['rpe_delta_m']:g}m" for s in summaries})
            subtitle = ' | '.join(protocols) or 'Нет успешно оценённых запусков'
            title = f'{dataset} · {sequence} — страница {number}/{total}'
            fig = plt.figure(figsize=(14,9), layout='constrained')
            body = page(fig, title+' · Маршрут', 'Полный GT и сохранённые позы; преобразование из оценки evo',
                        [Line2D([0],[0],color='.75',label='Полный GT'), Line2D([0],[0],color='.25',ls='--',label='GT участка'), *handles])
            ax = fig.add_subplot(body)
            if full_gt is not None:
                first = next(iter(cached.values()))[0]
                for part in segments(full_gt.timestamps, first):
                    ax.plot(full_gt.positions_xyz[part,0], full_gt.positions_xyz[part,1],color='.8',lw=1.2)
                times=[]; positions=[]
                for i in indices:
                    if i not in cached: continue
                    s, _, _, reference=cached[i]
                    times.append(np.rint(reference.timestamps*1e9).astype(np.int64)+s['time_origin_ns']-origin)
                    positions.append(reference.positions_xyz)
                if times:
                    time, ids=np.unique(np.concatenate(times),return_index=True)
                    points=np.concatenate(positions)[ids]
                    for part in segments(time/1e9, dict(summaries[0],time_origin_ns=origin)):
                        ax.plot(points[part,0],points[part,1],color='.25',ls='--',lw=1.4,zorder=3)
                for i in indices:
                    if i not in cached: continue
                    s, _, estimate, _=cached[i]
                    for part in segments(estimate.timestamps,s):
                        ax.plot(estimate.positions_xyz[part,0],estimate.positions_xyz[part,1],color=colors[i],lw=1.5)
                    ax.scatter(*estimate.positions_xyz[0,:2],color=colors[i],marker='o',s=30,zorder=4)
                    ax.scatter(*estimate.positions_xyz[-1,:2],color=colors[i],marker='X',s=45,zorder=4)
            else: ax.text(.5,.5,'Нет траекторий для отображения',ha='center',transform=ax.transAxes)
            if bounds: ax.set_xlim(bounds[0][0],bounds[1][0]); ax.set_ylim(bounds[0][1],bounds[1][1])
            ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('X, м'); ax.set_ylabel('Y, м'); ax.grid(alpha=.2)
            ax.set_title('● начало   × конец сохранённого участка',fontsize=10)
            path=output/(tag+'_trajectories'+suffix); save_figure(fig,path); files.append(path.name+'.png')

            fig = plt.figure(figsize=(15,10), layout='constrained')
            body = page(fig,title+' · Ошибки и метрики',subtitle,handles)
            grid=body.subgridspec(3,2,height_ratios=[1,1,.75])
            for index, (metric, metric_title) in enumerate(METRICS):
                ax=fig.add_subplot(grid[index//2,index%2]); count=0
                for i in indices:
                    if i not in cached: continue
                    s, results, _, _=cached[i]
                    if metric not in results: continue
                    result=results[metric]; t=result.np_arrays['timestamps']; offset=(s['time_origin_ns']-origin)/1e9
                    if metric.startswith('rpe_'): ax.scatter(t+offset,result.np_arrays['error_array'],color=colors[i],s=12)
                    else:
                        for part in segments(t,s): ax.plot(t[part]+offset,result.np_arrays['error_array'][part],color=colors[i],marker='.',ms=4,lw=1)
                    count+=1
                if metric in limits:
                    maximum, begin, end=limits[metric]; ax.set_ylim(0,max(maximum*1.1,1e-6))
                    ax.set_xlim(begin-.2,end+.2)
                if not count: ax.text(.5,.5,'Метрика недоступна',ha='center',transform=ax.transAxes)
                ax.set_title(metric_title,loc='left'); ax.set_xlabel('Время от общего начала GT, с'); ax.grid(alpha=.2)
            metrics_table(fig.add_subplot(grid[2,:]),selected)
            path=output/(tag+'_errors'+suffix); save_figure(fig,path); files.append(path.name+'.png')
        # A complete table is retained separately even when graphical pages are split.
        summary_csv=output/(tag+'_summary.csv')
        with summary_csv.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader(); writer.writerows(rows)
        old_manifest=output/'manifest.json'
        if old_manifest.is_file():
            previous=json.loads(old_manifest.read_text())
            for item in previous:
                if item.get('dataset')==dataset and item.get('sequence')==sequence:
                    for obsolete in set(item.get('plots',[]))-set(files):
                        path=output/obsolete
                        if path.parent==output and path.suffix=='.png':
                            path.unlink(missing_ok=True); path.with_suffix('.pdf').unlink(missing_ok=True)
        manifest.append(dict(dataset=dataset,sequence=sequence,plots=files,summary_table=summary_csv.name,
                             runs_per_page=MAX_RUNS_PER_PAGE,
                             evaluations=[row['evaluation_directory'] for row in rows if row['status']=='evaluated']))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return output
