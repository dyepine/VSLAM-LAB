# Explicit evaluation protocols

Native experiments can supply `Evaluation` to use evo-backed protocol evaluation
through `evaluate_exp` / `evaluate_sequence`. Without this field the existing
benchmark evaluator remains unchanged.

```yaml
example:
  Config: /absolute/path/to/sequences.yaml
  Module: orbslam3
  NumRuns: 2
  Parameters: {mode: stereo-vi, verbose: 0}
  Evaluation:
    align: SE3
    correct_scale: false
    estimate_frame: camera
    camera: rgb_0
    max_time_difference_s: 0.01
    evaluate_orientation: true
    max_pose_gap_s: 1.0
    rpe_delta_m: 1.0
    rpe_relative_tolerance: 0.1
```

`estimate_frame` describes the trajectory export, not the sensor inputs to the
method. A camera trajectory is right-multiplied by inverse calibration `T_BS`
exactly once to compare body poses. A body trajectory is used directly. This is
separate from world-frame SE(3) alignment, which does not correct metric scale.
The reference is expected to contain poses of the calibration's body frame.

All repetitions are processed. Each evaluation publishes an immutable directory
under the sequence output's `evaluations/<iteration>/<timestamp>`, including evo
ZIPs, protocol, metrics and association coverage. Native logs retain the path to
the latest evaluation. Log updates use a per-log lock and atomic replacement.
`comparison.csv` next to the native experiment YAML contains every run, including
execution and evaluation failures. `evaluate_exp` finishes the batch before
raising if any runs failed; `compare_exp` builds a table from saved evaluations.
No scores are averaged across different sequences. Protocol overrides merge
with each experiment's saved protocol, so an association-tolerance override need
not change its pose frame.

Association uses standard evo semantics, with nanoseconds converted to relative
seconds. Reported fractions refer to unique matched saved estimate poses;
matched-pair counts may differ because evo permits reuse. RPE uses distance
pairs within continuous matched segments. No interpolation or automatic
relaxation of the time tolerance is performed.

`protocol.evaluate_run` retains support for previously saved research artifacts;
it delegates to the same `evaluate_files` implementation. Plotting can consume
saved evo ZIPs without recalculating metrics.

Protocol comparisons also publish PNG/PDF figures under `figures/` next to the
experiment YAML. Trajectory projections use every saved observed estimate pose,
transformed by the alignment already stored in the evaluation, and measured GT
within its time span. Input data/calibration hashes are verified; plots never
refit alignment or interpolate observations. Error curves read cached evo ZIPs;
RPE all-pairs samples are scatter points. 2D axes use equal metric scale, while
3D explicitly labels independent axis scales. Summary figures include failures
and association coverage. `manifest.json` records the source evaluations.
