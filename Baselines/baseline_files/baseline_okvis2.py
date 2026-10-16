import os.path
from pathlib import Path
import math
import yaml


from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

SCRIPT_LABEL = f"\033[95m[{Path(__file__).name}]\033[0m "


class OKVIS2_baseline(BaselineVSLAMLAB):
    """OKVIS2 helper for VSLAM-LAB Baselines."""

    def __init__(self, baseline_name: str = 'okvis2', baseline_folder: str = 'OKVIS2') -> None:

        default_parameters = {'verbose': 1, 'mode': 'mono-vi', 'imu_overrides': {},
                              'vocabulary': str(VSLAMLAB_BASELINES / baseline_folder / 'resources' / 'small_voc.yml.gz')}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = (0.470, 0.862, 0.628) # 'green'
        self.modes = ['mono-vi']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5', 'radtan8', 'equid4']
        self.command_style = 'cpp'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        # Bias priors and saturation limits are estimator inputs, distinct from
        # calibrated noise densities. Only alter the per-experiment copy using
        # explicitly requested values, never the dataset calibration.
        overrides = exp.parameters.get('imu_overrides', {})
        allowed = {'sigma_bg', 'sigma_ba', 'a_max', 'g_max', 'a0', 'g0'}
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError('OKVIS2 imu_overrides contains unsupported fields')
        path = exp.folder / dataset.dataset_folder / sequence_name / 'calibration_exp.yaml'
        calibration = yaml.safe_load(path.read_text())
        settings = self.settings_yaml.read_text()
        if settings.startswith('%YAML:'):
            settings = settings.split('\n', 1)[1]
        imu_name = yaml.safe_load(settings)['imu']
        imu = next(entry for entry in calibration['imus'] if entry['imu_name'] == imu_name)
        imu.update(overrides)
        for name in ['sigma_bg', 'sigma_ba', 'a_max', 'g_max']:
            value = imu.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'OKVIS2 requires positive {name}; specify it in Parameters.imu_overrides')
        for name in ['a0', 'g0']:
            value = imu.get(name)
            if not isinstance(value, list) or len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(f'OKVIS2 requires a finite 3-vector {name}')
        if overrides:
            path.write_text('%YAML 1.2\n---\n' + yaml.safe_dump(
                calibration, sort_keys=False, default_flow_style=True))
        return super().build_execute_command(exp_it, exp, dataset, sequence_name)


class OKVIS2_baseline_dev(OKVIS2_baseline):
    """OKVIS2-DEV helper for VSLAM-LAB Baselines."""

    def __init__(self) -> None:
        super().__init__(baseline_name = 'okvis2-dev', baseline_folder = 'OKVIS2-DEV')
        self.color = tuple(max(c / 2.0, 0.0) for c in self.color)

    def is_installed(self) -> tuple[bool, str]:
        is_installed = os.path.isfile(os.path.join(self.baseline_path, 'bin', 'vslamlab_okvis2_mono_vi'))
        return (True, 'is installed') if is_installed else (False, 'not installed (auto install available)') 
    
