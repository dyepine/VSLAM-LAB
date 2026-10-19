"""Original OKVIS 1 stereo-inertial baseline, built with the host toolchain."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

from Baselines.BaselineVSLAMLAB import BaselineVSLAMLAB

ADAPTER = Path(__file__).resolve().parents[1] / 'extra-files/okvis1'


class OKVIS1_baseline(BaselineVSLAMLAB):
    def __init__(self):
        super().__init__('okvis1', 'OKVIS1',
                         {'mode': 'stereo-vi', 'verbose': 0, 'imu_overrides': {}})
        self.settings_yaml = ADAPTER / 'vslamlab_okvis1_settings.yaml'
        self.modes = ['stereo-vi']
        self.cam_models = ['pinhole', 'radtan4', 'radtan5', 'radtan8', 'equid4']
        self.color = (0.85, 0.5, 0.15)

    def is_installed(self):
        installed = (self.baseline_path / 'build/vslamlab_okvis1_stereo_vi').is_file()
        return installed, 'is installed' if installed else 'requires local CMake build (Ubuntu 22.04)'

    def ensure_installed(self):
        if not self.is_installed()[0]:
            subprocess.run(['/bin/bash', str(ADAPTER / 'build.sh')], check=True)

    def download_vslamlab_settings(self):
        return self.settings_yaml.is_file()

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        parameters = self.resolve_parameters(exp)
        if parameters['mode'] not in self.modes:
            raise ValueError('OKVIS 1 supports stereo-vi only')
        folder = exp.folder / dataset.dataset_folder / sequence_name
        arguments = {'sequence_path': dataset.sequence_path(sequence_name),
                     'calibration_yaml': folder / 'calibration_exp.yaml',
                     'rgb_csv': folder / 'rgb_exp.csv', 'exp_folder': folder,
                     'exp_it': exp_it, 'settings_yaml': self.settings_yaml, **parameters}
        arguments['imu_overrides'] = json.dumps(arguments['imu_overrides'])
        tokens = [sys.executable, str(ADAPTER / 'run.py')]
        for key, value in arguments.items():
            tokens.extend(['--' + key, str(value)])
        return shlex.join(tokens)
