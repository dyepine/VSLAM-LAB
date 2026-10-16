"""Create dated run directories without overwriting earlier or concurrent runs."""
from datetime import datetime
from pathlib import Path
import re
import shlex


def create_run_directory(results_root: Path, now: datetime | None = None) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now().astimezone()).strftime('%Y-%m-%d_%H-%M-%S')
    number = 1
    while True:
        suffix = '' if number == 1 else f'-{number:02d}'
        folder = results_root / (timestamp + suffix)
        try:
            folder.mkdir()
        except FileExistsError:
            number += 1
        else:
            return folder


def select_run(explicit, results_root, *, required_files=()):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'Run directory does not exist: {path}')
        return path
    pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d+)?$')
    runs = [path for path in results_root.iterdir()
            if path.is_dir() and pattern.match(path.name) and
            all((path / name).is_file() for name in required_files)]
    if not runs:
        if required_files:
            raise FileNotFoundError('No completed evaluation; run make evaluate first')
        raise FileNotFoundError('No dated runs; specify RUN_DIR=/path/to/run')
    # Numeric collision suffixes must sort numerically (-100 follows -99).
    return max(runs, key=lambda p: (p.name[:19], int(p.name[20:]) if len(p.name) > 19 else 1))


def source_directory(provenance):
    if provenance.get('input_directory'):
        return Path(provenance['input_directory'])
    # Support artifacts written by the initial launcher.
    for token in shlex.split(provenance.get('command', '')):
        if token.startswith('sequence_path:'):
            return Path(token.split(':', 1)[1])
    raise ValueError('Cannot locate source recording from run_provenance.json; specify --gt')
