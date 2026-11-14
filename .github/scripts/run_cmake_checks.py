"""Run explicitly selected script-mode CMake checks without configuring the engine."""

import argparse
import json
from pathlib import Path
import subprocess


def registrations(source):
    return json.loads((source / '.github/scripts/cmake_checks.json').read_text())


def selected_checks(source, checks):
    return sorted(set(checks) & registrations(source).keys())


def run_checks(source, checks, build=None):
    registry = registrations(source)
    unknown = set(checks) - registry.keys()
    if unknown:
        raise ValueError('Unknown script-mode CMake checks: ' + ', '.join(sorted(unknown)))
    build = build or source / 'build/ci-checks'
    for name in sorted(set(checks)):
        entry = registry[name]
        command = ['cmake', f'-DTEST_SOURCE_DIR:PATH={source}',
                   f'-DTEST_BINARY_DIR:PATH={build}', '-DTEST_GENERATOR=Ninja']
        command += [f'-D{key}:PATH={source / value}'
                    for key, value in entry.get('variables', {}).items()]
        command += ['-P', str(source / entry['script'])]
        print(f'Running {name}', flush=True)
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path.cwd())
    parser.add_argument('--checks', required=True)
    args = parser.parse_args()
    checks = json.loads(args.checks)
    if not isinstance(checks, list) or any(not isinstance(name, str) for name in checks):
        parser.error('checks must be a JSON list of check names')
    run_checks(args.source_dir.resolve(), checks)


if __name__ == '__main__':
    main()
