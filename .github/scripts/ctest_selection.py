"""Run only CTest cases owned by selected source modules."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from engine_ci import OWNERS


def test_owner(test, graph, owners, build_dir=None, source_dir=None):
    def owner_of(source):
        source = Path(source).resolve()
        for name, directory in sorted(owners.items(), key=lambda item: -len(str(item[1]))):
            if source.is_relative_to(directory):
                return name
        return None

    index = test.get('backtrace')
    while index is not None:
        node = graph['nodes'][index]
        if 'file' in node:
            owner = owner_of(graph['files'][node['file']])
            if owner:
                return owner
        index = node.get('parent')
    # GoogleTest-discovered cases do not carry a CMake registration backtrace.
    properties = {item['name']: item['value'] for item in test.get('properties', [])}
    if properties.get('DEF_SOURCE_LINE'):
        return owner_of(re.sub(r':\d+$', '', properties['DEF_SOURCE_LINE']))
    if build_dir and properties.get('WORKING_DIRECTORY'):
        working = Path(properties['WORKING_DIRECTORY']).resolve()
        if working.is_relative_to(build_dir):
            return owner_of(source_dir / working.relative_to(build_dir))
    return None


def select_indices(metadata, owners, modules, checks, build_dir=None, source_dir=None):
    selected = []
    for index, test in enumerate(metadata['tests'], 1):
        if test['name'] in checks or test_owner(test, metadata['backtraceGraph'], owners,
                                               build_dir, source_dir) in modules:
            selected.append(index)
    return selected


def select_tests(metadata, owners, modules, checks, build_dir=None, source_dir=None):
    return [metadata['tests'][index - 1]['name'] for index in
            select_indices(metadata, owners, modules, checks, build_dir, source_dir)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--owner', action='append', default=[])
    parser.add_argument('--config', default='Release')
    args = parser.parse_args()
    scope = json.loads(args.selection)
    owners = dict(value.split('=', 1) for value in args.owner) if args.owner else OWNERS
    owners = {name: Path(path).resolve() for name, path in owners.items()}
    metadata = json.loads(subprocess.check_output(
        ['ctest', '--test-dir', str(args.build_dir), '-C', args.config, '--show-only=json-v1']))
    indices = select_indices(metadata, owners, scope['modules'], scope['checks'],
                              args.build_dir.resolve(), Path.cwd())
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
            output.write(f'has_tests={str(bool(indices)).lower()}\n')
    if not indices:
        print('The selected modules do not register tests for this configuration.')
        return
    # A file avoids Windows command-line limits and distinguishes same-named
    # tests in different modules. The empty range adds no implicit tests.
    test_file = args.build_dir.resolve() / 'ctest-selection.txt'
    test_file.write_text('0,0,1,' + ','.join(map(str, indices)) + '\n', encoding='utf-8')
    print(f'Running {len(indices)} selected CTest cases: {scope}', flush=True)
    report = args.build_dir.resolve() / 'ctest-results.xml'
    subprocess.run(['ctest', '--test-dir', str(args.build_dir), '-C', args.config,
                    '-I', str(test_file), '--output-on-failure', '--no-tests=error',
                    '--output-junit', str(report)], check=True)


if __name__ == '__main__':
    main()
