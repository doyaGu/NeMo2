"""Run only CTest cases owned by selected source modules."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from engine_ci import OWNERS, module_selected, test_owners


def test_owner(test, graph, owners, build_dir=None, source_dir=None):
    properties = {item['name']: item['value'] for item in test.get('properties', [])}
    for label in properties.get('LABELS', []):
        if label.startswith('plugin:'):
            return label
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
    if properties.get('DEF_SOURCE_LINE'):
        return owner_of(re.sub(r':\d+$', '', properties['DEF_SOURCE_LINE']))
    if build_dir and properties.get('WORKING_DIRECTORY'):
        working = Path(properties['WORKING_DIRECTORY']).resolve()
        if working.is_relative_to(build_dir):
            return owner_of(source_dir / working.relative_to(build_dir))
    return None


def select_indices(metadata, owners, modules, checks, build_dir=None, source_dir=None, test_targets=None):
    selected = []
    for index, test in enumerate(metadata['tests'], 1):
        owner = test_owner(test, metadata['backtraceGraph'], owners, build_dir, source_dir)
        targets = (test_targets or {}).get(owner)
        if targets and not any(Path(argument).stem in targets for argument in test.get('command', [])):
            continue
        if test['name'] in checks or module_selected(owner, modules):
            selected.append(index)
    return selected


def select_tests(metadata, owners, modules, checks, build_dir=None, source_dir=None):
    return [metadata['tests'][index - 1]['name'] for index in
            select_indices(metadata, owners, modules, checks, build_dir, source_dir)]


def validate_selection(metadata, owners, scope, build_dir=None, source_dir=None, platform=None):
    """Require coverage for every requested owner, named check and target."""
    platform = platform or sys.platform
    policy = json.loads(Path(__file__).with_name('testless_modules.json').read_text())
    cache = {}
    if build_dir and (build_dir / 'CMakeCache.txt').exists():
        for line in (build_dir / 'CMakeCache.txt').read_text().splitlines():
            match = re.fullmatch(r'([^:#/][^:]*):[^=]+=(.*)', line)
            if match:
                cache[match[1]] = match[2]
    missing_checks = set(scope['checks']) - {test['name'] for test in metadata['tests']}
    if missing_checks:
        raise ValueError('Requested CTest checks are not registered: ' + ', '.join(sorted(missing_checks)))
    skipped = {}
    for module in scope['modules']:
        indices = select_indices(metadata, owners, [module], [], build_dir, source_dir,
                                 scope.get('test_targets'))
        if not indices:
            exception = policy.get(module, {})
            cache_matches = all(cache.get(key) == value for key, value in exception.get('cache', {}).items())
            if not scope.get('test_targets', {}).get(module) and cache_matches and exception.get('reason') and (
                    '*' in exception.get('platforms', []) or platform in exception.get('platforms', [])):
                skipped[module] = exception['reason']
                continue
            raise ValueError(f'No registered tests match requested module {module} on {platform}')
        for target in scope.get('test_targets', {}).get(module, []):
            if not select_indices(metadata, owners, [module], [], build_dir, source_dir,
                                  {module: [target]}):
                raise ValueError(f'Requested test target is not registered: {module}/{target}')
    return skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--owner', action='append', default=[])
    parser.add_argument('--config', default='Release')
    args = parser.parse_args()
    scope = json.loads(args.selection)
    owners = dict(value.split('=', 1) for value in args.owner) if args.owner else test_owners(scope['modules'])
    owners = {name: Path(path).resolve() for name, path in owners.items()}
    metadata = json.loads(subprocess.check_output(
        ['ctest', '--test-dir', str(args.build_dir), '-C', args.config, '--show-only=json-v1']))
    try:
        skipped = validate_selection(metadata, owners, scope, args.build_dir.resolve(), Path.cwd())
    except ValueError as error:
        parser.error(str(error))
    for module, reason in skipped.items():
        print(f'Skipping {module} on {sys.platform}: {reason}', flush=True)
    indices = select_indices(metadata, owners, scope['modules'], scope['checks'],
                              args.build_dir.resolve(), Path.cwd(), scope.get('test_targets'))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
            output.write(f'has_tests={str(bool(indices)).lower()}\n')
            require_gtest = any(module not in skipped and (
                module in {'vxmath', 'ck2', 'renderengine', 'plugins', 'player'} or module.startswith('plugin:'))
                for module in scope['modules'])
            output.write(f'require_gtest={str(require_gtest).lower()}\n')
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
