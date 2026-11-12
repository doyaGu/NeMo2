"""Select engine test modules from actual source changes."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from change_detection import baseline, changed_paths, documentation, emit


OWNERS = {
    'vxmath': 'Source/VxMath', 'ck2': 'Source/CK2',
    'renderengine': 'Source/RenderEngine', 'plugins': 'Source/Plugins',
    'buildingblocks': 'Source/BuildingBlocks',
    'ivp': 'Source/BuildingBlocks/physics_RT/ivp',
}
CHECKS = ['NumericalIsolation', 'NumericalIsolationRules', 'NeMo2Subproject',
          'NeMo2StageInstall', 'CMakeRegeneration']
TEST_OPTIONS = {'vxmath': 'VXMATH', 'ck2': 'CK2', 'renderengine': 'CKRE',
                'plugins': 'CKPLUGINS', 'buildingblocks': 'CKBB', 'ivp': 'IVP'}
GROUPS = {'bb': 'buildingblocks', 'plugin': 'plugins'}


def component_owner(path):
    owner = next((name for name, prefix in sorted(OWNERS.items(), key=lambda item: -len(item[1]))
                  if path == prefix or path.startswith(prefix + '/')), None)
    for prefix, group in GROUPS.items():
        if owner == group:
            relative = path.removeprefix(OWNERS[group] + '/')
            directory, separator, _ = relative.partition('/')
            if separator and directory not in {'tests', 'test', 'cmake', 'include', 'src', 'scripts', 'Common'} and re.fullmatch(r'[A-Za-z0-9_-]+', directory):
                return prefix + ':' + directory
    return owner


def test_owners(modules):
    owners = dict(OWNERS)
    for name in modules:
        if ':' in name:
            prefix, component = name.split(':', 1)
            owners[name] = OWNERS[GROUPS[prefix]] + '/' + component
    return owners


def module_selected(owner, modules):
    return owner in modules or (owner and ':' in owner and GROUPS[owner.split(':', 1)[0]] in modules)


def job_matrix(target='all', resources_only=False):
    platforms = json.loads(Path(__file__).with_name('ci_platforms.json').read_text())
    rows = []
    for platform in platforms:
        for name in ('runtime', 'static'):
            if target != 'all' and target != f'sdl-{platform["name"]}-{name}':
                continue
            if resources_only and (not platform['name'].startswith('windows-') or name != 'runtime'):
                continue
            rows.append({'platform': platform, 'linkage': {'name': name, 'build_static': 'ON' if name == 'static' else 'OFF'}})
    if not rows and target != 'sdl-linux-x64-full-tests':
        raise ValueError('unknown build/test target')
    return {'include': rows}


def direct_test_target(path, owner):
    # Only restrict verified one-source/one-executable core tests. Shared test
    # helpers, generated variants and module-specific harnesses keep their suite.
    if owner not in {'vxmath', 'ck2'} or '/tests/' not in path or not path.endswith('Test.cpp'):
        return None
    source = Path(__file__).resolve().parents[2]
    folder = source / OWNERS[owner] / 'tests'
    files = list(folder.rglob('CMakeLists.txt')) if folder.exists() else []
    text = '\n'.join(file.read_text() for file in files)
    stem = Path(path).stem
    if any(re.search(r'\b' + re.escape(stem) + r'\b', loop) for loop in re.findall(r'foreach\s*\(([^)]*)\)', text, re.S)):
        return None
    if text.count(stem + '.cpp') != 1:
        return None
    if re.search(r'(?:add_ck2_test|vx_add_test)\(\s*' + re.escape(stem) + r'\b', text):
        return stem
    if owner == 'vxmath':
        sources = re.search(r'set\(VXMATH_TEST_SOURCES\s+(.*?)\n\)', text, re.S)
        if sources and re.search(r'^\s*' + re.escape(stem) + r'\.cpp\s*$', sources[1], re.M):
            return stem
    return None


def classify(paths):
    modules, checks, scripts = set(), set(), set()
    test_targets, whole_modules = {}, set()
    build = layout = False
    resources_only = True
    for path in paths:
        if documentation(path):
            continue
        if path.startswith('Source/') and path.endswith('/Version.rc.in'):
            # Version resources affect Windows binaries, not engine behavior.
            build = True
            continue
        owner = component_owner(path)
        if owner:
            modules.add(owner)
            target = direct_test_target(path, owner)
            if target:
                test_targets.setdefault(owner, set()).add(target)
            else:
                whole_modules.add(owner)
            if owner == 'ivp':
                modules.add('bb:physics_RT')
                whole_modules.add('bb:physics_RT')
            build = True
            resources_only = False
            layout |= path.endswith(('.cmake', 'CMakeLists.txt', '.gitmodules'))
            if '/tests/' not in path and '/test/' not in path:
                checks.add('NumericalIsolation')
        elif path.startswith('Source/Managers/'):
            build = True
            resources_only = False
            layout |= path.endswith(('.cmake', 'CMakeLists.txt', '.gitmodules'))
            checks.add('NumericalIsolation')
        elif path.startswith('.github/'):
            name = Path(path).name
            if name in {'build_cache_state.py', 'test_build_cache_state.py'}:
                scripts.add('cache')
            elif name in {'verify_ctest_results.py', 'test_verify_ctest_results.py'}:
                scripts.add('ctest')
            else:
                scripts.add('ci')
        elif path.startswith('cmake/TestNumericalIsolation'):
            checks.update(CHECKS[:2])
        elif path in {'cmake/NeMo2Stage.cmake', 'cmake/TestStageInstall.cmake'}:
            checks.add('NeMo2StageInstall')
        elif path in {'cmake/TestSubproject.cmake'}:
            checks.add('NeMo2Subproject')
        elif path in {'cmake/TestBuildRegeneration.cmake'}:
            checks.add('CMakeRegeneration')
        elif path.endswith(('.cmake', 'CMakeLists.txt')) or path in {'CMakePresets.json', '.gitmodules'}:
            layout = True
            checks.update(CHECKS)
            build |= not path.startswith('cmake/Test')
            resources_only = False
        elif path.startswith('cmake/'):
            # Platform setup and diagnostic tools have their own workflow checks.
            scripts.add('ci')
    return dict(modules=sorted(modules), checks=sorted(checks), scripts=sorted(scripts),
                build=build, layout=layout, resources_only=build and resources_only,
                test_targets={name: sorted(targets) for name, targets in test_targets.items() if name not in whole_modules})


def selection(modules, checks, test_targets=None):
    if any(name not in OWNERS and not re.fullmatch(r'(bb|plugin):[A-Za-z0-9_-]+', name) for name in modules) or set(checks) - set(CHECKS):
        raise ValueError('unknown test module or aggregator check')
    result = {'modules': sorted(set(modules)), 'checks': sorted(set(checks))}
    if test_targets:
        if set(test_targets) - set(modules) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', target) for targets in test_targets.values() for target in targets):
            raise ValueError('unknown test target or owner')
        result['test_targets'] = test_targets
    return result


def flags(scope, static):
    modules = set(scope['modules'])
    bb = [name.split(':', 1)[1] for name in modules if name.startswith('bb:')]
    plugins = [name.split(':', 1)[1] for name in modules if name.startswith('plugin:')]
    groups = modules | ({'buildingblocks'} if bb else set()) | ({'plugins'} if plugins else set())
    values = {'BUILD_TESTING': True, 'NEMO_BUILD_TESTS': False,
              **{prefix + '_BUILD_TESTS': name in groups for name, prefix in TEST_OPTIONS.items()},
              'VXMATH_BUILD_STATIC': static or 'vxmath' in modules,
              'CK2_BUILD_STATIC': static or bool(groups & {'ck2', 'renderengine', 'buildingblocks'}),
              'CKRE_BUILD_STATIC': static or bool(groups & {'renderengine', 'buildingblocks'}),
              'CKRE_BUILD_TOOLS': 'renderengine' in modules and not static}
    target_filter = ';'.join(test_owners(scope['modules'])[name] + '/tests=' + '|'.join(targets)
                             for name, targets in scope.get('test_targets', {}).items())
    return ([f'-D{key}={"ON" if value else "OFF"}' for key, value in values.items()] +
            ['-DNEMO_TEST_TARGET_FILTER=' + target_filter] +
            ['-DCKBB_TEST_MODULES=' + (';'.join(sorted(bb)) if 'buildingblocks' not in modules else ''),
             '-DCKPLUGINS_TEST_MODULES=' + (';'.join(sorted(plugins)) if 'plugins' not in modules else '')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['select', 'flags'])
    parser.add_argument('--base', default='')
    parser.add_argument('--head', default='HEAD')
    parser.add_argument('--modules', default='changed')
    parser.add_argument('--selection', default='')
    parser.add_argument('--static', default='runtime')
    parser.add_argument('--target', default='all')
    args = parser.parse_args()
    if args.mode == 'flags':
        print('\n'.join(flags(json.loads(args.selection), args.static == 'static')))
        return
    if args.selection:
        scope = selection(**json.loads(args.selection))
        result = dict(scope, build=bool(scope['modules']), layout=False, scripts=[])
    else:
        result = classify(changed_paths(Path.cwd(), baseline(Path.cwd(), args.base, '.github/scripts/engine_ci.py'), args.head))
        if args.modules != 'changed':
            scope = selection(list(OWNERS) if args.modules == 'all' else
                              [name for name in args.modules.split(',') if name != 'integration'],
                              CHECKS if args.modules in {'all', 'integration'} else [])
            result.update(scope)
            result.pop('test_targets', None)
            result['resources_only'] = False
    scope = selection(result['modules'], result['checks'], result.get('test_targets'))
    encoded = json.dumps(scope, separators=(',', ':'))
    result.update(selection=encoded, test=bool(scope['modules'] or scope['checks']),
                  compile=bool(scope['modules']), cache_scope=hashlib.sha256(encoded.encode()).hexdigest()[:12],
                  cache_profile='render-tests' if 'renderengine' in scope['modules'] else 'build',
                  matrix=job_matrix(args.target, result.get('resources_only', False) and args.target == 'all'))
    emit(result)


if __name__ == '__main__':
    main()
