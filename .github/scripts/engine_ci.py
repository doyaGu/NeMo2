"""Select engine test modules from actual source changes."""

import argparse
import hashlib
import json
from pathlib import Path

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


def classify(paths):
    modules, checks, scripts = set(), set(), set()
    build = layout = False
    for path in paths:
        if documentation(path):
            continue
        owner = next((name for name, prefix in sorted(OWNERS.items(), key=lambda item: -len(item[1]))
                      if path == prefix or path.startswith(prefix + '/')), None)
        if owner:
            modules.add(owner)
            if owner == 'ivp':
                modules.add('buildingblocks')
            build = True
            layout |= path.endswith(('.cmake', 'CMakeLists.txt', '.gitmodules'))
            if '/tests/' not in path and '/test/' not in path:
                checks.add('NumericalIsolation')
        elif path.startswith('Source/Managers/'):
            build = True
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
        elif path.startswith('cmake/'):
            # Platform setup and diagnostic tools have their own workflow checks.
            scripts.add('ci')
    return dict(modules=sorted(modules), checks=sorted(checks), scripts=sorted(scripts),
                build=build, layout=layout)


def selection(modules, checks):
    if set(modules) - OWNERS.keys() or set(checks) - set(CHECKS):
        raise ValueError('unknown test module or aggregator check')
    return {'modules': sorted(set(modules)), 'checks': sorted(set(checks))}


def flags(scope, static):
    modules = set(scope['modules'])
    values = {'BUILD_TESTING': True, 'NEMO_BUILD_TESTS': False,
              **{prefix + '_BUILD_TESTS': name in modules for name, prefix in TEST_OPTIONS.items()},
              'VXMATH_BUILD_STATIC': static or 'vxmath' in modules,
              'CK2_BUILD_STATIC': static or bool(modules & {'ck2', 'renderengine', 'buildingblocks'}),
              'CKRE_BUILD_STATIC': static or bool(modules & {'renderengine', 'buildingblocks'}),
              'CKRE_BUILD_TOOLS': 'renderengine' in modules and not static}
    return [f'-D{key}={"ON" if value else "OFF"}' for key, value in values.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['select', 'flags'])
    parser.add_argument('--base', default='')
    parser.add_argument('--head', default='HEAD')
    parser.add_argument('--modules', default='changed')
    parser.add_argument('--selection', default='')
    parser.add_argument('--static', default='runtime')
    args = parser.parse_args()
    if args.mode == 'flags':
        print('\n'.join(flags(json.loads(args.selection), args.static == 'static')))
        return
    if args.selection:
        scope = selection(**json.loads(args.selection))
        result = dict(scope, build=bool(scope['modules']), layout=False, scripts=[])
    else:
        result = classify(changed_paths(Path.cwd(), baseline(Path.cwd(), args.base), args.head))
        if args.modules != 'changed':
            scope = selection(list(OWNERS) if args.modules == 'all' else
                              [name for name in args.modules.split(',') if name != 'integration'],
                              CHECKS if args.modules in {'all', 'integration'} else [])
            result.update(scope)
    scope = selection(result['modules'], result['checks'])
    encoded = json.dumps(scope, separators=(',', ':'))
    result.update(selection=encoded, test=bool(scope['modules'] or scope['checks']),
                  compile=bool(scope['modules']), cache_scope=hashlib.sha256(encoded.encode()).hexdigest()[:12])
    emit(result)


if __name__ == '__main__':
    main()
