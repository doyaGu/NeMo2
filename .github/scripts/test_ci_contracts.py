import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from engine_ci import classify, selection
from ctest_selection import validate_selection
from run_cmake_checks import run_checks, selected_checks


ENGINE = Path(__file__).resolve().parents[2]


class BuildSelectionTest(unittest.TestCase):
    def test_test_only_changes_do_not_request_a_runtime_build(self):
        for path in ('Source/CK2/tests/CKRandomContextTest.cpp',
                     'Source/RenderEngine/tests/helper.h',
                     'Source/BuildingBlocks/Logics/tests/RandomContextTest.cpp'):
            with self.subTest(path=path):
                scope = classify([path])
                self.assertFalse(scope['build'])
                self.assertTrue(scope['modules'])
        self.assertTrue(classify(['Source/CK2/tests/CKRandomContextTest.cpp',
                                  'Source/CK2/src/CKContext.cpp'])['build'])
        helper = classify(['cmake/NeMo2TestTargets.cmake'])
        self.assertFalse(helper['build'])
        self.assertEqual(helper['modules'], [])
        self.assertEqual(helper['scripts'], ['ci'])

    def test_reusable_workflow_preserves_test_only_selection(self):
        payload = selection(['ck2'], [], {'ck2': ['CKRandomContextTest']}, False)
        output = subprocess.check_output([
            'python3', '-B', str(ENGINE / '.github/scripts/engine_ci.py'), 'select',
            '--selection', json.dumps(payload), '--target', 'sdl-macos-arm64-runtime'], text=True)
        result = json.loads(output)
        self.assertFalse(result['build'])
        self.assertTrue(result['compile'])
        self.assertEqual(json.loads(result['selection']), payload)

    def test_nightly_baseline_uses_a_success_even_after_many_failures(self):
        workflow = (ENGINE / '.github/workflows/build.yml').read_text()
        command = re.search(r'BASE=\$\((gh run list.*?)\)\n', workflow, re.S)[1]
        # Model an API page containing only recent failures unless success is
        # requested server-side. The successful commit lies beyond that page.
        script = '''gh() {
          case " $* " in
            *" --status success "*) printf '%040d' 1;;
            *) printf '%040d' 2;;
          esac
        }
        ''' + 'BASE=$(' + command + ')\nprintf "%s" "$BASE"'
        self.assertEqual(subprocess.check_output(['bash', '-c', script], text=True), f'{1:040d}')

    def test_stage_changes_are_wired_to_lightweight_validate(self):
        for path in ('cmake/NeMo2Stage.cmake', 'cmake/NeMo2StageInstall.cmake'):
            scope = classify([path])
            self.assertEqual(scope['modules'], [])
            self.assertEqual(selected_checks(ENGINE, scope['checks']), ['NeMo2StageInstall'])
        workflow = (ENGINE / '.github/workflows/validate.yml').read_text()
        self.assertIn("needs.changes.outputs.light_checks != '[]'", workflow)
        self.assertIn("run_cmake_checks.py --checks '${{ needs.changes.outputs.light_checks }}'", workflow)

    def test_test_target_build_does_not_build_unrelated_runtime_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / 'Source/CK2/tests'
            tests.mkdir(parents=True)
            (root / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(TargetScope LANGUAGES CXX)\n'
                'set(NEMO_TEST_TARGET_FILTER "Source/CK2/tests=Selected")\n'
                'add_subdirectory(Source)\n'
                f'include("{ENGINE.as_posix()}/cmake/NeMo2TestTargets.cmake")\n')
            (root / 'Source/CMakeLists.txt').write_text('add_subdirectory(CK2/tests)\n')
            (tests / 'main.cpp').write_text('int main() { return 0; }\n')
            (tests / 'CMakeLists.txt').write_text(
                'add_executable(Selected main.cpp)\nadd_executable(Unselected main.cpp)\n'
                'add_library(UnrelatedRuntime STATIC main.cpp)\n'
                'foreach(target Selected Unselected UnrelatedRuntime)\n'
                'add_custom_command(TARGET ${target} POST_BUILD COMMAND "${CMAKE_COMMAND}" '
                '-E touch "${CMAKE_BINARY_DIR}/${target}.built")\nendforeach()\n')
            build = root / 'build'
            subprocess.run(['cmake', '-S', str(root), '-B', str(build)], check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run(['cmake', '--build', str(build), '--target', 'nemo_ci_tests'],
                           check=True, stdout=subprocess.DEVNULL)
            self.assertTrue((build / 'Selected.built').exists())
            self.assertFalse((build / 'Unselected.built').exists())
            self.assertFalse((build / 'UnrelatedRuntime.built').exists())


class RequiredTestsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd().resolve()
        self.owners = {'ck2': self.root / 'Source/CK2', 'vxmath': self.root / 'Source/VxMath'}
        self.metadata = {'tests': [{'name': 'One', 'backtrace': 0, 'command': ['/bin/One']}],
                         'backtraceGraph': {'nodes': [{'file': 0}],
                                            'files': [str(self.root / 'Source/CK2/tests/CMakeLists.txt')]}}

    def test_missing_owner_fails_even_when_another_owner_has_tests(self):
        with self.assertRaisesRegex(ValueError, 'vxmath'):
            validate_selection(self.metadata, self.owners, {'modules': ['ck2', 'vxmath'], 'checks': []})

    def test_missing_named_check_and_target_fail(self):
        with self.assertRaisesRegex(ValueError, 'MissingCheck'):
            validate_selection(self.metadata, self.owners, {'modules': ['ck2'], 'checks': ['MissingCheck']})
        with self.assertRaisesRegex(ValueError, 'MissingTarget'):
            validate_selection(self.metadata, self.owners,
                               {'modules': ['ck2'], 'checks': [], 'test_targets': {'ck2': ['One', 'MissingTarget']}})

    def test_empty_scope_is_an_intentional_noop(self):
        self.assertEqual(validate_selection(self.metadata, self.owners, {'modules': [], 'checks': []}), {})

    def test_only_explicit_platform_exceptions_allow_missing_tests(self):
        scope = {'modules': ['bb:Interface'], 'checks': []}
        skipped = validate_selection(self.metadata, self.owners, scope, platform='win32')
        self.assertTrue(skipped['bb:Interface'])
        with self.assertRaisesRegex(ValueError, 'bb:Interface'):
            validate_selection(self.metadata, self.owners, scope, platform='linux')
        with self.assertRaisesRegex(ValueError, 'bb:Unknown'):
            validate_selection(self.metadata, self.owners, {'modules': ['bb:Unknown'], 'checks': []})

    def test_static_plugin_exception_requires_the_configured_policy(self):
        scope = {'modules': ['plugin:AVIReader'], 'checks': []}
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            with self.assertRaises(ValueError):
                validate_selection(self.metadata, self.owners, scope, build_dir=build)
            (build / 'CMakeCache.txt').write_text('CKPLUGINS_BUILD_SHARED:BOOL=OFF\nCKPLUGINS_BUILD_TESTS:BOOL=ON\n')
            self.assertIn('plugin:AVIReader', validate_selection(self.metadata, self.owners, scope, build_dir=build))

    def test_runner_fails_missing_selection_instead_of_reporting_a_green_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.25)\nproject(Empty NONE)\nenable_testing()\n')
            build = root / 'build'
            subprocess.run(['cmake', '-S', str(root), '-B', str(build)], check=True,
                           stdout=subprocess.DEVNULL)
            runner = ENGINE / '.github/scripts/ctest_selection.py'
            output = root / 'outputs'
            command = ['python3', '-B', str(runner), '--build-dir', str(build), '--selection']
            env = dict(os.environ, GITHUB_OUTPUT=str(output))
            failed = subprocess.run(command + [json.dumps({'modules': ['ck2'], 'checks': []})],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('No registered tests match requested module ck2', failed.stderr)
            self.assertFalse(output.exists())
            subprocess.run(command + [json.dumps({'modules': [], 'checks': []})],
                           cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)
            self.assertIn('has_tests=false', output.read_text())

    def test_explicit_static_plugin_skip_keeps_named_checks_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(StaticReaders NONE)\nenable_testing()\n'
                'set(CKPLUGINS_BUILD_SHARED OFF CACHE BOOL "")\n'
                'set(CKPLUGINS_BUILD_TESTS ON CACHE BOOL "")\n'
                'add_test(NAME NumericalIsolation COMMAND "${CMAKE_COMMAND}" -E true)\n')
            build = root / 'build'
            subprocess.run(['cmake', '-S', str(root), '-B', str(build)], check=True,
                           stdout=subprocess.DEVNULL)
            output = root / 'outputs'
            subprocess.run(['python3', '-B', str(ENGINE / '.github/scripts/ctest_selection.py'),
                            '--build-dir', str(build), '--selection',
                            json.dumps({'modules': ['plugins'], 'checks': ['NumericalIsolation']})],
                           cwd=root, env=dict(os.environ, GITHUB_OUTPUT=str(output)),
                           check=True, stdout=subprocess.DEVNULL)
            self.assertIn('has_tests=true', output.read_text())
            self.assertIn('require_gtest=false', output.read_text())
            self.assertTrue((build / 'ctest-results.xml').exists())


class CMakeCheckRunnerTest(unittest.TestCase):
    def test_only_requested_checks_execute_and_unknown_names_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.github/scripts').mkdir(parents=True)
            (root / 'selected.cmake').write_text('file(WRITE "${TEST_BINARY_DIR}/ran" "yes")\n')
            (root / 'unselected.cmake').write_text('message(FATAL_ERROR "unselected check ran")\n')
            (root / '.github/scripts/cmake_checks.json').write_text(json.dumps({
                'Selected': {'script': 'selected.cmake'}, 'Unselected': {'script': 'unselected.cmake'}}))
            run_checks(root, ['Selected'])
            self.assertTrue((root / 'build/ci-checks/ran').exists())
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                run_checks(root, ['Unknown'])


if __name__ == '__main__':
    unittest.main()
