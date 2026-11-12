import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from change_detection import baseline, changed_paths, ensure_commit
from engine_ci import classify, flags, OWNERS, job_matrix, test_owners
from ctest_selection import select_tests, select_indices


class SelectionTest(unittest.TestCase):
    def test_documentation_and_licenses_do_not_build_or_test(self):
        result = classify(['README.md', 'BUILD.md', 'LICENSE', 'LICENSES/notice.txt',
                           'Source/CK2/README.md', 'Source/RenderEngine/docs/design.md'])
        self.assertFalse(result['build'])
        self.assertEqual(result['modules'], [])
        self.assertEqual(result['checks'], [])

    def test_only_changed_module_enables_its_suite(self):
        result = classify(['Source/VxMath/src/VxVector.cpp'])
        self.assertEqual(result['modules'], ['vxmath'])
        options = flags(result, False)
        self.assertIn('-DVXMATH_BUILD_TESTS=ON', options)
        for prefix in ('CK2', 'CKRE', 'CKBB', 'CKPLUGINS', 'IVP'):
            self.assertIn(f'-D{prefix}_BUILD_TESTS=OFF', options)
        self.assertIn('-DNEMO_BUILD_TESTS=OFF', options)

    def test_version_resources_do_not_enable_engine_suites(self):
        result = classify(['Source/VxMath/src/Version.rc.in', 'Source/CK2/src/Version.rc.in',
                           'Source/RenderEngine/src/Version.rc.in', 'Source/Plugins/Version.rc.in',
                           'Source/BuildingBlocks/Version.rc.in',
                           'Source/Managers/ParameterOperations/Version.rc.in'])
        self.assertTrue(result['build'])
        self.assertFalse(result['layout'])
        self.assertEqual(result['modules'], [])
        self.assertEqual(result['checks'], [])

    def test_resources_do_not_hide_other_module_changes(self):
        result = classify(['Source/VxMath/src/Version.rc.in', 'Source/CK2/src/CKObject.cpp'])
        self.assertEqual(result['modules'], ['ck2'])
        self.assertEqual(result['checks'], ['NumericalIsolation'])

    def test_runtime_resources_still_enable_the_owning_suite(self):
        result = classify(['Source/RenderEngine/src/Shaders.rc'])
        self.assertEqual(result['modules'], ['renderengine'])

    def test_changed_golden_image_belongs_to_renderer(self):
        self.assertEqual(classify(['Source/RenderEngine/tests/reference/frame.png'])['modules'],
                         ['renderengine'])

    def test_tt_changes_do_not_select_physics_or_other_building_blocks(self):
        result = classify(['Source/BuildingBlocks/TT_InterfaceManager_RT/InterfaceManager.cpp'])
        self.assertEqual(result['modules'], ['bb:TT_InterfaceManager_RT'])
        options = flags(result, False)
        self.assertIn('-DCKBB_TEST_MODULES=TT_InterfaceManager_RT', options)
        self.assertIn('-DIVP_BUILD_TESTS=OFF', options)
        self.assertIn('-DCKRE_BUILD_TESTS=OFF', options)

    def test_reader_changes_do_not_select_other_readers(self):
        result = classify(['Source/Plugins/AVIReader/FrameDecoder.cpp'])
        self.assertEqual(result['modules'], ['plugin:AVIReader'])
        self.assertIn('-DCKPLUGINS_TEST_MODULES=AVIReader', flags(result, False))

    def test_shared_build_files_and_ivp_keep_their_dependents(self):
        self.assertEqual(classify(['Source/BuildingBlocks/CMakeLists.txt'])['modules'], ['buildingblocks'])
        self.assertEqual(classify(['Source/BuildingBlocks/Materials-Textures/Behaviors/Create.cpp'])['modules'], ['bb:Materials-Textures'])
        self.assertEqual(classify(['Source/BuildingBlocks/physics_RT/ivp/solver.cxx'])['modules'], ['bb:physics_RT', 'ivp'])

    def test_windows_resources_only_schedule_windows_dynamic_jobs(self):
        result = classify(['Source/VxMath/src/Version.rc.in', 'README.md'])
        rows = job_matrix(resources_only=result['resources_only'])['include']
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row['platform']['name'].startswith('windows-') and row['linkage']['name'] == 'runtime' for row in rows))
        mixed = classify(['Source/VxMath/src/Version.rc.in', 'Source/CK2/src/CKObject.cpp'])
        self.assertEqual(len(job_matrix(resources_only=mixed['resources_only'])['include']), 14)

    def test_manual_target_creates_one_job_instead_of_thirteen_empty_jobs(self):
        rows = job_matrix('sdl-macos-arm64-static')['include']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['platform']['name'], 'macos-arm64')
        self.assertEqual(rows[0]['linkage']['name'], 'static')
        with self.assertRaises(ValueError):
            job_matrix('unknown')

    def test_reader_path_tests_keep_component_ownership_with_shared_registration(self):
        metadata = {'tests': [
            {'name': 'AVIReaderAnsiPathTest', 'properties': [{'name': 'LABELS', 'value': ['plugin:AVIReader']}]},
            {'name': 'ImageReaderAnsiPathTest', 'properties': [{'name': 'LABELS', 'value': ['plugin:ImageReader']}]}],
            'backtraceGraph': {'nodes': [], 'files': []}}
        self.assertEqual(select_tests(metadata, {}, ['plugin:AVIReader'], []), ['AVIReaderAnsiPathTest'])
        self.assertEqual(len(select_tests(metadata, {}, ['plugins'], [])), 2)

    def test_component_runner_never_executes_other_bb_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(Selection NONE)\nenable_testing()\n'
                'add_subdirectory(Source/BuildingBlocks/TT_InterfaceManager_RT)\n'
                'add_subdirectory(Source/BuildingBlocks/physics_RT)\n')
            for name, command in [('TT_InterfaceManager_RT', 'touch'), ('physics_RT', 'false')]:
                folder = root / 'Source/BuildingBlocks' / name
                folder.mkdir(parents=True)
                (folder / 'CMakeLists.txt').write_text(
                    f'add_test(NAME {name}Case COMMAND "${{CMAKE_COMMAND}}" -E {command}'
                    + (' "${CMAKE_BINARY_DIR}/selected-ran"' if command == 'touch' else '') + ')\n')
            build = root / 'build'
            subprocess.run(['cmake', '-S', str(root), '-B', str(build)], check=True, stdout=subprocess.DEVNULL)
            runner = Path(__file__).with_name('ctest_selection.py').resolve()
            subprocess.run([sys.executable, '-B', str(runner), '--build-dir', str(build),
                            '--selection', json.dumps({'modules': ['bb:TT_InterfaceManager_RT'], 'checks': []})],
                           cwd=root, check=True, stdout=subprocess.DEVNULL)
            self.assertTrue((build / 'selected-ran').exists())
            self.assertNotIn('physics_RTCase', (build / 'ctest-results.xml').read_text())

    def test_cmake_filters_register_only_requested_components(self):
        engine = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bb = (engine / 'Source/BuildingBlocks/cmake/CKBBTests.cmake').as_posix()
            plugins = (engine / 'Source/Plugins/cmake/CKPluginsTests.cmake').as_posix()
            (root / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(Filter NONE)\n'
                'set(BUILD_TESTING ON)\nset(CKBB_BUILD_TESTS ON)\nset(CKPLUGINS_BUILD_TESTS ON)\n'
                f'include("{bb}")\ninclude("{plugins}")\n'
                'set(CKBB_TEST_MODULES TT_InterfaceManager_RT)\n'
                'ckbb_tests_enabled(TT_InterfaceManager_RT tt)\nckbb_tests_enabled(physics_RT physics)\n'
                'set(CKPLUGINS_TEST_MODULES AVIReader)\n'
                'ckplugins_tests_enabled(AVIReader avi)\nckplugins_tests_enabled(ImageReader image)\n'
                'if (NOT tt OR physics OR NOT avi OR image)\nmessage(FATAL_ERROR "Unrelated tests enabled")\nendif()\n'
                'set(CKBB_TEST_MODULES "")\nckbb_tests_enabled(physics_RT all)\n'
                'if (NOT all)\nmessage(FATAL_ERROR "Full-module default changed")\nendif()\n')
            subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build')], check=True, stdout=subprocess.DEVNULL)

    def test_test_only_changes_select_verified_targets_and_keep_shared_helpers(self):
        scope = classify(['Source/CK2/tests/CKBufferParserFlushTest.cpp'])
        self.assertEqual(scope['test_targets'], {'ck2': ['CKBufferParserFlushTest']})
        self.assertEqual(classify(['Source/VxMath/tests/VxMath/NativeUtf8ProcessTest.cpp'])['test_targets'], {})
        self.assertEqual(classify(['Source/VxMath/tests/VxMath/ConfigLifecycleReferenceTest.cpp'])['test_targets'], {})
        mixed = classify(['Source/CK2/tests/CKBufferParserFlushTest.cpp', 'Source/CK2/src/CKContext.cpp'])
        self.assertEqual(mixed['test_targets'], {})
        metadata = {'tests': [
            {'name': 'Selected', 'backtrace': 0, 'command': ['/build/CKBufferParserFlushTest']},
            {'name': 'Other', 'backtrace': 0, 'command': ['/build/CKFileIoTest']}],
            'backtraceGraph': {'nodes': [{'file': 0}], 'files': [str(Path.cwd() / 'Source/CK2/tests/CMakeLists.txt')]}}
        self.assertEqual(select_indices(metadata, {'ck2': Path.cwd() / 'Source/CK2'}, ['ck2'], [],
                                        test_targets=scope['test_targets']), [1])

    def test_warm_release_and_render_snapshots_do_not_recompile_each_other(self):
        # A real compiler counter checks cache reuse, and the same fixture proves
        # unselected test executables are omitted from the default build.
        engine = Path(__file__).resolve().parents[2]
        helper = (engine / 'cmake/NeMo2TestTargets.cmake').as_posix()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, build = root / 'source', root / 'build'
            (source / 'Source').mkdir(parents=True)
            counter = root / 'compiler-count.txt'
            launcher = root / 'launcher.py'
            launcher.write_text('import pathlib,subprocess,sys\n'
                                f'with pathlib.Path({str(counter)!r}).open("a") as log: log.write("compile\\n")\n'
                                'sys.exit(subprocess.call(sys.argv[1:]))\n')
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(CacheProbe LANGUAGES CXX)\n'
                f'set(CMAKE_CXX_COMPILER_LAUNCHER "{sys.executable};{launcher.as_posix()}")\n'
                'add_subdirectory(Source)\nset(NEMO_TEST_TARGET_FILTER "Source=Selected")\n'
                f'include("{helper}")\n')
            (source / 'Source/CMakeLists.txt').write_text(
                'add_executable(Selected probe.cpp)\nadd_executable(Unselected probe.cpp)\n'
                'if(HOOKS)\ntarget_compile_definitions(Selected PRIVATE TEST_HOOKS=1)\nendif()\n')
            (source / 'Source/probe.cpp').write_text(
                '#include <cstdio>\n#ifndef TEST_HOOKS\n#define TEST_HOOKS 0\n#endif\n'
                'int main(){std::printf("%d",TEST_HOOKS);}\n')
            generator = 'Ninja' if shutil.which('ninja') else None

            def configure_and_build(hooks):
                command = ['cmake', '-S', str(source), '-B', str(build), f'-DHOOKS={"ON" if hooks else "OFF"}']
                if generator:
                    command += ['-G', generator]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                subprocess.run(['cmake', '--build', str(build), '--config', 'Release'], check=True, stdout=subprocess.DEVNULL)
                suffix = '.exe' if sys.platform == 'win32' else ''
                binaries = build / 'Source'
                if (binaries / 'Release').exists():
                    binaries /= 'Release'
                self.assertFalse((binaries / ('Unselected' + suffix)).exists())
                self.assertEqual(subprocess.check_output([str(binaries / ('Selected' + suffix))], text=True), str(int(hooks)))

            for profile, hooks in [('build', False), ('render-tests', True)]:
                configure_and_build(hooks)
                shutil.copytree(build, root / 'snapshots' / profile)
            self.assertEqual(len(counter.read_text().splitlines()), 2)
            for profile, hooks in [('build', False), ('render-tests', True)]:
                shutil.rmtree(build)
                shutil.copytree(root / 'snapshots' / profile, build)
                configure_and_build(hooks)
            self.assertEqual(len(counter.read_text().splitlines()), 2)

    def test_stage_check_does_not_enable_engine_suites(self):
        result = classify(['cmake/TestStageInstall.cmake'])
        self.assertFalse(result['build'])
        self.assertEqual(result['modules'], [])
        self.assertEqual(result['checks'], ['NeMo2StageInstall'])

    def test_changed_ci_tools_select_only_their_checks(self):
        self.assertEqual(classify(['.github/scripts/build_cache_state.py'])['scripts'], ['cache'])
        result = classify(['.github/workflows/test.yml'])
        self.assertEqual(result['scripts'], ['ci'])
        self.assertFalse(result['build'])
        self.assertEqual(result['modules'], [])

    def test_recursive_submodule_diff_ignores_documentation_only_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            module, engine = root / 'module', root / 'engine'
            for repo in (module, engine):
                repo.mkdir()
                self.git(repo, 'init', '-q')
            (module / 'README.md').write_text('initial docs')
            (module / 'source.cpp').write_text('int value = 1;')
            self.commit(module)
            self.git(engine, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q',
                     str(module), 'Source/VxMath')
            self.commit(engine)
            before = self.git(engine, 'rev-parse', 'HEAD').strip()
            child = engine / 'Source/VxMath'
            (child / 'README.md').write_text('updated docs')
            self.commit(child)
            self.commit(engine)
            after_docs = self.git(engine, 'rev-parse', 'HEAD').strip()
            self.assertEqual(classify(changed_paths(engine, before, after_docs))['modules'], [])
            self.assertFalse(classify(changed_paths(engine, before, after_docs))['build'])
            (child / 'source.cpp').write_text('int value = 2;')
            self.commit(child)
            self.commit(engine)
            self.assertEqual(classify(changed_paths(engine, after_docs))['modules'], ['vxmath'])

    @staticmethod
    def git(root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                       stderr=subprocess.PIPE)

    def commit(self, root):
        self.git(root, 'add', '.')
        self.git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-qm', 'fixture')

    def test_first_nightly_does_not_replay_pre_policy_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.git(root, 'init', '-q')
            (root / 'source.cpp').write_text('old source')
            self.commit(root)
            old = self.git(root, 'rev-parse', 'HEAD').strip()
            (root / 'source.cpp').write_text('activation snapshot')
            self.commit(root)
            start = self.git(root, 'rev-parse', 'HEAD').strip()
            (root / 'policy.py').write_text('selective policy')
            self.commit(root)
            (root / 'README.md').write_text('later documentation')
            self.commit(root)
            with patch.dict('os.environ', {'GITHUB_EVENT_NAME': 'schedule'}):
                self.assertEqual(baseline(root, old, 'policy.py'), start)
                self.assertEqual(baseline(root, '', 'policy.py'), start)
            with patch.dict('os.environ', {'GITHUB_EVENT_NAME': 'push'}):
                self.assertEqual(baseline(root, old, 'policy.py'), old)

    def test_shallow_fetch_race_retries_without_recursing_other_modules(self):
        missing = subprocess.CompletedProcess([], 1)
        raced = subprocess.CompletedProcess([], 128, b'', b'fatal: shallow file has changed since we read it')
        fetched = subprocess.CompletedProcess([], 0, b'', b'')
        with patch('change_detection.subprocess.run', side_effect=[missing, raced, missing, fetched]) as run, patch('change_detection.time.sleep'):
            ensure_commit(Path('/fixture'), 'a' * 40)
            self.assertEqual(sum('fetch' in call.args[0] for call in run.call_args_list), 2)
            self.assertTrue(all('fetch.recurseSubmodules=false' in call.args[0] for call in run.call_args_list if 'fetch' in call.args[0]))
        invalid = subprocess.CompletedProcess([], 128, b'', b'fatal: not our ref')
        with patch('change_detection.subprocess.run', side_effect=[missing, invalid]):
            with self.assertRaises(subprocess.CalledProcessError):
                ensure_commit(Path('/fixture'), 'b' * 40)

    def test_ctest_runner_never_executes_unselected_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(Selection NONE)\n'
                'enable_testing()\nadd_subdirectory(Source/VxMath)\nadd_subdirectory(Source/CK2)\n')
            for name, command in [('VxMath', 'touch'), ('CK2', 'false')]:
                folder = root / 'Source' / name
                folder.mkdir(parents=True)
                (folder / 'CMakeLists.txt').write_text(
                    f'add_test(NAME {name}Case COMMAND "${{CMAKE_COMMAND}}" -E {command}'
                    + (' "${CMAKE_BINARY_DIR}/selected-ran"' if command == 'touch' else '') + ')\n')
            build = root / 'build'
            subprocess.run(['cmake', '-S', str(root), '-B', str(build)], check=True,
                           stdout=subprocess.DEVNULL)
            runner = Path(__file__).with_name('ctest_selection.py').resolve()
            subprocess.run([sys.executable, '-B', str(runner), '--build-dir', str(build),
                            '--selection', json.dumps({'modules': ['vxmath'], 'checks': []})],
                           cwd=root, check=True, stdout=subprocess.DEVNULL)
            self.assertTrue((build / 'selected-ran').exists())
            report = (build / 'ctest-results.xml').read_text()
            self.assertIn('VxMathCase', report)
            self.assertNotIn('CK2Case', report)

    def test_discovered_cases_keep_their_module_owner(self):
        root = Path.cwd().resolve()
        metadata = {'tests': [{'name': 'Discovered', 'properties': [
            {'name': 'WORKING_DIRECTORY', 'value': str(root / 'build/Source/Plugins/tests')}]}],
            'backtraceGraph': {'nodes': [], 'files': []}}
        owners = {'plugins': root / 'Source/Plugins'}
        self.assertEqual(select_tests(metadata, owners, ['plugins'], [], root / 'build', root),
                         ['Discovered'])
        self.assertEqual(select_tests(metadata, owners, ['vxmath'], [], root / 'build', root), [])

    def test_empty_selection_never_falls_back_to_all_tests(self):
        metadata = {'tests': [{'name': 'Other'}], 'backtraceGraph': {'nodes': [], 'files': []}}
        self.assertEqual(select_tests(metadata, {}, [], []), [])


if __name__ == '__main__':
    unittest.main()
