import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from change_detection import changed_paths
from engine_ci import classify, flags, OWNERS
from ctest_selection import select_tests


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

    def test_changed_golden_image_belongs_to_renderer(self):
        self.assertEqual(classify(['Source/RenderEngine/tests/reference/frame.png'])['modules'],
                         ['renderengine'])

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
