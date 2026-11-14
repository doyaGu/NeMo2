"""Exercise module composition through the real aggregator CMake interface."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ENGINE = Path(__file__).resolve().parents[2]


class ModuleCompositionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.25)
project(ModuleComposition NONE)
list(PREPEND CMAKE_MODULE_PATH "{ENGINE.as_posix()}/cmake")
include(NeMo2Cache)
include(NeMo2Options)
function(fixture_module name prefix)
    if (${{prefix}}_BUILD_SHARED)
        add_library(${{name}} INTERFACE)
    endif()
    if (${{prefix}}_BUILD_STATIC)
        add_library(${{name}}Static INTERFACE)
        if (NOT TARGET ${{name}})
            add_library(${{name}} ALIAS ${{name}}Static)
        endif()
    endif()
endfunction()
add_subdirectory(Source)
include(NeMo2ComponentRegistry)
file(WRITE "${{CMAKE_BINARY_DIR}}/selection.txt"
    "targets=${{NEMO_RUNTIME_TARGETS}}\\n"
    "libraries=${{NEMO_RUNTIME_LIBRARY_TARGETS}}\\n"
    "test_only=${{NEMO_TEST_ONLY_TARGETS}}\\n"
    "renderer_install=${{CKRE_INSTALL}}\\n"
    "supported=${{NEMO_SUPPORTED_MODULE_RUNTIME_TARGETS}}\\n"
    "renderer_static=${{CKRE_BUILD_STATIC}}\\n"
    "renderer_tests=${{CKRE_BUILD_TESTS}}\\n"
    "ck2_static=${{CK2_BUILD_STATIC}}\\n"
    "plugins_tests=${{CKPLUGINS_BUILD_TESTS}}\\n"
    "bb_tests=${{CKBB_BUILD_TESTS}}\\n")
''')
        sdl = self.source / 'sdl'
        sdl.mkdir()
        (sdl / 'SDL3Config.cmake').write_text('add_library(SDL3::SDL3 INTERFACE IMPORTED)\n')
        (sdl / 'SDL3ConfigVersion.cmake').write_text(
            'set(PACKAGE_VERSION 3.4.20)\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n')
        source = self.source / 'Source'
        source.mkdir()
        shutil.copyfile(ENGINE / 'Source/CMakeLists.txt', source / 'CMakeLists.txt')
        for module, prefix in [('VxMath', 'VXMATH'), ('CK2', 'CK2')]:
            folder = source / module
            folder.mkdir()
            (folder / 'CMakeLists.txt').write_text(f'fixture_module({module} {prefix})\n')
        children = {
            'RenderEngine': '''
cmake_minimum_required(VERSION 3.16)
option(CKRE_BUILD_SDL_GPU_RASTERIZER "" ON)
fixture_module(CK2_3D CKRE)
if(CKRE_BUILD_SDL_GPU_RASTERIZER)
    fixture_module(CKSdlGpuRasterizer CKRE)
endif()
set_property(GLOBAL PROPERTY CKRE_RUNTIME_CONFIGS CK2_3D.ini)
''',
            'Managers': '''
cmake_minimum_required(VERSION 3.16)
option(CKMANAGERS_BUILD_ALL "" ON)
foreach(entry "SdlInputManager|SDLINPUT|SDLINPUT" "SdlSoundManager|SDLSOUND|SDLSOUND" "ParameterOperations|PARAMETEROPERATIONS|CKPARAMOP")
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 name)
    list(GET fields 1 option)
    list(GET fields 2 prefix)
    option(CKMANAGERS_BUILD_${option} "" ON)
    if(CKMANAGERS_BUILD_ALL OR CKMANAGERS_BUILD_${option})
        fixture_module(${name} ${prefix})
    endif()
endforeach()
''',
            'Plugins': '''
cmake_minimum_required(VERSION 3.16)
foreach(name IN LISTS _nemo_plugin_modules)
    string(TOUPPER "${name}" option)
    option(CKPLUGINS_BUILD_${option} "" ON)
    if(CKPLUGINS_BUILD_${option})
        fixture_module(${name} CKPLUGINS)
    endif()
endforeach()
''',
            'BuildingBlocks': '''
cmake_minimum_required(VERSION 3.16)
option(CKBB_BUILD_ALL_MODULES "" ON)
foreach(name IN LISTS _nemo_bb_modules)
    if(name STREQUAL "MidiManager" AND NOT WIN32)
        continue()
    endif()
    set(option "${name}")
    if(name STREQUAL "Materials")
        set(option Materials-Textures)
    endif()
    option(CKBB_BUILD_${option} "" OFF)
    if(CKBB_BUILD_ALL_MODULES OR CKBB_BUILD_${option})
        fixture_module(${name} CKBB)
    endif()
endforeach()
'''}
        for name, content in children.items():
            folder = source / name
            folder.mkdir()
            (folder / 'CMakeLists.txt').write_text(content)

    def configure(self, modules, *, build='build', extra=(), success=True):
        directory = self.root / build
        result = subprocess.run(['cmake', '-S', str(self.source), '-B', str(directory),
                                 f'-DSDL3_DIR={self.source / "sdl"}', f'-DNEMO_MODULES={modules}',
                                 *extra], capture_output=True, text=True)
        if not success:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            return result.stderr
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return dict(line.split('=', 1) for line in (directory / 'selection.txt').read_text().splitlines())

    def test_core_only_and_independent_combinations(self):
        for modules in ('NONE', '', 'SdlSoundManager;ParameterOperations',
                        'ImageReader;Logics;Materials', 'TT_DatabaseManager_RT;TT_InterfaceManager_RT'):
            with self.subTest(modules=modules):
                result = self.configure(modules)
                expected = {'VxMath', 'CK2'} | (set(modules.split(';')) - {'NONE', ''})
                if 'Materials' in expected:
                    expected.add('CK2_3D')
                self.assertEqual(set(result['targets'].split(';')), expected)
                self.assertIn('TT_Gravity_RT', result['supported'].split(';'))
                self.assertEqual(result['renderer_static'], 'OFF')

    def test_rasterizer_adds_render_engine_without_unrelated_modules(self):
        result = self.configure('CKSdlGpuRasterizer;CKSdlGpuRasterizer')
        self.assertEqual(set(result['targets'].split(';')),
                         {'VxMath', 'CK2', 'CK2_3D', 'CKSdlGpuRasterizer'})

    def test_all_selects_the_available_catalogue(self):
        result = self.configure('ALL')
        expected = set(result['supported'].split(';')) | {'VxMath', 'CK2'}
        if sys.platform != 'win32':
            expected.discard('MidiManager')
        self.assertEqual(set(result['targets'].split(';')), expected)

    def test_core_suite_does_not_enable_excluded_module_suites(self):
        result = self.configure('NONE', extra=['-DNEMO_BUILD_TESTS=ON'])
        self.assertEqual(result['ck2_static'], 'ON')
        for key in ('renderer_static', 'renderer_tests', 'plugins_tests', 'bb_tests'):
            self.assertEqual(result[key], 'OFF')

    def test_composition_changes_select_their_configuration_regressions(self):
        from engine_ci import classify
        result = classify(['cmake/NeMo2Modules.cmake'])
        self.assertEqual(result['modules'], [])
        self.assertEqual(result['checks'], ['NeMo2RuntimeManifest', 'NeMo2StageInstall', 'NeMo2Subproject'])
        self.assertEqual(result['scripts'], ['ci'])
        dependencies = classify(['cmake/NeMo2ModuleDependencies.json'])
        self.assertEqual(result, dependencies)

    def test_static_composition_uses_the_same_module_names(self):
        result = self.configure('SdlInputManager;ImageReader;Logics', extra=['-DNEMO_BUILD_STATIC=ON'])
        self.assertEqual(set(result['targets'].split(';')),
                         {'VxMath', 'CK2', 'SdlInputManager', 'ImageReader', 'Logics'})

    def test_explicit_selection_does_not_overwrite_legacy_cache_options(self):
        flags = ['-DCKBB_BUILD_ALL_MODULES=OFF', '-DCKBB_BUILD_Logics=ON',
                 '-DCKMANAGERS_BUILD_ALL=OFF', '-DCKMANAGERS_BUILD_SDLINPUT=OFF',
                 '-DCKMANAGERS_BUILD_SDLSOUND=OFF', '-DCKMANAGERS_BUILD_PARAMETEROPERATIONS=ON',
                 '-DCKRE_BUILD_SDL_GPU_RASTERIZER=OFF']
        legacy = set(self.configure('AUTO', extra=flags)['targets'].split(';'))
        self.assertIn('Logics', legacy)
        self.assertNotIn('Cameras', legacy)
        selected = self.configure('SdlInputManager;Cameras')
        self.assertEqual(set(selected['targets'].split(';')), {'VxMath', 'CK2', 'CK2_3D', 'SdlInputManager', 'Cameras'})
        self.assertEqual(set(self.configure('AUTO')['targets'].split(';')), legacy)

    def test_test_dependencies_only_follow_selected_modules(self):
        for module, renderer, core_static in [('Logics', 'OFF', 'OFF'),
                                               ('TT_InterfaceManager_RT', 'OFF', 'ON'),
                                               ('physics_RT', 'ON', 'ON')]:
            with self.subTest(module=module):
                result = self.configure(module, build=module, extra=['-DCKBB_BUILD_TESTS=ON'])
                self.assertEqual(result['renderer_static'], renderer)
                self.assertEqual(result['ck2_static'], core_static)
                self.assertEqual(result['renderer_tests'], 'OFF')
                self.assertNotIn('CKSdlGpuRasterizer', result['targets'].split(';'))

    def test_invalid_and_unavailable_modules_fail_clearly(self):
        self.assertIn('Unknown NEMO_MODULES entry', self.configure('MisspelledModule', success=False))
        self.assertIn('produced no target', self.configure('Logics', success=False,
                      extra=['-DCKBB_BUILD_SHARED=OFF', '-DCKBB_BUILD_STATIC=OFF']))
        if sys.platform != 'win32':
            self.assertIn('supported only on Windows', self.configure('MidiManager', success=False))

    def test_test_filters_cannot_request_excluded_modules(self):
        self.assertIn('which NEMO_MODULES excludes', self.configure('Logics', success=False,
                      extra=['-DCKBB_BUILD_TESTS=ON', '-DCKBB_TEST_MODULES=physics_RT']))
        self.assertIn('which NEMO_MODULES excludes', self.configure('ImageReader', build='reader', success=False,
                      extra=['-DCKPLUGINS_BUILD_TESTS=ON', '-DCKPLUGINS_TEST_MODULES=AVIReader']))

    def test_newly_selected_module_inherits_tests_on_reconfiguration(self):
        self.configure('NONE', extra=['-DNEMO_BUILD_TESTS=ON'])
        reused = self.configure('ImageReader', extra=['-DNEMO_BUILD_TESTS=ON'])
        fresh = self.configure('ImageReader', build='fresh', extra=['-DNEMO_BUILD_TESTS=ON'])
        self.assertEqual(reused['plugins_tests'], 'ON')
        self.assertEqual(reused['plugins_tests'], fresh['plugins_tests'])

    def test_test_dependencies_do_not_persist_after_tests_are_disabled(self):
        self.configure('TT_InterfaceManager_RT', extra=['-DCKBB_BUILD_TESTS=ON', '-DCK2_BUILD_STATIC=OFF'])
        reused = self.configure('Logics', extra=['-DCKBB_BUILD_TESTS=OFF'])
        fresh = self.configure('Logics', build='fresh', extra=['-DCKBB_BUILD_TESTS=OFF'])
        self.assertEqual(reused['ck2_static'], 'OFF')
        self.assertEqual(reused['ck2_static'], fresh['ck2_static'])

    def test_runtime_dependencies_are_resolved_without_enabling_tests(self):
        result = self.configure('TT_ParticleSystems_RT;Sounds')
        self.assertIn('CK2_3D', result['targets'].split(';'))
        self.assertIn('SdlSoundManager', result['targets'].split(';'))
        self.assertNotIn('CKSdlGpuRasterizer', result['targets'].split(';'))
        result = self.configure('TT_Toolbox_RT')
        self.assertEqual(set(result['targets'].split(';')),
                         {'VxMath', 'CK2', 'TT_Toolbox_RT', 'CK2_3D',
                          'SdlInputManager', 'SdlSoundManager'})

    def test_test_libraries_do_not_change_runtime_composition(self):
        runtime = self.configure('physics_RT')
        tested = self.configure('physics_RT', extra=['-DCKBB_BUILD_TESTS=ON'])
        self.assertEqual(runtime['targets'], tested['targets'])
        self.assertEqual(runtime['libraries'], tested['libraries'])
        self.assertIn('CK2_3DStatic', tested['test_only'].split(';'))
        self.assertNotIn('CK2_3DStatic', tested['libraries'].split(';'))

    def test_legacy_granular_selection_limits_test_dependencies(self):
        result = self.configure('AUTO', extra=['-DCKBB_BUILD_ALL_MODULES=OFF',
                                '-DCKBB_BUILD_Logics=ON', '-DCKBB_BUILD_TESTS=ON'])
        self.assertEqual(result['ck2_static'], 'OFF')
        self.assertEqual(result['renderer_static'], 'OFF')

    def test_test_only_renderer_is_not_a_runtime_module(self):
        overrides = self.source / 'overrides'
        overrides.mkdir()
        (overrides / 'NeMo2TestDependencies.cmake').write_text(
            'function(nemo_bb_test_dependencies result)\n'
            'set(${result} renderengine PARENT_SCOPE)\nendfunction()\n')
        project = self.source / 'CMakeLists.txt'
        project.write_text(project.read_text().replace('include(NeMo2Options)',
            'list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/overrides")\ninclude(NeMo2Options)'))
        result = self.configure('Logics', extra=['-DCKBB_BUILD_TESTS=ON'])
        self.assertEqual(set(result['targets'].split(';')), {'VxMath', 'CK2', 'Logics'})
        self.assertIn('CK2_3DStatic', result['test_only'].split(';'))
        self.assertEqual(result['renderer_install'], 'OFF')

    def test_explicit_test_overrides_survive_aggregate_defaults(self):
        result = self.configure('NONE', extra=['-DVXMATH_BUILD_TESTS=ON', '-DNEMO_BUILD_TESTS=OFF'])
        self.assertIn('VxMathStatic', result['test_only'].split(';'))
        result = self.configure('ImageReader', extra=['-DNEMO_BUILD_TESTS=ON', '-DCKPLUGINS_BUILD_TESTS=OFF'])
        self.assertEqual(result['plugins_tests'], 'OFF')

    def test_auto_rejects_missing_runtime_dependencies(self):
        error = self.configure('AUTO', success=False,
            extra=['-DCKRE_BUILD_SHARED=OFF', '-DCKRE_BUILD_STATIC=OFF'])
        self.assertIn('requires runtime module', error)

    def test_dependency_closure_is_transitive_and_rejects_cycles(self):
        script = self.root / 'dependencies.cmake'
        script.write_text(f'''cmake_minimum_required(VERSION 3.25)
include("{ENGINE.as_posix()}/cmake/NeMo2Modules.cmake")
set(_nemo_runtime_dependencies [[{{"Logics":["Sounds"],"Sounds":["SdlSoundManager"]}}]])
nemo_resolve_modules("Logics;Sounds" resolved)
if (NOT resolved STREQUAL "VxMath;CK2;SdlSoundManager;Sounds;Logics")
    message(FATAL_ERROR "Incorrect recursive closure: ${{resolved}}")
endif()
set(_nemo_runtime_dependencies [[{{"Logics":["Sounds"],"Sounds":["Logics"]}}]])
nemo_resolve_modules("Logics" resolved)
''')
        result = subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Cyclic NeMo2 module dependency', result.stderr)


if __name__ == '__main__':
    unittest.main()
