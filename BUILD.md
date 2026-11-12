# Building NeMo2

Install CMake 3.25+, SDL3 3.4.20+, and a C/C++ toolchain. Presets use Visual
Studio 2022 on Windows and Ninja with GCC or Clang on Linux and macOS. Configure
`CMAKE_PREFIX_PATH` if SDL3 is outside the package search path.

## Clone and configure

```bash
git clone --recurse-submodules https://github.com/doyaGu/NeMo2.git
cd NeMo2
cmake --preset linux-x64-runtime -DCMAKE_PREFIX_PATH=/path/to/SDL3
cmake --build --preset linux-x64-runtime-release
ctest --preset linux-x64-runtime-release
```

After updating, run `git submodule sync --recursive` and
`git submodule update --init --recursive`. Build from the pinned revisions;
`--remote` selects newer module commits and changes the assembled engine.

The configure matrix is `<platform>-<mode>`:

| Platform | Architectures |
|----------|---------------|
| Windows | `windows-x86`, `windows-x64`, `windows-arm64` |
| Linux | `linux-x64`, `linux-arm64` |
| macOS | `macos-x64`, `macos-arm64` |

Modes are `runtime`, `static`, and `tests`. Build presets append `-release`;
`-stage-release` builds and installs the selected libraries and modules using
their existing Runtime and Development install rules. The default destination
is `build/<configure-preset>/stage`. NeMo2 stages libraries, headers, module
configuration, fonts, and SDL3; application packages are assembled downstream.

## Options and tests

| Option | Default | Effect |
|--------|---------|--------|
| `NEMO_BUILD_STATIC` | `OFF` | Default module linkage to static |
| `NEMO_BUILD_TESTS` | `OFF` | Enable all engine module and IVP tests |
| `NEMO_TARGET_ARCH` | empty | Architecture label used in manager summaries |
| `BUILD_TESTING` | `ON` | Register aggregator checks through CTest |

Existing module options such as `CKRE_BUILD_TESTS`, `VXMATH_BUILD_STATIC`, and
`CKBB_BUILD_physics_RT` remain available. Explicit module cache values take
precedence over aggregator defaults. Engine tests select the additional static
libraries needed to access internal CK2 and render-engine functions.

```bash
cmake --preset linux-x64-tests
cmake --build --preset linux-x64-tests-release
ctest --preset linux-x64-tests-release --output-on-failure
```

The aggregator checks numerical ownership, verifies parent-project settings,
and checks CMake regeneration. Rendering tests require a working GPU or software
rasterizer. CI uses Xvfb and Mesa Vulkan on Linux, WARP on Windows, and Metal on
macOS. GPU capture and replay workflows live in this repository.

VxMath downloads and verifies its font assets when installation is enabled.
For offline builds, set `VXMATH_FONT_DIR` to a directory containing the files
listed in `Source/VxMath/fonts/sources.json`. Tests also accept a local GoogleTest
checkout through `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`.

## Application configuration

Select the application's configuration profile before opening `VxConfig`
sections or starting worker threads:

```cpp
#include "VxUserConfig.h"

// During application startup:
VxSetUserConfigProfile("SampleApplication", "SAMPLE_CONFIG_HOME");
```

The profile copies both names. `VxGetUserConfigPath(NULL, path)` resolves its
application directory, and `VxConfig` stores `registry.ini` there. The selected
environment variable overrides the parent directory; for example,
`SAMPLE_CONFIG_HOME=/tmp/config` selects `/tmp/config/SampleApplication/`.
Without that override, the platform's user configuration directory is used.
Environment values are read on each lookup. An explicit application name passed
to `VxGetUserConfigPath` takes precedence over the profile name.

The default profile uses `NeMo2` and `NEMO_USER_CONFIG_HOME`. Null or empty profile
arguments restore those defaults. Changing profiles does not move existing
configuration files. Hosts retaining an existing application identity should
select its existing name and environment variable explicitly.

TT InterfaceManager's `CGameInfo` accepts an optional registry path in its
constructor. Hosts can supply their existing serialized path; an omitted path
uses `Software\\Virtools\\Settings`. Composition-provided registry paths still
take precedence when creating game info through a behavior. To retain an
application default across behavior-created game info, set
`InterfaceManager::SetDefaultRegistryPath` on each context before loading a
composition, and pass `GetDefaultRegistryPath()` when constructing host game
info. The logical registry path is independent of the profile's filesystem
location.

Independent contexts can use explicit files without changing the process profile:

```cpp
VxConfig config(VXCONFIG_ROOT_CURRENT_USER, "Software\\Sample", registryFile);
```

The filename is copied and remains selected across `OpenRoot` calls. A null or
empty filename retains the existing profile lookup. Create the parent directory
before writing. TT hosts can select a file on each context's InterfaceManager
with `SetRegistryFile`; all its registry behaviors use that file.

Rendering hosts can call `CKSetRenderSettingsFile(context, iniFile)` from
`CKRenderConfiguration.h` after the render manager exists and before creating
render contexts. The file is parsed into context-owned storage, and a failed
load preserves the previous settings. Passing null restores the legacy file
lookup. The selected options and defaults apply to future resources; existing
resources keep their properties. Low-level aggregate diagnostic counters and
the debug output sink retain their process-wide lifetime.

## Incremental builds and CI

Keep a configured build directory when updating engine sources or application
code. CMake rebuilds the targets whose inputs changed; downstream applications
using `add_subdirectory` reuse unchanged NeMo2 objects in their own build tree.
Use a fresh directory when migrating source layouts or changing generators and
toolchains. Standalone and nested build trees contain different absolute paths
and cannot be exchanged as binary caches.

CI Build and Test share incremental caches per runner image, architecture,
linkage, SDL version, presets, and compiler/linker policy. Each source revision
gets a snapshot, with fallback to compatible earlier snapshots and pre-sharing
caches. Workflow, test-script, and documentation changes do not invalidate the
cache family. Source content hashes restore unchanged checkout timestamps;
changed inputs are marked newer than cached outputs. Configure always runs,
and stage directories and test reports are regenerated.

The nightly Build compares source changes with the previous completed nightly.
Changed submodule revisions are inspected recursively, so documentation-only
updates do not trigger builds or tests. Build calls Test at the same commit after
saving its caches. Test enables only the changed modules' test options and their
required static dependencies; all other module test options are explicitly off.
BB test dependencies are declared in `cmake/NeMo2BBTestDependencies.json`.
The TT InterfaceManager tests require only CK2's static runtime; they do not
enable or link a render-engine static library.
Building blocks and readers are selected individually: changing
`TT_InterfaceManager_RT` does not register the physics or other BB tests, and
changing `AVIReader` does not register the other readers' tests. Module-wide
selections remain available for shared build changes. Verified core test-only
edits build and run the corresponding executables rather than the entire suite.
CTest cases are selected by the source directory that registered them. Build
disables test options again when restoring a Test snapshot. GPU diagnostic builds
keep caching disabled so their diagnostic compiler settings cannot leak into release builds.

Build and Test remain independently dispatchable. Test defaults to
`modules=changed`; use `modules=vxmath`, `ck2`, `renderengine`, `plugins`,
`buildingblocks`, or `ivp` to request one module. `modules=integration` selects
aggregator checks. A complete engine test run requires an explicit
`modules=all` or the `sdl-linux-x64-full-tests` target. Neither is selected
automatically. The platform selector `target=all` covers the platform matrix
for the selected modules.

Use `modules=bb:TT_InterfaceManager_RT` or `modules=plugin:AVIReader` to request
one component; comma-separated selections are supported. The CMake filters
`CKBB_TEST_MODULES` and `CKPLUGINS_TEST_MODULES` accept semicolon-separated module
names. Their empty defaults retain the existing complete module test suites.

Build and Test use the platform definitions in `.github/scripts/ci_platforms.json`.
A specific target creates one job. Changes limited to Windows version resources
create only the three Windows dynamic-library jobs.
Manual test-module requests retain the platform matrix requested by `target`.

Test snapshots are grouped by library compilation policy, including mixed
shared/static dependencies and renderer test hooks. Components with the same
policy reuse each other's snapshots; tests matching the release policy retain
the shared Build cache. Other test configurations first restore their own warm
snapshot and only use a normal Build snapshot as a cold-start fallback. Build
continues to restore release snapshots, so warm release/test runs do not
recompile renderer objects merely to switch library definitions.

CI helper changes run only their own Python checks. Build-script changes select
aggregator checks; README and license changes run neither compilation nor tests.
Changes limited to `Version.rc.in` update Windows version resources without
enabling engine unit-test suites.

```bash
gh workflow run test.yml --ref main -f modules=vxmath -f target=sdl-linux-x64-runtime
```

For a focused procedural reference capture, select one runtime target and the
scenes to capture. This builds the capture tool from the incremental cache and
uploads candidate frames for review; it does not run test suites or update fixtures:

```bash
gh workflow run build.yml --ref main -f target=sdl-linux-x64-runtime -f capture-scenes=sprites_text,present_fxaa
```

`capture-frames` defaults to five; `capture-validation=true` enables the GPU
validation layer. Static scenes can be captured with one requested frame to
review their inputs without presenting preceding frames.

## Application integration

```cmake
set(NEMO_BUILD_STATIC ON CACHE BOOL "")
add_subdirectory(NeMo2)
target_link_libraries(MyApplication PRIVATE CK2 VxMath)

get_property(engine_manifest GLOBAL PROPERTY NEMO_RUNTIME_COMPONENT_MANIFEST)
include("${engine_manifest}")
```

The manifest path is available after `add_subdirectory` returns. It contains
`NEMO_CORE_RUNTIME_TARGETS`, `NEMO_MODULE_RUNTIME_TARGETS`, their category lists,
runtime output names, and render-engine configuration/disabled-output lists.
`NEMO_RUNTIME_TARGETS` combines the configured core and module targets. Disabled
modules are omitted, and output names come from the selected CMake targets.
`NEMO_SUPPORTED_*_TARGETS` records the separate supported module catalogue.
Applications decide which enabled modules they require. The legacy
`NEMO_BUILDING_BLOCK_REQUIRED_OUTPUTS` name now means all selected BB outputs;
`NEMO_BUILDING_BLOCK_OPTIONAL_OUTPUTS` is empty.

NeMo2 does not change a parent's build type, installation prefix, or language
standard, and creates its standalone `stage` target only at the top level.
Applications requiring the shared project/compiler defaults can include the
`NeMo2Cache`, `NeMo2Project`, `NeMo2Compiler`, and `NeMo2LinkCache` helpers before
adding application and engine subdirectories to use the same MSVC runtime
throughout the project.

Modules keep their own versions and version-override options. Installed modules
provide individual CMake package configurations. Use `add_subdirectory` to
integrate the complete engine into an application build.
