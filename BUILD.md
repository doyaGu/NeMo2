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

CI helper changes run only their own Python checks. Build-script changes select
aggregator checks; README and license changes run neither compilation nor tests.

```bash
gh workflow run test.yml --ref main -f modules=vxmath -f target=sdl-linux-x64-runtime
```

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
`NEMO_RUNTIME_TARGETS` combines core and module declarations. Check `TARGET`
before depending on optional modules, as the supported declarations also cover
platform-dependent and explicitly disabled modules.

NeMo2 does not change a parent's build type, installation prefix, or language
standard, and creates its standalone `stage` target only at the top level.
Applications requiring the shared project/compiler defaults can include the
`NeMo2Cache`, `NeMo2Project`, `NeMo2Compiler`, and `NeMo2LinkCache` helpers before
adding application and engine subdirectories to use the same MSVC runtime
throughout the project.

Modules keep their own versions and version-override options. Installed modules
provide individual CMake package configurations. Use `add_subdirectory` to
integrate the complete engine into an application build.
