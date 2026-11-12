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
adding their application and engine subdirectories. Ballanced uses these helpers
to keep Player and the engine on the same MSVC runtime.

Modules keep their own versions and existing version-override options. Use their
individual package configurations when consuming an installed module; this
aggregator does not provide the former unified `find_package(NeMo2)` interface.
