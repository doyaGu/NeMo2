# NeMo2

NeMo2 is a C++ reimplementation of the Virtools runtime. It includes the behavior
engine, math library, SDL_GPU rasterizer, input and sound managers, file readers,
and building blocks.

## Build

Requirements: CMake 3.25+, SDL3 3.4.20+, and a C/C++ toolchain. Linux and macOS
presets use Ninja; Windows presets use Visual Studio.

```bash
git clone --recurse-submodules https://github.com/doyaGu/NeMo2.git
cd NeMo2
cmake --preset linux-x64-runtime
cmake --build --preset linux-x64-runtime-release
```

Choose the preset for your platform and architecture: `windows-x86`,
`windows-x64`, `windows-arm64`, `linux-x64`, `linux-arm64`, `macos-x64`, or
`macos-arm64`, followed by `-runtime` or `-static`. Set `CMAKE_PREFIX_PATH` when
SDL3 is outside the default package search path.

After updating the repository, initialize its nested dependencies:

```bash
git submodule update --init --recursive
```

See [BUILD.md](BUILD.md) for the complete preset matrix, dependency setup,
staging, and incremental builds.

## Modules

| Path | Repository | Purpose |
|------|------------|---------|
| `Source/VxMath` | [VxMath](https://github.com/doyaGu/VxMath) | Math, platform utilities, configuration, fonts |
| `Source/CK2` | [CK2](https://github.com/doyaGu/CK2) | Behavior engine and serialization |
| `Source/RenderEngine` | [CKRenderEngine](https://github.com/doyaGu/CKRenderEngine) | Rendering and SDL_GPU rasterizer |
| `Source/Managers/ParameterOperations` | [CKParameterOperations](https://github.com/doyaGu/CKParameterOperations) | Parameter operations |
| `Source/Managers/SdlInputManager` | [SdlInputManager](https://github.com/doyaGu/SdlInputManager) | Input |
| `Source/Managers/SdlSoundManager` | [SdlSoundManager](https://github.com/doyaGu/SdlSoundManager) | Sound |
| `Source/Plugins` | [CKPlugins](https://github.com/doyaGu/CKPlugins) | File readers |
| `Source/BuildingBlocks` | [CKBuildingBlocks](https://github.com/doyaGu/CKBuildingBlocks) | Behaviors, TT modules, physics |

## Application integration

Add NeMo2 to a CMake project and link the module targets your application uses:

```cmake
add_subdirectory(NeMo2)
target_link_libraries(MyApplication PRIVATE CK2 VxMath)
```

`NEMO_BUILD_STATIC` selects static libraries and modules; `NEMO_BUILD_TESTS`
enables the engine test suite. Both default to `OFF`. Module-specific options
remain available. Applications can use the `NEMO_RUNTIME_COMPONENT_MANIFEST`
global property to obtain the configured runtime module list.

Applications select their configuration namespace through
`VxSetUserConfigProfile` in `VxUserConfig.h`. The default namespace is `NeMo2`,
with `NEMO_USER_CONFIG_HOME` as an optional parent-directory override. See
[application configuration](BUILD.md#application-configuration) for startup and
TT integration.

## Tests and rendering tools

```bash
cmake --preset linux-x64-tests
cmake --build --preset linux-x64-tests-release
ctest --preset linux-x64-tests-release --output-on-failure
```

CI runs the changed modules' suites across the supported platform and linkage
combinations. Documentation and license changes skip compilation and tests. The Test workflow's
`gpu-debug` input captures rendering evidence for the GPU replay workflow.

[CONTEXT.md](CONTEXT.md) defines engine terms;
[docs/RENDERING.md](docs/RENDERING.md) covers rendering implementation.

## License

NeMo2 is licensed under [Apache 2.0](LICENSE). Each module provides its own license.
