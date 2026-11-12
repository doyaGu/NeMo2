# NeMo2

Virtools engine reimplementation using SDL3. NeMo2 assembles VxMath, CK2,
the render engine, managers, file readers, and building blocks from their
independently versioned Git repositories. It builds without BallancePlayer or
game assets.

```bash
git clone --recurse-submodules https://github.com/doyaGu/NeMo2.git
cd NeMo2
cmake --preset macos-arm64-runtime
cmake --build --preset macos-arm64-runtime-release
ctest --preset macos-arm64-runtime-release
```

Use the corresponding Windows or Linux preset for your platform. CMake 3.25+
and SDL3 3.4.20+ are required. See [BUILD.md](BUILD.md) for the platform matrix,
static builds, offline dependencies, and integration with an application.

## Repositories

| Path | Repository | Purpose |
|------|------------|---------|
| `Source/VxMath` | [VxMath](https://github.com/doyaGu/VxMath) | Math, platform utilities, configuration, fonts |
| `Source/CK2` | [CK2](https://github.com/doyaGu/CK2) | Virtools behavior engine and serialization |
| `Source/RenderEngine` | [CKRenderEngine](https://github.com/doyaGu/CKRenderEngine) | Rendering and SDL_GPU rasterizer |
| `Source/Managers/ParameterOperations` | [CKParameterOperations](https://github.com/doyaGu/CKParameterOperations) | Parameter operations |
| `Source/Managers/SdlInputManager` | [SdlInputManager](https://github.com/doyaGu/SdlInputManager) | Input |
| `Source/Managers/SdlSoundManager` | [SdlSoundManager](https://github.com/doyaGu/SdlSoundManager) | Sound |
| `Source/Plugins` | [CKPlugins](https://github.com/doyaGu/CKPlugins) | File readers |
| `Source/BuildingBlocks` | [CKBuildingBlocks](https://github.com/doyaGu/CKBuildingBlocks) | Behaviors, TT modules, physics |

`git submodule update --init --recursive` checks out the recorded revisions.
Module histories, dependencies, C++ interfaces, and target names are retained.

## Applications and compatibility

[Ballanced](https://github.com/doyaGu/Ballanced) supplies BallancePlayer, original
game asset staging, Player acceptance tests, and game release packages. It
consumes NeMo2 through `add_subdirectory` and pins an engine commit.

The first split preserves existing configuration behavior, including the
Ballance configuration directory, `BALLANCE_USER_CONFIG_HOME`, and TT defaults.
It does not introduce a generic player host. The previous NeMo2 implementation
remains available at `legacy/pre-ballanced-split`; its source layout and unified
package interface have been replaced by the module repositories.

## Tests and rendering tools

`NEMO_BUILD_TESTS=ON` enables the complete engine suite. The Test workflow covers
the supported platform/linkage matrix and a full Linux x64 configuration. Its
`gpu-debug` input captures rendering evidence for the GPU replay workflow.
Player JIT scripts and the Ballance level route live in Ballanced.

[CONTEXT.md](CONTEXT.md) defines engine terms;
[docs/RENDERING.md](docs/RENDERING.md) records rendering implementation guidance.

## Licenses

NeMo2 retains its [Apache 2.0 license](LICENSE). Submodules retain their own
licenses. Build orchestration and CI helpers migrated from Ballanced retain
the [Ballanced MIT notice](LICENSES/Ballanced-MIT.txt).
