# NeMo2

An open-source reimplementation of the Virtools game engine.

## Overview

Virtools was a commercial 3D game development environment and engine, widely used from the late 1990s to early 2010s for creating interactive 3D applications, games, and multimedia content. It was known for its visual programming approach using "building blocks" (behaviors).

NeMo2 provides a reimplementation of the core Virtools libraries (CK2 and VxMath), enabling developers to:

- Load and save Virtools file formats (.cmo, .nmo)
- Work with Virtools objects, behaviors, and scenes
- Access the Virtools math library and utility functions
- Build applications that work with Virtools content

## Features

### VxMath Library

- **Math Types:** Vectors (2D/3D/4D), Matrices, Quaternions, Planes, Rays, Spheres, OBB, Bounding Boxes, Frustums
- **Geometry:** Intersection tests, distance calculations, frustum culling
- **Image Processing:** Blit operations, mipmaps, resizing, normal/bump map conversion, pixel format conversions
- **Containers:** XArray, XString, XHashTable, XSHashTable, XNHashTable, XList, XBitArray
- **Utilities:** Memory pools, memory-mapped files, shared libraries, path parsing, directory parsing
- **System:** Threads, mutexes, time profiling, processor detection
- **Configuration:** VxConfiguration for settings management

### CK2 Library

- **Core Engine:**

  - CKContext: Main interface for creating objects, managing scenes, loading/saving files
  - Object management with unique IDs
  - Class hierarchy system

- **Managers:**

  - CKParameterManager, CKTimeManager, CKMessageManager
  - CKBehaviorManager, CKAttributeManager, CKPluginManager
  - CKPathManager, CKRenderManager, CKSoundManager
  - CKInputManager, CKCollisionManager

- **Objects:**

  - CK3dEntity, CKCamera, CKLight (3D objects)
  - CK2dEntity, CKSprite (2D objects)
  - CKMesh, CKMaterial, CKTexture (Rendering assets)
  - CKSound, CKWaveSound, CKMidiSound (Audio)
  - CKBehavior (Visual scripting behaviors)
  - CKScene, CKLevel, CKPlace, CKGroup (Scene organization)
  - CKCharacter, CKAnimation (Character animation)

- **File I/O:**
  - Load/Save .cmo, .nmo files
  - CKStateChunk for serialization
  - Data compression support

## Requirements

- **Platform:** Windows only
- **CMake:** Version 3.12 or higher
- **C++ Standard:** C++17
- **Compiler:** MSVC

## Building

### Clone the Repository

```bash
git clone --recursive https://github.com/doyaGu/NeMo2.git
cd NeMo2
```

If you already cloned without `--recursive`, initialize submodules:

```bash
git submodule update --init --recursive
```

### Build

```bash
# Configure (32-bit build for Virtools compatibility)
cmake -A Win32 -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Install (optional)
cmake --install build --prefix install
```

### Build Options

- `NEMO_BUILD_TESTS` - Build the test suite (default: ON)
- `CMAKE_BUILD_TYPE` - Debug/Release/MinSizeRel/RelWithDebInfo

### Output

- **VxMath.dll** - Math library
- **CK2.dll** - Core engine library

## Usage

### CMake Integration

```cmake
find_package(NeMo2 REQUIRED)
target_link_libraries(your_target NeMo2::CK2 NeMo2::VxMath)
```

### Basic API Example

```cpp
#include "CKAll.h"

int main() {
    // Initialize the engine
    CKStartUp();

    // Create a context
    CKContext *context;
    CKCreateContext(&context, windowHandle, 0, 0);

    // Load a Virtools file
    CKObjectArray *objects = CreateCKObjectArray();
    context->Load("game.cmo", objects, CK_LOAD_DEFAULT);

    // Process the engine
    context->Play();
    while (running) {
        context->Process();
    }

    // Cleanup
    context->ClearAll();
    CKCloseContext(context);
    CKShutdown();

    return 0;
}
```

## Testing

The project includes a comprehensive test suite using GoogleTest.

```bash
cd build
ctest --config Release
```

## Dependencies

Third-party dependencies are included as git submodules:

- **miniz** - Compression library (MIT license)
- **stb** - Image loading/processing (public domain)
- **yyjson** - JSON parsing (MIT license)

Test dependencies (fetched automatically via CMake FetchContent):

- **GoogleTest** - v1.17.0

## Project Structure

```
NeMo2/
├── CMakeLists.txt          # Main CMake build configuration
├── LICENSE                 # Apache License 2.0
├── cmake/                  # CMake configuration files
├── include/                # Public headers
├── src/                    # Source files
├── deps/                   # Third-party dependencies
└── tests/                  # Test suite
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
