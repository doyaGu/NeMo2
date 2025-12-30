cmake_minimum_required(VERSION 3.16)

# =============================================================================
# Project Configuration
# =============================================================================

# Track whether we are the top-level project
set(NEMO_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NEMO_IS_TOP_LEVEL ON)
endif()

# Prevent in-tree builds
if(NEMO_IS_TOP_LEVEL AND CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-tree builds are not supported. Run CMake from a separate directory: cmake -B build")
endif()

project(NeMo2
    VERSION 0.1.0
    DESCRIPTION "NeMo2 - Virtools Reimplementation"
    LANGUAGES CXX C
)

# =============================================================================
# Build Options
# =============================================================================

option(NEMO_BUILD_TESTS "Build the test suite" ON)
option(NEMO_BUILD_BENCHMARKS "Build the benchmark suite" OFF)

# =============================================================================
# Global Settings
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")

if(MSVC)
    set(CMAKE_SUPPRESS_REGENERATION TRUE)
endif()

# =============================================================================
# Platform Detection
# =============================================================================

if(WIN32)
    set(NEMO_PLATFORM_WINDOWS TRUE)
    set(NEMO_PLATFORM_POSIX FALSE)
    message(STATUS "Target platform: Windows")
elseif(APPLE)
    set(NEMO_PLATFORM_WINDOWS FALSE)
    set(NEMO_PLATFORM_MACOS TRUE)
    set(NEMO_PLATFORM_POSIX TRUE)
    message(STATUS "Target platform: macOS")
elseif(UNIX)
    set(NEMO_PLATFORM_WINDOWS FALSE)
    set(NEMO_PLATFORM_LINUX TRUE)
    set(NEMO_PLATFORM_POSIX TRUE)
    message(STATUS "Target platform: Linux/Unix")
else()
    message(WARNING "Unknown platform")
endif()

# =============================================================================
# Build Type Configuration
# =============================================================================

if(NEMO_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as no build type was specified")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Debug/Release)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# =============================================================================
# Installation Configuration
# =============================================================================

include(GNUInstallDirs)

set(NEMO_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/NeMo2")

if(NEMO_IS_TOP_LEVEL AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "Installation directory" FORCE)
    message(STATUS "Setting default install directory to ${CMAKE_INSTALL_PREFIX}")
endif()

# =============================================================================
# Architecture Detection
# =============================================================================

include(TestBigEndian)
test_big_endian(NEMO_BIG_ENDIAN)
if(NEMO_BIG_ENDIAN)
    message(STATUS "Target architecture: Big Endian")
    add_compile_definitions(VX_BIG_ENDIAN=1)
else()
    message(STATUS "Target architecture: Little Endian")
    add_compile_definitions(VX_BIG_ENDIAN=0)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Target pointer size: 64-bit")
    add_compile_definitions(VX_64BIT=1)
else()
    message(STATUS "Target pointer size: 32-bit")
    add_compile_definitions(VX_64BIT=0)
endif()

# Detect x86/x64 specifically
set(NEMO_ARCH_X86 0)
set(NEMO_ARCH_X64 0)

if(MSVC AND DEFINED CMAKE_GENERATOR_PLATFORM)
    if(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
        set(NEMO_ARCH_X64 1)
    elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
        set(NEMO_ARCH_X86 1)
    endif()
endif()

if(NOT NEMO_ARCH_X86 AND NOT NEMO_ARCH_X64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        set(NEMO_ARCH_X64 1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86|i[3-6]86)$")
        set(NEMO_ARCH_X86 1)
    endif()
endif()

add_compile_definitions(
    VX_ARCH_X86=${NEMO_ARCH_X86}
    VX_ARCH_X64=${NEMO_ARCH_X64}
)

add_compile_definitions(
    VX_PLATFORM_WINDOWS=$<IF:$<PLATFORM_ID:Windows>,1,0>
    VX_PLATFORM_POSIX=$<IF:$<OR:$<PLATFORM_ID:Linux>,$<PLATFORM_ID:Darwin>>,1,0>
    VX_COMPILER_MSVC=$<IF:$<OR:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,1,0>
    VX_COMPILER_GCC=$<IF:$<OR:$<C_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:GNU>>,1,0>
    VX_COMPILER_CLANG=$<IF:$<OR:$<C_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:AppleClang>>,1,0>
)

# =============================================================================
# Compiler Settings
# =============================================================================

if(NEMO_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/Zc:strictStrings->>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-fpermissive>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-Wno-write-strings>>
)

add_compile_definitions(
    $<$<C_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
    $<$<C_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_WARNINGS>
)

if(NEMO_PLATFORM_POSIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# =============================================================================
# CMake Modules and Directories
# =============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(NEMO_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(NEMO_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

# =============================================================================
# Subdirectories
# =============================================================================

add_subdirectory(deps)
add_subdirectory(src)

if(NEMO_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(NEMO_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# =============================================================================
# Package Configuration
# =============================================================================

include(CMakePackageConfigHelpers)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeMo2Config.cmake.in")
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeMo2Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/NeMo2Config.cmake"
        INSTALL_DESTINATION "${NEMO_INSTALL_CMAKEDIR}"
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/NeMo2Config.cmake"
        DESTINATION "${NEMO_INSTALL_CMAKEDIR}"
    )
endif()

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NeMo2ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/NeMo2ConfigVersion.cmake"
    DESTINATION "${NEMO_INSTALL_CMAKEDIR}"
)

# =============================================================================
# Build Summary
# =============================================================================

if(WIN32)
    set(NEMO_PLATFORM_NAME "Windows")
elseif(APPLE)
    set(NEMO_PLATFORM_NAME "macOS")
elseif(UNIX)
    set(NEMO_PLATFORM_NAME "Linux/Unix")
else()
    set(NEMO_PLATFORM_NAME "Unknown")
endif()

if(CMAKE_CONFIGURATION_TYPES)
    set(NEMO_BUILD_TYPE_LABEL "${CMAKE_CONFIGURATION_TYPES}")
else()
    set(NEMO_BUILD_TYPE_LABEL "${CMAKE_BUILD_TYPE}")
endif()

message(STATUS "")
message(STATUS "========== NeMo2 Build Configuration ==========")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Platform:          ${NEMO_PLATFORM_NAME}")
message(STATUS "Build Type:        ${NEMO_BUILD_TYPE_LABEL}")
message(STATUS "Install Prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build Tests:       ${NEMO_BUILD_TESTS}")
message(STATUS "C++ Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "================================================")
message(STATUS "")
