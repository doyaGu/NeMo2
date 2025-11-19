cmake_minimum_required(VERSION 3.12)

# Prevent in-tree builds
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-tree builds are not supported. Run CMake from a separate directory: cmake -B build")
endif ()

# Project definition
project(NeMo2
    VERSION 0.1.0
    DESCRIPTION "NeMo2 - Virtools Reimplementation"
    LANGUAGES CXX C
)

# ==================== Global Settings ====================
# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED NO)
set(CMAKE_CXX_EXTENSIONS YES)

# Use folders to organize targets in an IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")

# Platform check
if (NOT WIN32)
    message(FATAL_ERROR "Only Windows is supported.")
endif ()

# Use relative paths (reduce path size for command-line limits on Windows)
if (WIN32)
    set(CMAKE_USE_RELATIVE_PATHS TRUE)
    set(CMAKE_SUPPRESS_REGENERATION TRUE)
endif ()

# ==================== Build Type Configuration ====================
# Set default build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as no build type was specified")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Debug/Release)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# ==================== Installation Configuration ====================
# Set default install prefix
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "The root directory of the installation" FORCE)
    message(STATUS "Setting default install directory to ${CMAKE_INSTALL_PREFIX}")
endif ()

# ==================== Compiler Settings ====================
# Generate compile_commands.json for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Disable strict const-qualification conformance for pointers initialized by using string literals
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/Zc:strictStrings->>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-fpermissive>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-Wno-write-strings>>
)

# Disable MSVC unsafe warnings
add_compile_definitions(
    $<$<C_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
    $<$<C_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_WARNINGS>
)

# ==================== Build Options ====================
option(NEMO_BUILD_TESTS "Build the test suite" ON)

# ==================== CMake Modules ====================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ==================== Project Directories ====================
set(NEMO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(NEMO_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# ==================== Subdirectories ====================
add_subdirectory(deps)
add_subdirectory(src)

if (NEMO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

# ==================== Package Configuration ====================
include(CMakePackageConfigHelpers)

# Check if config template exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeMo2Config.cmake.in")
    # Create package config file
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeMo2Config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/NeMo2Config.cmake"
        INSTALL_DESTINATION lib/cmake/NeMo2
    )

    # Install package config file
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/NeMo2Config.cmake"
        DESTINATION lib/cmake/NeMo2
    )
endif()

# Create package version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NeMo2ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install package version file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/NeMo2ConfigVersion.cmake"
    DESTINATION lib/cmake/NeMo2
)

# ==================== Summary ====================
message(STATUS "")
message(STATUS "========== NeMo2 Build Configuration ==========")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build Tests:       ${NEMO_BUILD_TESTS}")
message(STATUS "C++ Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "==============================================")
message(STATUS "")


