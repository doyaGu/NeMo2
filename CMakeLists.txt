cmake_minimum_required(VERSION 3.25)

project(NeMo2 VERSION 0.1.0
        DESCRIPTION "Virtools engine reimplementation"
        HOMEPAGE_URL https://github.com/doyaGu/NeMo2
        LANGUAGES C CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)
include(NeMo2Cache)
include(NeMo2Options)

if (PROJECT_IS_TOP_LEVEL)
    include(NeMo2Project)
    include(NeMo2Compiler)
    include(NeMo2LinkCache)
endif ()

include(CTest)
add_subdirectory(Source)
include(NeMo2TestTargets)
include(NeMo2ComponentRegistry)
if (PROJECT_IS_TOP_LEVEL)
    include(NeMo2Stage)
endif ()

if (BUILD_TESTING)
    add_test(NAME NeMo2RuntimeManifest
            COMMAND "${CMAKE_COMMAND}"
            "-DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/Testing"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestRuntimeComponentSelection.cmake")
    if (VXMATH_FONT_DIR)
        get_filename_component(_nemo_font_directory "${VXMATH_FONT_DIR}" ABSOLUTE)
    else ()
        set(_nemo_font_directory "${CMAKE_BINARY_DIR}/_deps/vxmath-fonts")
    endif ()
    add_test(NAME NumericalIsolation
            COMMAND "${CMAKE_COMMAND}"
            "-DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestNumericalIsolation.cmake")
    add_test(NAME NumericalIsolationRules
            COMMAND "${CMAKE_COMMAND}"
            "-DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/Testing"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestNumericalIsolationRules.cmake")
    add_test(NAME NeMo2Subproject
            COMMAND "${CMAKE_COMMAND}"
            "-DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/Testing"
            "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DTEST_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DTEST_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DTEST_SDL3_DIR=${SDL3_DIR}"
            "-DTEST_FONT_DIR=${_nemo_font_directory}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestSubproject.cmake")
    add_test(NAME NeMo2StageInstall
            COMMAND "${CMAKE_COMMAND}"
            "-DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/Testing"
            "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DTEST_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DTEST_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestStageInstall.cmake")
    if (PROJECT_IS_TOP_LEVEL)
        add_test(NAME CMakeRegeneration
                COMMAND "${CMAKE_COMMAND}"
                "-DTEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                "-DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
                "-DTEST_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
                "-DTEST_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
                "-DTEST_CONFIG=$<CONFIG>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestBuildRegeneration.cmake")
    endif ()
endif ()
