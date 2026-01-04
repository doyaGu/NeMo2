include_guard(GLOBAL)

# =============================================================================
# NeMo2 CMake Helper Functions
#
# This file is the single source of truth for CMake helper functions in NeMo2.
# It consolidates the previous NeMoHelpers.cmake and NeMo2Helpers.cmake helpers.
# =============================================================================

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

function(_nemo_set_target_output_dirs TARGET_NAME RUNTIME_SUBDIR)
    if(RUNTIME_SUBDIR STREQUAL "")
        set(_runtime_dir "${CMAKE_BINARY_DIR}/bin")
    else()
        set(_runtime_dir "${CMAKE_BINARY_DIR}/bin/${RUNTIME_SUBDIR}")
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${_runtime_dir}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
endfunction()

function(_nemo_set_target_version TARGET_NAME)
    # These are safe even if the project doesn't define VERSION, but NeMo2 does.
    set_target_properties(${TARGET_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endfunction()

function(_nemo_get_output_name DEFAULT_NAME OVERRIDE OUT_VAR)
    set(_name "${DEFAULT_NAME}")
    if(NOT "${OVERRIDE}" STREQUAL "")
        set(_name "${OVERRIDE}")
    endif()
    set(${OUT_VAR} "${_name}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# nemo_add_library
# Creates a library target with standardized settings
#
# Arguments:
#   NAME - Library name
#   TYPE - SHARED, STATIC, or INTERFACE
#   PUBLIC_HEADERS - List of public header files
#   PRIVATE_HEADERS - List of private header files
#   SOURCES - List of source files
#   PUBLIC_DEPS - Public dependencies
#   PRIVATE_DEPS - Private dependencies
#   COMPILE_DEFS - Private compile definitions
#   EXPORT_ALL - Enable WINDOWS_EXPORT_ALL_SYMBOLS on Windows (for SHARED only)
# -----------------------------------------------------------------------------
function(nemo_add_library)
    cmake_parse_arguments(NEMO_LIB
        "EXPORT_ALL"
        "NAME;TYPE"
        "PUBLIC_HEADERS;PRIVATE_HEADERS;SOURCES;PUBLIC_DEPS;PRIVATE_DEPS;COMPILE_DEFS"
        ${ARGN}
    )

    if(NOT NEMO_LIB_NAME)
        message(FATAL_ERROR "nemo_add_library: NAME is required")
    endif()

    if(NOT NEMO_LIB_TYPE)
        set(NEMO_LIB_TYPE SHARED)
    endif()

    # Create library
    if(NEMO_LIB_TYPE STREQUAL "INTERFACE")
        add_library(${NEMO_LIB_NAME} INTERFACE)
        add_library(NeMo2::${NEMO_LIB_NAME} ALIAS ${NEMO_LIB_NAME})
    else()
        add_library(${NEMO_LIB_NAME} ${NEMO_LIB_TYPE}
            ${NEMO_LIB_SOURCES}
            ${NEMO_LIB_PUBLIC_HEADERS}
            ${NEMO_LIB_PRIVATE_HEADERS}
        )
        add_library(NeMo2::${NEMO_LIB_NAME} ALIAS ${NEMO_LIB_NAME})

        set_target_properties(${NEMO_LIB_NAME} PROPERTIES
            OUTPUT_NAME "${NEMO_LIB_NAME}"
            FOLDER "Libraries"
        )
        _nemo_set_target_output_dirs(${NEMO_LIB_NAME} "")
        _nemo_set_target_version(${NEMO_LIB_NAME})

        if(NEMO_LIB_EXPORT_ALL AND NEMO_LIB_TYPE STREQUAL "SHARED" AND WIN32)
            set_target_properties(${NEMO_LIB_NAME} PROPERTIES
                WINDOWS_EXPORT_ALL_SYMBOLS ON
            )
        endif()

        # Set C++ standard
        target_compile_features(${NEMO_LIB_NAME} PUBLIC cxx_std_17)

        # Include directories
        target_include_directories(${NEMO_LIB_NAME}
            PUBLIC
                $<BUILD_INTERFACE:${NEMO_INCLUDE_DIR}>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
        )

        # Compile definitions
        if(NEMO_LIB_COMPILE_DEFS)
            target_compile_definitions(${NEMO_LIB_NAME} PRIVATE ${NEMO_LIB_COMPILE_DEFS})
        endif()
    endif()

    # Link dependencies
    if(NEMO_LIB_PUBLIC_DEPS)
        target_link_libraries(${NEMO_LIB_NAME} PUBLIC ${NEMO_LIB_PUBLIC_DEPS})
    endif()

    if(NEMO_LIB_PRIVATE_DEPS)
        target_link_libraries(${NEMO_LIB_NAME} PRIVATE ${NEMO_LIB_PRIVATE_DEPS})
    endif()
endfunction()

# -----------------------------------------------------------------------------
# nemo_install_library
# Installs a library target with headers
#
# Arguments:
#   TARGET - Target name
#   HEADERS - List of public header files to install
# -----------------------------------------------------------------------------
function(nemo_install_library)
    cmake_parse_arguments(NEMO_INST
        ""
        "TARGET"
        "HEADERS"
        ${ARGN}
    )

    if(NOT NEMO_INST_TARGET)
        message(FATAL_ERROR "nemo_install_library: TARGET is required")
    endif()

    install(TARGETS ${NEMO_INST_TARGET}
        EXPORT NeMo2Targets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    if(NEMO_INST_HEADERS)
        install(FILES ${NEMO_INST_HEADERS}
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# nemo_add_test
# Creates a test executable with standardized settings
#
# Arguments:
#   NAME - Test name
#   SOURCES - List of source files
#   DEPS - Dependencies
# -----------------------------------------------------------------------------
function(nemo_add_test)
    cmake_parse_arguments(NEMO_TEST
        ""
        "NAME"
        "SOURCES;DEPS"
        ${ARGN}
    )

    if(NOT NEMO_TEST_NAME)
        message(FATAL_ERROR "nemo_add_test: NAME is required")
    endif()

    if(NOT NEMO_TEST_SOURCES)
        message(FATAL_ERROR "nemo_add_test: SOURCES is required")
    endif()

    add_executable(${NEMO_TEST_NAME} ${NEMO_TEST_SOURCES})

    target_compile_definitions(${NEMO_TEST_NAME} PRIVATE VX_TESTING)
    set_target_properties(${NEMO_TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_link_libraries(${NEMO_TEST_NAME} PRIVATE
        ${NEMO_TEST_DEPS}
        GTest::gtest_main
        GTest::gmock
    )

    add_test(NAME ${NEMO_TEST_NAME} COMMAND ${NEMO_TEST_NAME})
endfunction()

# -----------------------------------------------------------------------------
# nemo_add_benchmark
# Creates a benchmark executable with standardized settings
#
# Arguments:
#   NAME - Benchmark name
#   SOURCES - List of source files
#   DEPS - Dependencies
# -----------------------------------------------------------------------------
function(nemo_add_benchmark)
    cmake_parse_arguments(NEMO_BENCH
        ""
        "NAME"
        "SOURCES;DEPS"
        ${ARGN}
    )

    if(NOT NEMO_BENCH_NAME)
        message(FATAL_ERROR "nemo_add_benchmark: NAME is required")
    endif()

    if(NOT NEMO_BENCH_SOURCES)
        message(FATAL_ERROR "nemo_add_benchmark: SOURCES is required")
    endif()

    add_executable(${NEMO_BENCH_NAME} ${NEMO_BENCH_SOURCES})

    target_compile_definitions(${NEMO_BENCH_NAME} PRIVATE VX_BENCHMARKING)
    set_target_properties(${NEMO_BENCH_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_link_libraries(${NEMO_BENCH_NAME} PRIVATE
        ${NEMO_BENCH_DEPS}
        benchmark::benchmark
        benchmark::benchmark_main
    )

    set_property(GLOBAL APPEND PROPERTY NEMO_BENCHMARK_TARGETS ${NEMO_BENCH_NAME})
endfunction()

# -----------------------------------------------------------------------------
# nemo_add_building_block(<name>
#     SOURCES <source_files>...
#     [OUTPUT_NAME <output_name>]
#     [EXTRA_INCLUDE_DIRS <include_dirs>...]
#     [EXTRA_LINK_LIBS <link_libs>...]
#     [EXTRA_DEFINITIONS <definitions>...]
#     [NO_BEHAVIORS]
# )
#
# Creates a building block shared library with standard configuration.
# -----------------------------------------------------------------------------
function(nemo_add_building_block NAME)
    cmake_parse_arguments(BB
        "NO_BEHAVIORS"
        "OUTPUT_NAME"
        "SOURCES;EXTRA_INCLUDE_DIRS;EXTRA_LINK_LIBS;EXTRA_DEFINITIONS"
        ${ARGN}
    )

    # Create the shared library
    add_library(${NAME} SHARED ${BB_SOURCES})
    add_library(NeMo2::${NAME} ALIAS ${NAME})

    # Add behavior sources unless NO_BEHAVIORS is specified
    if(NOT BB_NO_BEHAVIORS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Behaviors")
        aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/Behaviors ${NAME}_BEHAVIORS_SRCS)
        target_sources(${NAME} PRIVATE ${${NAME}_BEHAVIORS_SRCS})
    endif()

    _nemo_get_output_name(${NAME} "${BB_OUTPUT_NAME}" _output_name)
    set_target_properties(${NAME} PROPERTIES
        OUTPUT_NAME "${_output_name}"
        FOLDER "BuildingBlocks"
    )
    _nemo_set_target_output_dirs(${NAME} "BuildingBlocks")
    _nemo_set_target_version(${NAME})

    # C++17 standard
    target_compile_features(${NAME} PRIVATE cxx_std_17)

    # Include directories
    target_include_directories(${NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(BB_EXTRA_INCLUDE_DIRS)
        target_include_directories(${NAME} PRIVATE ${BB_EXTRA_INCLUDE_DIRS})
    endif()

    # Standard dependencies
    target_link_libraries(${NAME} PRIVATE
        NeMo2::CK2
        NeMo2::VxMath
        NeMo2::CompileOptions
    )
    if(BB_EXTRA_LINK_LIBS)
        target_link_libraries(${NAME} PRIVATE ${BB_EXTRA_LINK_LIBS})
    endif()

    # Extra definitions
    if(BB_EXTRA_DEFINITIONS)
        target_compile_definitions(${NAME} PRIVATE ${BB_EXTRA_DEFINITIONS})
    endif()

    # Installation
    install(TARGETS ${NAME}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/BuildingBlocks"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endfunction()

# -----------------------------------------------------------------------------
# nemo_add_plugin(<name>
#     SOURCES <source_files>...
#     [OUTPUT_NAME <output_name>]
#     [CXX_STANDARD <standard>]
#     [EXTRA_INCLUDE_DIRS <include_dirs>...]
#     [EXTRA_LINK_LIBS <link_libs>...]
#     [EXTRA_DEFINITIONS <definitions>...]
# )
#
# Creates a plugin shared library with standard configuration.
# -----------------------------------------------------------------------------
function(nemo_add_plugin NAME)
    cmake_parse_arguments(PLUGIN
        ""
        "OUTPUT_NAME;CXX_STANDARD"
        "SOURCES;EXTRA_INCLUDE_DIRS;EXTRA_LINK_LIBS;EXTRA_DEFINITIONS"
        ${ARGN}
    )

    # Create the shared library
    add_library(${NAME} SHARED ${PLUGIN_SOURCES})
    add_library(NeMo2::${NAME} ALIAS ${NAME})

    _nemo_get_output_name(${NAME} "${PLUGIN_OUTPUT_NAME}" _output_name)
    set_target_properties(${NAME} PROPERTIES
        OUTPUT_NAME "${_output_name}"
        FOLDER "Plugins"
    )
    _nemo_set_target_output_dirs(${NAME} "Plugins")
    _nemo_set_target_version(${NAME})

    # C++ standard (default to C++98 for plugins for compatibility)
    set(_cxx_standard 98)
    if(PLUGIN_CXX_STANDARD)
        set(_cxx_standard ${PLUGIN_CXX_STANDARD})
    endif()
    target_compile_features(${NAME} PRIVATE cxx_std_${_cxx_standard})

    # Include directories
    target_include_directories(${NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(PLUGIN_EXTRA_INCLUDE_DIRS)
        target_include_directories(${NAME} PRIVATE ${PLUGIN_EXTRA_INCLUDE_DIRS})
    endif()

    # Standard dependencies
    target_link_libraries(${NAME} PRIVATE
        NeMo2::CK2
        NeMo2::VxMath
        NeMo2::CompileOptions
    )
    if(PLUGIN_EXTRA_LINK_LIBS)
        target_link_libraries(${NAME} PRIVATE ${PLUGIN_EXTRA_LINK_LIBS})
    endif()

    # Standard MSVC definitions for plugins
    target_compile_definitions(${NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_WARNINGS>
    )
    if(PLUGIN_EXTRA_DEFINITIONS)
        target_compile_definitions(${NAME} PRIVATE ${PLUGIN_EXTRA_DEFINITIONS})
    endif()

    # Installation
    install(TARGETS ${NAME}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/Plugins"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endfunction()

# -----------------------------------------------------------------------------
# nemo_add_manager(<name>
#     SOURCES <source_files>...
#     [OUTPUT_NAME <output_name>]
#     [CXX_STANDARD <standard>]
#     [EXTRA_INCLUDE_DIRS <include_dirs>...]
#     [EXTRA_LINK_LIBS <link_libs>...]
#     [EXTRA_DEFINITIONS <definitions>...]
# )
#
# Creates a manager shared library with standard configuration.
# -----------------------------------------------------------------------------
function(nemo_add_manager NAME)
    cmake_parse_arguments(MGR
        ""
        "OUTPUT_NAME;CXX_STANDARD"
        "SOURCES;EXTRA_INCLUDE_DIRS;EXTRA_LINK_LIBS;EXTRA_DEFINITIONS"
        ${ARGN}
    )

    # Create the shared library
    add_library(${NAME} SHARED ${MGR_SOURCES})
    add_library(NeMo2::${NAME} ALIAS ${NAME})

    _nemo_get_output_name(${NAME} "${MGR_OUTPUT_NAME}" _output_name)
    set_target_properties(${NAME} PROPERTIES
        OUTPUT_NAME "${_output_name}"
        FOLDER "Managers"
    )
    _nemo_set_target_output_dirs(${NAME} "Managers")
    _nemo_set_target_version(${NAME})

    # C++ standard (default to C++17 for managers)
    set(_cxx_standard 17)
    if(MGR_CXX_STANDARD)
        set(_cxx_standard ${MGR_CXX_STANDARD})
    endif()
    target_compile_features(${NAME} PRIVATE cxx_std_${_cxx_standard})

    # Include directories
    target_include_directories(${NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(MGR_EXTRA_INCLUDE_DIRS)
        target_include_directories(${NAME} PRIVATE ${MGR_EXTRA_INCLUDE_DIRS})
    endif()

    # Standard dependencies
    target_link_libraries(${NAME} PRIVATE
        NeMo2::CK2
        NeMo2::VxMath
        NeMo2::CompileOptions
    )
    if(MGR_EXTRA_LINK_LIBS)
        target_link_libraries(${NAME} PRIVATE ${MGR_EXTRA_LINK_LIBS})
    endif()

    # Extra definitions
    if(MGR_EXTRA_DEFINITIONS)
        target_compile_definitions(${NAME} PRIVATE ${MGR_EXTRA_DEFINITIONS})
    endif()

    # Installation
    install(TARGETS ${NAME}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/Managers"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
endfunction()
