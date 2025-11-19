include(CTest)
enable_testing()

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
if (WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif ()

# Disable installation of GoogleTest
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_disable_pthreads OFF CACHE BOOL "" FORCE)

# Only fetch if not already available (handle subproject case)
if (NOT TARGET gtest_main)
    FetchContent_MakeAvailable(googletest)
endif ()

# Function to add VxMath tests
function(vx_add_test TEST_NAME)
    cmake_parse_arguments(TEST "" "" "SOURCES;DEPENDENCIES" ${ARGN})

    if (NOT TEST_SOURCES)
        message(FATAL_ERROR "vx_add_test: SOURCES argument is required")
    endif ()

    add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_compile_definitions(${TEST_NAME} PRIVATE VX_TESTING)
    set_target_properties(${TEST_NAME} PROPERTIES
            FOLDER "Tests"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_include_directories(${TEST_NAME} PRIVATE
            ${NEMO_INCLUDE_DIR}
            ${NEMO_INCLUDE_DIR}
    )

    target_link_libraries(${TEST_NAME} PRIVATE
            ${TEST_TARGET}
            gtest_main
            gmock
            ${TEST_DEPENDENCIES}
    )

    # Register with CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

set(VX_TEST_SOURCES
        SystemInfoTest.cpp
        ImageTest.cpp
        PathParsingTest.cpp
        StringTest.cpp
        GeometryTest.cpp
        ArrayTest.cpp
        VectorTest.cpp
        QuaternionTest.cpp
        MatrixTest.cpp
        FrustumTest.cpp
        PlaneTest.cpp
        RayTest.cpp
        SphereTest.cpp
        URLTest.cpp
        WindowTest.cpp
        ConfigurationTest.cpp
        MemoryMappedFileTest.cpp
        TimeProfilerTest.cpp
        IntersectionTest.cpp
)

foreach (TEST_SOURCE ${VX_TEST_SOURCES})
    # Extract test name from source file
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    vx_add_test(${TEST_NAME} SOURCES ${TEST_SOURCE} DEPENDENCIES VxMath)
endforeach ()

function(add_ck2_test TEST_NAME)
    cmake_parse_arguments(TEST "" "" "SOURCES;DEPENDENCIES" ${ARGN})

    add_executable(${TEST_NAME} ${TEST_SOURCES})
    set_target_properties(${TEST_NAME} PROPERTIES
            FOLDER "Tests"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_compile_definitions(${TEST_NAME} PRIVATE "CK2_EXPORTS")

    target_include_directories(${TEST_NAME} PRIVATE
            ${NEMO_INCLUDE_DIR}
            ${NEMO_INCLUDE_DIR}
    )

    target_link_libraries(${TEST_NAME} PRIVATE
            gtest_main
            gmock
            ${TEST_DEPENDENCIES}
    )
endfunction()

add_ck2_test(PlayerTest
        SOURCES
        PlayerTest.cpp
        Player.cpp
        DEPENDENCIES
        CK2 VxMath
)

add_executable(Player Player.cpp)

set_target_properties(Player PROPERTIES
        WIN32_EXECUTABLE TRUE
)

target_compile_definitions(Player PRIVATE
        PLAYER_STANDALONE
)

target_include_directories(Player PRIVATE
        ${NEMO_INCLUDE_DIR}
        ${NEMO_SOURCE_DIR}
)

target_link_libraries(Player PRIVATE
        CK2
        VxMath
)
