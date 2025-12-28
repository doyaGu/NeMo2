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

add_subdirectory(VxMath)

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
