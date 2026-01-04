# =============================================================================
# Test Configuration
# =============================================================================

# =============================================================================
# GoogleTest
# =============================================================================

include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE
        FIND_PACKAGE_ARGS NAMES GTest
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_disable_pthreads OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES
        FOLDER "Dependencies/GoogleTest"
)

# =============================================================================
# Test Subdirectories
# =============================================================================

add_subdirectory(VxMath)
add_subdirectory(CK2)
