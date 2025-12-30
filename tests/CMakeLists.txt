# =============================================================================
# Test Configuration
# =============================================================================

include(CTest)
enable_testing()

# =============================================================================
# GoogleTest
# =============================================================================

include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_disable_pthreads OFF CACHE BOOL "" FORCE)

if(NOT TARGET gtest_main)
    FetchContent_MakeAvailable(googletest)
endif()

set_target_properties(gtest PROPERTIES FOLDER "Dependencies/GoogleTest")
set_target_properties(gtest_main PROPERTIES FOLDER "Dependencies/GoogleTest")
set_target_properties(gmock PROPERTIES FOLDER "Dependencies/GoogleTest")
set_target_properties(gmock_main PROPERTIES FOLDER "Dependencies/GoogleTest")

# =============================================================================
# Test Subdirectories
# =============================================================================

add_subdirectory(VxMath)
