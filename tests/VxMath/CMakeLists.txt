# VxMath Tests

#=============================================================================
# Ground-truth VxMath.dll setup
#=============================================================================
# Option to use ground-truth DLL from bin/ for testing (default: OFF uses built lib)
option(VXMATH_USE_GROUND_TRUTH_DLL "Link tests against ground-truth bin/VxMath.dll" TRUE)

if (VXMATH_USE_GROUND_TRUTH_DLL AND WIN32)
    # Create an IMPORTED target for the ground-truth DLL
    set(GROUND_TRUTH_DLL_PATH "${CMAKE_SOURCE_DIR}/bin/VxMath.dll")
    set(GROUND_TRUTH_LIB_PATH "${CMAKE_SOURCE_DIR}/bin/VxMath.lib")

    if (EXISTS "${GROUND_TRUTH_DLL_PATH}" AND EXISTS "${GROUND_TRUTH_LIB_PATH}")
        add_library(VxMath_GroundTruth SHARED IMPORTED)
        set_target_properties(VxMath_GroundTruth PROPERTIES
                IMPORTED_LOCATION "${GROUND_TRUTH_DLL_PATH}"
                IMPORTED_IMPLIB "${GROUND_TRUTH_LIB_PATH}"
                INTERFACE_INCLUDE_DIRECTORIES "${NEMO_INCLUDE_DIR}"
        )
        set(TEST_TARGET VxMath_GroundTruth)
        message(STATUS "Using ground-truth VxMath.dll for tests: ${GROUND_TRUTH_DLL_PATH}")

        # Copy ground-truth DLL to all possible build output directories
        foreach(CONFIG_TYPE Debug Release RelWithDebInfo MinSizeRel)
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CONFIG_TYPE}")
            file(COPY_FILE "${GROUND_TRUTH_DLL_PATH}" "${CMAKE_BINARY_DIR}/bin/${CONFIG_TYPE}/VxMath.dll" ONLY_IF_DIFFERENT)
        endforeach()
        # Also copy to the non-config bin directory
        file(COPY_FILE "${GROUND_TRUTH_DLL_PATH}" "${CMAKE_BINARY_DIR}/bin/VxMath.dll" ONLY_IF_DIFFERENT)
    else()
        message(WARNING "Ground-truth DLL not found at ${GROUND_TRUTH_DLL_PATH}, falling back to built library")
        set(TEST_TARGET VxMath)
    endif()
else()
    set(TEST_TARGET VxMath)
endif()

# Function to add VxMath tests
function(vx_add_test TEST_NAME)
    cmake_parse_arguments(TEST "" "" "SOURCES;DEPENDENCIES" ${ARGN})

    if (NOT TEST_SOURCES)
        message(FATAL_ERROR "vx_add_test: SOURCES argument is required")
    endif ()

    add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_compile_definitions(${TEST_NAME} PRIVATE VX_TESTING)
    if (TEST_TARGET STREQUAL "VxMath_GroundTruth")
        target_compile_definitions(${TEST_NAME} PRIVATE VX_USING_GROUND_TRUTH)
    endif()
    set_target_properties(${TEST_NAME} PROPERTIES
            FOLDER "Tests"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_include_directories(${TEST_NAME} PRIVATE
            ${NEMO_INCLUDE_DIR}
            ${PROJECT_SOURCE_DIR}/src/VxMath
            ${STB_INCLUDE_DIR}
    )

    target_link_libraries(${TEST_NAME} PRIVATE
            ${TEST_TARGET}
            ${TEST_DEPENDENCIES}
            gtest_main
            gmock
    )

    # Register with CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

set(VXMATH_TEST_SOURCES
        VectorTest.cpp
        QuaternionTest.cpp
        MatrixTest.cpp
        FrustumTest.cpp
        PlaneTest.cpp
        RayTest.cpp
        SphereTest.cpp
        GeometryTest.cpp
        IntersectionTest.cpp
        DistanceTest.cpp
        PropertyTests.cpp
        SIMDDispatchConsistencyTest.cpp
        ImageTest.cpp
        # SIMD Tests
        SIMDFeatureTest.cpp
        SIMDUtilityTest.cpp
        SIMDVectorTest.cpp
        SIMDMatrixTest.cpp
        SIMDQuaternionTest.cpp
        # BlitEngine Tests
        BlitEngineFormatConversionTest.cpp
        BlitEngineAlphaTest.cpp
        BlitEngineResizeTest.cpp
        BlitEngineQuantizeTest.cpp
        BlitEnginePalettedTest.cpp
        BlitEngineUpsideDownTest.cpp
        BlitEnginePixelFormatTest.cpp
        # BlitEngine Full Coverage Tests
        BlitEngineEdgeCaseTest.cpp
        BlitEngineContractTest.cpp
        BlitEngineMatrixTest.cpp
)

foreach (TEST_SOURCE ${VXMATH_TEST_SOURCES})
    # Extract test name from source file
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    # Do NOT add VxMath as a dependency - we use TEST_TARGET (which is VxMath_GroundTruth)
    vx_add_test(${TEST_NAME} SOURCES ${TEST_SOURCE})
    
    # Add special include path for SIMD tests
    if (${TEST_NAME} MATCHES "^SIMD")
        target_include_directories(${TEST_NAME} PRIVATE
                ${PROJECT_SOURCE_DIR}/src/VxMath
                ${PROJECT_SOURCE_DIR}/deps/simde
                ${CMAKE_CURRENT_SOURCE_DIR}  # For SIMDTestCommon.h
        )
    endif()
endforeach ()
