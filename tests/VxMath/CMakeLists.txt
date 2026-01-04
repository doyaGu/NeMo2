# =============================================================================
# VxMath Tests
# =============================================================================

# =============================================================================
# Test Sources
# =============================================================================

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
        BlitEngineEdgeCaseTest.cpp
        BlitEngineContractTest.cpp
        BlitEngineMatrixTest.cpp
)

# =============================================================================
# Add Tests
# =============================================================================

foreach (TEST_SOURCE ${VXMATH_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SOURCE})

    target_compile_definitions(${TEST_NAME} PRIVATE VX_TESTING)

    set_target_properties(${TEST_NAME} PROPERTIES
            FOLDER "Tests/VxMath"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_include_directories(${TEST_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/src/VxMath
    )

    set(TEST_DEPS NeMo2::VxMath stb GTest::gtest_main GTest::gmock)

    # SIMD tests need additional configuration
    if (TEST_NAME MATCHES "^SIMD")
        list(APPEND TEST_DEPS simde)
    endif ()

    target_link_libraries(${TEST_NAME} PRIVATE ${TEST_DEPS})

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach ()
