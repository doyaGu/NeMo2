# =============================================================================
# VxMath Tests
# =============================================================================

set(TEST_TARGET VxMath)

function(vx_add_test TEST_NAME)
    cmake_parse_arguments(TEST "" "" "SOURCES;DEPENDENCIES" ${ARGN})

    if(NOT TEST_SOURCES)
        message(FATAL_ERROR "vx_add_test: SOURCES argument is required")
    endif()

    add_executable(${TEST_NAME} ${TEST_SOURCES})

    target_compile_definitions(${TEST_NAME} PRIVATE VX_TESTING)

    set_target_properties(${TEST_NAME} PROPERTIES
        FOLDER "Tests/VxMath"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_include_directories(${TEST_NAME} PRIVATE
        ${NEMO_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/src/VxMath
    )

    if(TARGET stb)
        target_link_libraries(${TEST_NAME} PRIVATE stb)
    endif()

    target_link_libraries(${TEST_NAME} PRIVATE
        ${TEST_TARGET}
        ${TEST_DEPENDENCIES}
        gtest_main
        gmock
    )

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

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

foreach(TEST_SOURCE ${VXMATH_TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

    vx_add_test(${TEST_NAME} SOURCES ${TEST_SOURCE})

    # SIMD tests need additional configuration
    if(TEST_NAME MATCHES "^SIMD")
        target_include_directories(${TEST_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/src/VxMath
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        if(TARGET simde)
            target_link_libraries(${TEST_NAME} PRIVATE simde)
        endif()
    endif()
endforeach()
