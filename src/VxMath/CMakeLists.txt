# =============================================================================
# VxMath Library
# =============================================================================

set(VXMATH_PUBLIC_HEADERS
        ${NEMO_INCLUDE_DIR}/VxPlatform.h
        ${NEMO_INCLUDE_DIR}/VxEndian.h
        ${NEMO_INCLUDE_DIR}/VxMathCompiler.h
        ${NEMO_INCLUDE_DIR}/VxMemory.h
        ${NEMO_INCLUDE_DIR}/VxMathDefines.h
        ${NEMO_INCLUDE_DIR}/VxMath.h
        ${NEMO_INCLUDE_DIR}/VxDefines.h
        ${NEMO_INCLUDE_DIR}/XUtil.h
        ${NEMO_INCLUDE_DIR}/XP.h
        ${NEMO_INCLUDE_DIR}/XSmartPtr.h
        ${NEMO_INCLUDE_DIR}/XString.h
        ${NEMO_INCLUDE_DIR}/XArray.h
        ${NEMO_INCLUDE_DIR}/XBitArray.h
        ${NEMO_INCLUDE_DIR}/XSArray.h
        ${NEMO_INCLUDE_DIR}/XClassArray.h
        ${NEMO_INCLUDE_DIR}/XList.h
        ${NEMO_INCLUDE_DIR}/XHashFun.h
        ${NEMO_INCLUDE_DIR}/XHashTable.h
        ${NEMO_INCLUDE_DIR}/XSHashTable.h
        ${NEMO_INCLUDE_DIR}/XNHashTable.h
        ${NEMO_INCLUDE_DIR}/XMatrix.h
        ${NEMO_INCLUDE_DIR}/VxSharedLibrary.h
        ${NEMO_INCLUDE_DIR}/VxMemoryMappedFile.h
        ${NEMO_INCLUDE_DIR}/VxMemoryPool.h
        ${NEMO_INCLUDE_DIR}/CKPathSplitter.h
        ${NEMO_INCLUDE_DIR}/CKDirectoryParser.h
        ${NEMO_INCLUDE_DIR}/VxWindowFunctions.h
        ${NEMO_INCLUDE_DIR}/VxVector.h
        ${NEMO_INCLUDE_DIR}/Vx2dVector.h
        ${NEMO_INCLUDE_DIR}/VxMatrix.h
        ${NEMO_INCLUDE_DIR}/VxConfiguration.h
        ${NEMO_INCLUDE_DIR}/VxQuaternion.h
        ${NEMO_INCLUDE_DIR}/VxRect.h
        ${NEMO_INCLUDE_DIR}/VxOBB.h
        ${NEMO_INCLUDE_DIR}/VxRay.h
        ${NEMO_INCLUDE_DIR}/VxSphere.h
        ${NEMO_INCLUDE_DIR}/VxPlane.h
        ${NEMO_INCLUDE_DIR}/VxIntersect.h
        ${NEMO_INCLUDE_DIR}/VxDistance.h
        ${NEMO_INCLUDE_DIR}/VxFrustum.h
        ${NEMO_INCLUDE_DIR}/VxColor.h
        ${NEMO_INCLUDE_DIR}/VxTimeProfiler.h
        ${NEMO_INCLUDE_DIR}/VxImageDescEx.h
        ${NEMO_INCLUDE_DIR}/VxMutex.h
        ${NEMO_INCLUDE_DIR}/VxThread.h
)

set(VXMATH_PRIVATE_HEADERS
        VxEigenMatrix.h
        VxBlitEngine.h
        VxSIMD.h
        VxSIMD.inl
        NeuQuant.h
)

set(VXMATH_SOURCES
        VxMath.cpp
        VxGraphicUtils.cpp
        VxBlitEngine.cpp
        VxProcessor.cpp
        CKPathSplitter.cpp
        CKDirectoryParser.cpp
        VxColor.cpp
        VxVector.cpp
        VxBbox.cpp
        VxMatrix.cpp
        VxConfiguration.cpp
        VxQuaternion.cpp
        VxRect.cpp
        VxRay.cpp
        VxPlane.cpp
        VxIntersect_BoxBox.cpp
        VxIntersect_Plane.cpp
        VxIntersect_FaceFace.cpp
        VxIntersect_Frustum.cpp
        VxIntersect_Sphere.cpp
        VxDistance.cpp
        VxFrustum.cpp
        VxMemory.cpp
        XString.cpp
        EscapeURL.cpp
        NeuQuant.cpp
)

# Platform-specific sources
if (WIN32)
    list(APPEND VXMATH_SOURCES
            VxSharedLibraryWin32.cpp
            VxMemoryMappedFileWin32.cpp
            VxTimeProfilerWin32.cpp
            VxWindowFunctionsWin32.cpp
            VxMutexWin32.cpp
            VxThreadWin32.cpp
    )
else ()
    list(APPEND VXMATH_SOURCES
            VxSharedLibraryPosix.cpp
            VxMemoryMappedFilePosix.cpp
            VxTimeProfilerPosix.cpp
            VxWindowFunctionsPosix.cpp
            VxMutexPosix.cpp
            VxThreadPosix.cpp
    )
endif ()

# =============================================================================
# Library Target
# =============================================================================

add_library(VxMath SHARED
        ${VXMATH_SOURCES}
        ${VXMATH_PUBLIC_HEADERS}
        ${VXMATH_PRIVATE_HEADERS}
)
add_library(NeMo2::VxMath ALIAS VxMath)

set_target_properties(VxMath PROPERTIES
        OUTPUT_NAME "VxMath"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        FOLDER "Libraries"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_compile_features(VxMath PUBLIC cxx_std_17)
target_compile_definitions(VxMath PRIVATE VX_API)

# MSVC on x64 implicitly enables SSSE3; GCC/Clang need explicit flag
target_compile_options(VxMath PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mssse3>
)

target_include_directories(VxMath
        PUBLIC
        $<BUILD_INTERFACE:${NEMO_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(VxMath
        PUBLIC
        NeMo2::CompileOptions
        PRIVATE
        miniz stb simde
)

if (WIN32)
    target_link_libraries(VxMath PRIVATE urlmon)
elseif (UNIX)
    target_link_libraries(VxMath PRIVATE ${CMAKE_DL_LIBS})
endif ()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS VxMath
        EXPORT NeMo2Targets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(FILES ${VXMATH_PUBLIC_HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
