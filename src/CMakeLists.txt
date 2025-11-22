set(VXMATH_PUBLIC_HEADERS
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
)

set(VXMATH_SOURCES
        VxMath.cpp
        VxStructureUtils.cpp
        VxGraphicUtils.cpp
        VxProcessor.cpp
        VxSharedLibraryWin32.cpp
        VxMemoryMappedFileWin32.cpp
        VxMemoryPool.cpp
        CKPathSplitter.cpp
        CKDirectoryParser.cpp
        VxWindowFunctionsWin32.cpp
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
        VxTimeProfilerWin32.cpp
        VxMutex.cpp
        VxThread.cpp
        VxMemory.cpp
        XString.cpp
        EscapeURL.cpp
)

set(CK2_PUBLIC_HEADERS
        ${NEMO_INCLUDE_DIR}/CKAll.h

        # Defines and Global Functions
        ${NEMO_INCLUDE_DIR}/CKDefines2.h
        ${NEMO_INCLUDE_DIR}/CKGlobals.h
        ${NEMO_INCLUDE_DIR}/CKBaseManager.h
        ${NEMO_INCLUDE_DIR}/CKContext.h
        ${NEMO_INCLUDE_DIR}/CKInterfaceObjectManager.h

        # Managers
        ${NEMO_INCLUDE_DIR}/CKParameterManager.h
        ${NEMO_INCLUDE_DIR}/CKTimeManager.h
        ${NEMO_INCLUDE_DIR}/CKMessageManager.h
        ${NEMO_INCLUDE_DIR}/CKRenderManager.h
        ${NEMO_INCLUDE_DIR}/CKBehaviorManager.h
        ${NEMO_INCLUDE_DIR}/CKAttributeManager.h
        ${NEMO_INCLUDE_DIR}/CKPluginManager.h
        ${NEMO_INCLUDE_DIR}/CKPathManager.h

        # External Managers
        ${NEMO_INCLUDE_DIR}/CKFloorManager.h
        ${NEMO_INCLUDE_DIR}/CKGridManager.h
        ${NEMO_INCLUDE_DIR}/CKSoundManager.h
        ${NEMO_INCLUDE_DIR}/CKMidiManager.h
        ${NEMO_INCLUDE_DIR}/CKInputManager.h
        ${NEMO_INCLUDE_DIR}/CKCollisionManager.h

        # Misc
        ${NEMO_INCLUDE_DIR}/CKMaterial.h
        ${NEMO_INCLUDE_DIR}/CKTexture.h
        ${NEMO_INCLUDE_DIR}/CKRenderContext.h
        ${NEMO_INCLUDE_DIR}/CKSynchroObject.h

        # Parameters
        ${NEMO_INCLUDE_DIR}/CKParameter.h
        ${NEMO_INCLUDE_DIR}/CKParameterIn.h
        ${NEMO_INCLUDE_DIR}/CKParameterOut.h
        ${NEMO_INCLUDE_DIR}/CKParameterLocal.h
        ${NEMO_INCLUDE_DIR}/CKParameterOperation.h

        # Behaviors
        ${NEMO_INCLUDE_DIR}/CKBehaviorIO.h
        ${NEMO_INCLUDE_DIR}/CKBehaviorLink.h
        ${NEMO_INCLUDE_DIR}/CKBehaviorPrototype.h
        ${NEMO_INCLUDE_DIR}/CKBehavior.h
        ${NEMO_INCLUDE_DIR}/CKMessage.h
        ${NEMO_INCLUDE_DIR}/CKObjectDeclaration.h

        # Level/Scene/place
        ${NEMO_INCLUDE_DIR}/CKLevel.h
        ${NEMO_INCLUDE_DIR}/CKPlace.h
        ${NEMO_INCLUDE_DIR}/CKGroup.h
        ${NEMO_INCLUDE_DIR}/CKScene.h

        # Save/load
        ${NEMO_INCLUDE_DIR}/CKStateChunk.h
        ${NEMO_INCLUDE_DIR}/CKFile.h

        # Sound
        ${NEMO_INCLUDE_DIR}/CKSound.h
        ${NEMO_INCLUDE_DIR}/CKWaveSound.h
        ${NEMO_INCLUDE_DIR}/CKMidiSound.h
        ${NEMO_INCLUDE_DIR}/CKSoundReader.h

        # Curves
        ${NEMO_INCLUDE_DIR}/CK2dCurve.h
        ${NEMO_INCLUDE_DIR}/CK2dCurvePoint.h
        ${NEMO_INCLUDE_DIR}/CKCurve.h
        ${NEMO_INCLUDE_DIR}/CKCurvePoint.h

        # Character and Animation
        ${NEMO_INCLUDE_DIR}/CKAnimation.h
        ${NEMO_INCLUDE_DIR}/CKKeyedAnimation.h
        ${NEMO_INCLUDE_DIR}/CKObjectAnimation.h
        ${NEMO_INCLUDE_DIR}/CKKinematicChain.h
        ${NEMO_INCLUDE_DIR}/CKCharacter.h

        # Base Objects
        ${NEMO_INCLUDE_DIR}/CKObject.h
        ${NEMO_INCLUDE_DIR}/CKSceneObject.h
        ${NEMO_INCLUDE_DIR}/CKRenderObject.h
        ${NEMO_INCLUDE_DIR}/CKBeObject.h
        ${NEMO_INCLUDE_DIR}/CKDependencies.h

        # 2d Objects
        ${NEMO_INCLUDE_DIR}/CK2dEntity.h
        ${NEMO_INCLUDE_DIR}/CKSprite.h
        ${NEMO_INCLUDE_DIR}/CKSpriteText.h

        # 3d Objects
        ${NEMO_INCLUDE_DIR}/CKMesh.h
        ${NEMO_INCLUDE_DIR}/CKPatchMesh.h
        ${NEMO_INCLUDE_DIR}/CK3dEntity.h
        ${NEMO_INCLUDE_DIR}/CKCamera.h
        ${NEMO_INCLUDE_DIR}/CKTargetCamera.h
        ${NEMO_INCLUDE_DIR}/CKSprite3D.h
        ${NEMO_INCLUDE_DIR}/CKLight.h
        ${NEMO_INCLUDE_DIR}/CKTargetLight.h
        ${NEMO_INCLUDE_DIR}/CK3dObject.h
        ${NEMO_INCLUDE_DIR}/CKBodyPart.h
        ${NEMO_INCLUDE_DIR}/CKSkin.h

        # Containers
        ${NEMO_INCLUDE_DIR}/XObjectArray.h
        ${NEMO_INCLUDE_DIR}/CKDataArray.h
        ${NEMO_INCLUDE_DIR}/CKDebugContext.h
        ${NEMO_INCLUDE_DIR}/CKMemoryPool.h
)

set(CK2_PRIVATE_HEADERS)

set(CK2_SOURCES
        CK2Win32.cpp

        # Global Functions and Base Classes
        CKGlobals.cpp
        CKBaseManager.cpp
        CKContext.cpp

        # Managers
        CKObjectManager.cpp
        CKParameterManager.cpp
        CKTimeManager.cpp
        CKMessageManager.cpp
        CKBehaviorManager.cpp
        CKAttributeManager.cpp
        CKPluginManager.cpp
        CKPathManager.cpp
        CKGridManager.cpp

        # Misc
        CKDebugContext.cpp
        CKSynchroObject.cpp
        CKRenderContext.cpp
        CKBitmapData.cpp
        CKMemoryPool.cpp

        # Parameters
        CKParameter.cpp
        CKParameterIn.cpp
        CKParameterOut.cpp
        CKParameterLocal.cpp
        CKParameterOperation.cpp
        CKCallbackFunctions.cpp

        # Behaviors
        CKBehaviorIO.cpp
        CKBehaviorLink.cpp
        CKBehaviorPrototype.cpp
        CKBehavior.cpp
        CKMessage.cpp
        CKObjectDeclaration.cpp

        # Level/Scene/place
        CKLevel.cpp
        CKGroup.cpp
        CKScene.cpp

        # Save/load
        CKStateChunk.cpp
        CKJpegDecoder.cpp
        CKFile.cpp

        # Sound
        CKSound.cpp
        CKSoundManager.cpp
        CKWaveSound.cpp
        CKMidiSound.cpp

        # Curves
        CK2dCurve.cpp
        CK2dCurvePoint.cpp

        # Base Objects
        CKObject.cpp
        CKSceneObject.cpp
        CKSceneObjectDesc.cpp
        CKBeObject.cpp
        CKDependencies.cpp
        CKInterfaceObjectManager.cpp

        CKRenderContext.cpp

        # Containers
        XObjectArray.cpp
        CKDataArray.cpp
        CKObjectArray.cpp
)

# ==================== VxMath Library ====================
add_library(VxMath SHARED
    ${VXMATH_SOURCES}
    ${VXMATH_PUBLIC_HEADERS}
    ${VXMATH_PRIVATE_HEADERS}
)

# Set target properties
set_target_properties(VxMath PROPERTIES
        OUTPUT_NAME "VxMath"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Compile definitions
target_compile_definitions(VxMath PRIVATE VX_API)

# Include directories
target_include_directories(VxMath
    PUBLIC
        $<BUILD_INTERFACE:${NEMO_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${STB_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(VxMath
    PRIVATE
        miniz
)

# ==================== CK2 Library ====================
add_library(CK2 SHARED
    ${CK2_SOURCES}
    ${CK2_PUBLIC_HEADERS}
    ${CK2_PRIVATE_HEADERS}
)

# Add library alias for consistent usage
add_library(NeMo2::CK2 ALIAS CK2)

set_target_properties(CK2 PROPERTIES
        OUTPUT_NAME "CK2"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Compile definitions
target_compile_definitions(CK2 PRIVATE CK_API)

# Include directories
target_include_directories(CK2
    PUBLIC
        $<BUILD_INTERFACE:${NEMO_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${STB_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(CK2
    PUBLIC
        VxMath
    PRIVATE
        miniz
)

# Windows-specific settings
if(WIN32)
    target_compile_definitions(CK2 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ==================== Installation ====================
# Install runtime libraries
install(TARGETS VxMath CK2
    EXPORT NeMo2Targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

# Install public headers
install(FILES ${VXMATH_PUBLIC_HEADERS}
    DESTINATION include
)

install(FILES ${CK2_PUBLIC_HEADERS}
    DESTINATION include
)

# Export targets for use by other projects
install(EXPORT NeMo2Targets
    FILE NeMo2Targets.cmake
    NAMESPACE NeMo2::
    DESTINATION lib/cmake/NeMo2
)
