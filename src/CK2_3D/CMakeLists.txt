# =============================================================================
# CK2_3D (Render Engine) Library
# =============================================================================

set(CKRE_PUBLIC_HEADERS)

set(CKRE_PRIVATE_HEADERS
    CKRenderEngineTypes.h
    CKRenderEngineEnums.h
    MeshAdjacency.h
    RadixSort.h
    MeshStriper.h
    NvStripifier.h
    VertexCacheOptimizer.h
    NearestPointGrid.h
    PlaceFitter.h
    RCKRenderManager.h
    RCKRenderContext.h
    RCKMaterial.h
    RCKTexture.h
    RCKMesh.h
    RCKPatchMesh.h
    RCKAnimation.h
    RCKKeyedAnimation.h
    RCKObjectAnimation.h
    RCKKeyframeData.h
    RCKKinematicChain.h
    RCKLayer.h
    RCKRenderObject.h
    RCK2dEntity.h
    RCK3dEntity.h
    RCKCamera.h
    RCKLight.h
    RCKCurvePoint.h
    RCKCurve.h
    RCK3dObject.h
    RCKSprite3D.h
    RCKCharacter.h
    RCKPlace.h
    RCKGrid.h
    RCKBodyPart.h
    RCKTargetCamera.h
    RCKTargetLight.h
    RCKSprite.h
    RCKSpriteText.h
    RCKSkin.h
    CKRenderedScene.h
    CKSceneGraph.h
    RCKVertexBuffer.h
)

set(CKRE_SOURCES
    CK2_3D.cpp
    CKRenderManager.cpp
    CKRenderContext.cpp
    CKCallbacksContainer.cpp
    CKMaterial.cpp
    CKTexture.cpp
    CKMesh.cpp
    CKMeshUtils.cpp
    CKPatchMesh.cpp
    CKAnimation.cpp
    CKKeyedAnimation.cpp
    CKObjectAnimation.cpp
    CKKeyframeData.cpp
    CKKinematicChain.cpp
    CKLayer.cpp
    CKRenderObject.cpp
    CK2dEntity.cpp
    CK3dEntity.cpp
    CKCamera.cpp
    CKLight.cpp
    CKCurvePoint.cpp
    CKCurve.cpp
    CK3dObject.cpp
    CKSprite3D.cpp
    CKCharacter.cpp
    CKPlace.cpp
    CKGrid.cpp
    CKBodyPart.cpp
    CKTargetCamera.cpp
    CKTargetLight.cpp
    CKSprite.cpp
    CKSpriteText.cpp
    CKSkin.cpp
    MeshAdjacency.cpp
    RadixSort.cpp
    MeshStriper.cpp
    NvStripifier.cpp
    VertexCacheOptimizer.cpp
    NearestPointGrid.cpp
    PlaceFitter.cpp
    CKRenderedScene.cpp
    CKSceneGraph.cpp
    CKVertexBuffer.cpp
    CKRasterizer.cpp
    CKRasterizerDriver.cpp
    CKRasterizerContext.cpp
)

# =============================================================================
# Library Target
# =============================================================================

add_library(CK2_3D SHARED
    ${CKRE_SOURCES}
    ${CKRE_PUBLIC_HEADERS}
    ${CKRE_PRIVATE_HEADERS}
)
add_library(NeMo2::CK2_3D ALIAS CK2_3D)

set_target_properties(CK2_3D PROPERTIES
    OUTPUT_NAME "CK2_3D"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    FOLDER "Libraries"
)

target_include_directories(CK2_3D
    PUBLIC
        $<BUILD_INTERFACE:${NEMO_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(CK2_3D PUBLIC CK2 VxMath)

# =============================================================================
# Installation
# =============================================================================

install(TARGETS CK2_3D
    EXPORT NeMo2Targets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
