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
)

# =============================================================================
# Library Target
# =============================================================================

nemo_add_library(
        NAME CK2_3D
        TYPE SHARED
        PUBLIC_HEADERS ${CKRE_PUBLIC_HEADERS}
        PRIVATE_HEADERS ${CKRE_PRIVATE_HEADERS}
        SOURCES ${CKRE_SOURCES}
        PUBLIC_DEPS
        NeMo2::CK2
        NeMo2::VxMath
        NeMo2::CompileOptions
        EXPORT_ALL
)

# =============================================================================
# Submodules
# =============================================================================

add_subdirectory(CKRasterizer)

target_link_libraries(CK2_3D PRIVATE NeMo2::CKRasterizerLib)

# =============================================================================
# Tracy Integration (if enabled)
# =============================================================================

if (NEMO_ENABLE_TRACY)
    target_compile_definitions(CK2_3D PRIVATE TRACY_ENABLE)

    if (TARGET Tracy::TracyClient)
        target_link_libraries(CK2_3D PUBLIC Tracy::TracyClient)
    elseif (TARGET TracyClient)
        target_link_libraries(CK2_3D PUBLIC TracyClient)
    else ()
        message(FATAL_ERROR "NEMO_ENABLE_TRACY is ON but TracyClient target was not found")
    endif ()
endif ()

# =============================================================================
# Installation
# =============================================================================

nemo_install_library(
        TARGET CK2_3D
        HEADERS ${CKRE_PUBLIC_HEADERS}
)
