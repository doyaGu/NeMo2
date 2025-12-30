#ifndef CKRENDEREDSCENE_H
#define CKRENDEREDSCENE_H

#include "CKContext.h"
#include "VxRect.h"

class RCKRenderContext;
class RCKMaterial;
class RCK3dEntity;
class RCKCamera;
class CKMaterial;

class CKRenderedScene {
    friend class RCKRenderContext;
    friend class RCK2dEntity;
public:
    CKRenderedScene(CKRenderContext *rc);
    ~CKRenderedScene();
    CKERROR Draw(CK_RENDER_FLAGS Flags);
    void SetDefaultRenderStates(CKRasterizerContext *rst);
    void PrepareCameras(CK_RENDER_FLAGS Flags = CK_RENDER_USECURRENTSETTINGS);
    void SetupLights(CKRasterizerContext *rst);
    void ResizeViewport(const VxRect &rect);
    void UpdateViewportSize(int forceUpdate, CK_RENDER_FLAGS Flags);
    void ForceCameraSettingsUpdate();
    void AddObject(CKRenderObject *obj);
    void RemoveObject(CKRenderObject *obj);
    void DetachAll();
    CKMaterial *GetBackgroundMaterial() const { return m_BackgroundMaterial; }
    CK3dEntity *GetRootEntity() const { return m_RootEntity; }

protected:
    CKRenderContext *m_RenderContext;
    CKContext *m_Context;
    CKMaterial *m_BackgroundMaterial;
    CK3dEntity *m_RootEntity;
    CKCamera *m_AttachedCamera;
    XObjectPointerArray m_3DEntities;
    XObjectPointerArray m_Cameras;
    XObjectPointerArray m_Lights;
    CKDWORD m_FogMode;
    float m_FogStart;
    float m_FogEnd;
    float m_FogDensity;
    CKDWORD m_FogColor;
    CKDWORD m_AmbientLight;
    CKDWORD m_LightCount;
    XArray<CK2dEntity *> m_2DEntities;
};

#endif // CKRENDEREDSCENE_H