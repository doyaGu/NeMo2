#ifndef RCKRENDERMANAGER_H
#define RCKRENDERMANAGER_H

#include "XObjectArray.h"
#include "CKRenderEngineTypes.h"
#include "CKRenderEngineEnums.h"
#include "CKRenderManager.h"
#include "CKSceneGraph.h"
#include "VertexCacheOptimizer.h"

class RCK3dEntity;

class RCKRenderManager : public CKRenderManager {
public:
    explicit RCKRenderManager(CKContext *context);
    ~RCKRenderManager() override;

    CKERROR PreClearAll() override;
    CKERROR PreProcess() override;
    CKERROR PostProcess() override;
    CKERROR SequenceAddedToScene(CKScene *scn, CK_ID *objids, int count) override;
    CKERROR SequenceRemovedFromScene(CKScene *scn, CK_ID *objids, int count) override;
    CKERROR OnCKEnd() override;
    CKERROR OnCKPause() override;
    CKERROR SequenceToBeDeleted(CK_ID *objids, int count) override;
    CKERROR SequenceDeleted(CK_ID *objids, int count) override;
    CKDWORD GetValidFunctionsMask() override;

    int GetRenderDriverCount() override;
    VxDriverDesc *GetRenderDriverDescription(int Driver) override;
    void GetDesiredTexturesVideoFormat(VxImageDescEx &VideoFormat) override;
    void SetDesiredTexturesVideoFormat(VxImageDescEx &VideoFormat) override;
    CKRenderContext *GetRenderContext(int pos) override;
    CKRenderContext *GetRenderContextFromPoint(CKPOINT &pt) override;
    int GetRenderContextCount() override;
    void Process() override;
    void FlushTextures() override;
    CKRenderContext *CreateRenderContext(void *Window, int Driver = 0, CKRECT *rect = NULL, CKBOOL Fullscreen = FALSE, int Bpp = -1, int Zbpp = -1, int StencilBpp = -1, int RefreshRate = 0) override;
    CKERROR DestroyRenderContext(CKRenderContext *context) override;
    void RemoveRenderContext(CKRenderContext *context) override;
    CKVertexBuffer *CreateVertexBuffer() override;
    void DestroyVertexBuffer(CKVertexBuffer *VB) override;
    void SetRenderOptions(CKSTRING RenderOptionString, CKDWORD Value) override;
    const VxEffectDescription &GetEffectDescription(int EffectIndex) override;
    int GetEffectCount() override;
    int AddEffect(const VxEffectDescription &NewEffect) override;

    CKMaterial *GetDefaultMaterial();

    CKDWORD CreateObjectIndex(CKRST_OBJECTTYPE type);
    CKBOOL ReleaseObjectIndex(CKDWORD index, CKRST_OBJECTTYPE type);

    void DetachAllObjects();
    void DestroyingDevice(CKRenderContext *ctx);

    void DeleteAllVertexBuffers();

    void AddTemporaryCallback(
        CKCallbacksContainer *callbacks,
        void *Function,
        void *Argument,
        CKBOOL preOrPost);
    void RemoveTemporaryCallback(CKCallbacksContainer *callbacks);
    void ClearTemporaryCallbacks();
    void RemoveAllTemporaryCallbacks();
    void RegisterDefaultEffects();

    void SaveLastFrameMatrix();
    void CleanMovedEntities();

    // Last-frame tracking list management (mirrors CK2_3D.dll sub_1000D770/D7A0)
    void RegisterLastFrameEntity(RCK3dEntity *entity);
    void UnregisterLastFrameEntity(RCK3dEntity *entity);

    // Scene graph node management
    CKSceneGraphNode *CreateNode(RCK3dEntity *entity);
    void DeleteNode(CKSceneGraphNode *node);
    CKSceneGraphRootNode *GetRootNode() { return &m_SceneGraphRootNode; }

    // Entity movement tracking (called when entities move)
    void AddMovedEntity(RCK3dEntity *entity) { m_MovedEntities.PushBack((CKObject*)entity); }

    // Render context mask management
    CKDWORD GetRenderContextMaskFree() { return m_RenderContextMaskFree; }
    void ReleaseRenderContextMaskFree(CKDWORD mask) { m_RenderContextMaskFree |= mask; }

    // Driver management
    CKRasterizerDriver *GetDriver(int DriverIndex);
    CKRasterizerContext *GetFullscreenContext();
    int GetPreferredSoftwareDriver();

public:
    XClassArray<VxCallBack> m_TemporaryPreRenderCallbacks;  // 0x28
    XClassArray<VxCallBack> m_TemporaryPostRenderCallbacks; // 0x34
    XSObjectArray m_RenderContexts;                          // 0x40
    XArray<CKRasterizer *> m_Rasterizers;                    // 0x48
    VxDriverDescEx *m_Drivers;                               // 0x54
    int m_DriverCount;                                       // 0x58
    CKMaterial *m_DefaultMat;                                // 0x5C
    CKDWORD m_RenderContextMaskFree;                         // 0x60
    CKSceneGraphRootNode m_SceneGraphRootNode;               // 0x64 (84 bytes)
    XObjectPointerArray m_MovedEntities;                     // 0xB8
    XObjectPointerArray m_Entities;                          // 0xC4
    VertexCacheOptimizer m_VertexCacheOptimizer;             // 0xD0
    XArray<CKVertexBuffer *> m_VertexBuffers;
    VxOption m_ForceLinearFog;
    VxOption m_ForceSoftware;
    VxOption m_EnsureVertexShader;
    VxOption m_DisableFilter;
    VxOption m_DisableDithering;
    VxOption m_Antialias;
    VxOption m_DisableMipmap;
    VxOption m_DisableSpecular;
    VxOption m_UseIndexBuffers;
    VxOption m_EnableScreenDump;
    VxOption m_EnableDebugMode;
    VxOption m_VertexCache;
    VxOption m_SortTransparentObjects;
    VxOption m_TextureCacheManagement;
    VxOption m_DisablePerspectiveCorrection;
    VxOption m_TextureVideoFormat;
    VxOption m_SpriteVideoFormat;
    XArray<VxOption*> m_Options;
    CK2dEntity *m_2DRootFore;
    CK2dEntity *m_2DRootBack;
    CK_ID m_2DRootBackId;
    CK_ID m_2DRootForeId;
    XClassArray<VxEffectDescription> m_Effects;
};

#endif // RCKRENDERMANAGER_H
