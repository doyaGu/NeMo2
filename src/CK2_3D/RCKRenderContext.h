#ifndef RCKRENDERCONTEXT_H
#define RCKRENDERCONTEXT_H

#include "CKRenderEngineTypes.h"
#include "CKRenderContext.h"
#include "CKRenderedScene.h"
#include "CKRasterizerEnums.h"

// Forward declarations
class RCKMaterial;
class RCK3dEntity;
class RCKSprite3D;

struct UserDrawPrimitiveDataClass : public VxDrawPrimitiveData {
    UserDrawPrimitiveDataClass();
    ~UserDrawPrimitiveDataClass();

    UserDrawPrimitiveDataClass(const UserDrawPrimitiveDataClass &) = delete;
    UserDrawPrimitiveDataClass &operator=(const UserDrawPrimitiveDataClass &) = delete;

    alignas(VxDrawPrimitiveData) unsigned char m_CachedDP[sizeof(VxDrawPrimitiveData)];
    CKWORD *m_Indices;
    int m_MaxIndexCount;
    int m_MaxVertexCount;

    VxDrawPrimitiveData *GetStructure(CKRST_DPFLAGS DpFlags, int VertexCount);
    CKWORD *GetIndices(int IndicesCount);
    void ClearStructure();
    void AllocateStructure();
};

struct CKRenderContextSettings
{
    CKRECT m_Rect;
    CKDWORD m_Bpp;
    CKDWORD m_Zbpp;
    CKDWORD m_StencilBpp;
};

struct CKRenderExtents {
    VxRect m_Rect;      // 0x00 - Screen extent rectangle
    CKDWORD m_Flags;   // 0x10 - Pointer to entity (stored as CKDWORD)
    CK_ID m_Camera;     // 0x14 - Associated camera
};

struct CKObjectExtents {
    VxRect m_Rect;          // 0x00 - Screen extent rectangle
    CK3dEntity *m_Entity;   // 0x10 - Pointer to entity
    CK_ID m_Camera;         // 0x14 - Associated camera
};

class RCKRenderContext : public CKRenderContext {
    friend class RCKRenderManager;
public:
    void AddObject(CKRenderObject *obj) override;
    void AddObjectWithHierarchy(CKRenderObject *obj) override;
    void RemoveObject(CKRenderObject *obj) override;
    CKBOOL IsObjectAttached(CKRenderObject *obj) override;
    const XObjectArray & Compute3dRootObjects() override;
    const XObjectArray & Compute2dRootObjects() override;
    CK2dEntity * Get2dRoot(CKBOOL background) override;
    void DetachAll() override;
    void ForceCameraSettingsUpdate() override;
    void PrepareCameras(CK_RENDER_FLAGS Flags = CK_RENDER_USECURRENTSETTINGS) override;
    CKERROR Clear(CK_RENDER_FLAGS Flags = CK_RENDER_USECURRENTSETTINGS, CKDWORD Stencil = 0) override;
    CKERROR DrawScene(CK_RENDER_FLAGS Flags = CK_RENDER_USECURRENTSETTINGS) override;
    CKERROR BackToFront(CK_RENDER_FLAGS Flags = CK_RENDER_USECURRENTSETTINGS) override;
    CKERROR Render(CK_RENDER_FLAGS Flags = CK_RENDER_USECURRENTSETTINGS) override;
    void AddPreRenderCallBack(CK_RENDERCALLBACK Function, void *Argument, CKBOOL Temporary = FALSE) override;
    void RemovePreRenderCallBack(CK_RENDERCALLBACK Function, void *Argument) override;
    void AddPostRenderCallBack(CK_RENDERCALLBACK Function, void *Argument, CKBOOL Temporary = FALSE, CKBOOL BeforeTransparent = FALSE) override;
    void RemovePostRenderCallBack(CK_RENDERCALLBACK Function, void *Argument) override;
    void AddPostSpriteRenderCallBack(CK_RENDERCALLBACK Function, void *Argument, CKBOOL Temporary = FALSE) override;
    void RemovePostSpriteRenderCallBack(CK_RENDERCALLBACK Function, void *Argument) override;
    VxDrawPrimitiveData *GetDrawPrimitiveStructure(CKRST_DPFLAGS Flags, int VertexCount) override;
    CKWORD *GetDrawPrimitiveIndices(int IndicesCount) override;
    void Transform(VxVector *Dest, VxVector *Src, CK3dEntity *Ref = NULL) override;
    void TransformVertices(int VertexCount, VxTransformData *data, CK3dEntity *Ref = NULL) override;
    CKERROR GoFullScreen(int Width = 640, int Height = 480, int Bpp = -1, int Driver = 0, int RefreshRate = 0) override;
    CKERROR StopFullScreen() override;
    CKBOOL IsFullScreen() override;
    int GetDriverIndex() override;
    CKBOOL ChangeDriver(int NewDriver) override;
    WIN_HANDLE GetWindowHandle() override;
    void ScreenToClient(Vx2DVector *ioPoint) override;
    void ClientToScreen(Vx2DVector *ioPoint) override;
    CKERROR SetWindowRect(VxRect &rect, CKDWORD Flags = 0) override;
    void GetWindowRect(VxRect &rect, CKBOOL ScreenRelative = FALSE) override;
    int GetHeight() override;
    int GetWidth() override;
    CKERROR Resize(int PosX = 0, int PosY = 0, int SizeX = 0, int SizeY = 0, CKDWORD Flags = 0) override;
    void SetViewRect(VxRect &rect) override;
    void GetViewRect(VxRect &rect) override;
    VX_PIXELFORMAT GetPixelFormat(int *Bpp = NULL, int *Zbpp = NULL, int *StencilBpp = NULL) override;
    void SetState(VXRENDERSTATETYPE State, CKDWORD Value) override;
    CKDWORD GetState(VXRENDERSTATETYPE State) override;
    CKBOOL SetTexture(CKTexture *tex, CKBOOL Clamped = 0, int Stage = 0) override;
    CKBOOL SetTextureStageState(CKRST_TEXTURESTAGESTATETYPE State, CKDWORD Value, int Stage = 0) override;
    CKRasterizerContext *GetRasterizerContext() override;
    void SetClearBackground(CKBOOL ClearBack = TRUE) override;
    CKBOOL GetClearBackground() override;
    void SetClearZBuffer(CKBOOL ClearZ = TRUE) override;
    CKBOOL GetClearZBuffer() override;
    void GetGlobalRenderMode(VxShadeType *Shading, CKBOOL *Texture, CKBOOL *Wireframe) override;
    void SetGlobalRenderMode(VxShadeType Shading = GouraudShading, CKBOOL Texture = TRUE, CKBOOL Wireframe = FALSE) override;
    void SetCurrentRenderOptions(CKDWORD flags) override;
    CKDWORD GetCurrentRenderOptions() override;
    void ChangeCurrentRenderOptions(CKDWORD Add, CKDWORD Remove) override;
    void SetCurrentExtents(VxRect &extents) override;
    void GetCurrentExtents(VxRect &extents) override;
    void SetAmbientLight(float R, float G, float B) override;
    void SetAmbientLight(CKDWORD Color) override;
    CKDWORD GetAmbientLight() override;
    void SetFogMode(VXFOG_MODE Mode) override;
    void SetFogStart(float Start) override;
    void SetFogEnd(float End) override;
    void SetFogDensity(float Density) override;
    void SetFogColor(CKDWORD Color) override;
    VXFOG_MODE GetFogMode() override;
    float GetFogStart() override;
    float GetFogEnd() override;
    float GetFogDensity() override;
    CKDWORD GetFogColor() override;
    CKBOOL DrawPrimitive(VXPRIMITIVETYPE pType, CKWORD *indices, int indexcount, VxDrawPrimitiveData *data) override;
    void SetWorldTransformationMatrix(const VxMatrix &M) override;
    void SetProjectionTransformationMatrix(const VxMatrix &M) override;
    void SetViewTransformationMatrix(const VxMatrix &M) override;
    const VxMatrix &GetWorldTransformationMatrix() override;
    const VxMatrix &GetProjectionTransformationMatrix() override;
    const VxMatrix &GetViewTransformationMatrix() override;
    CKBOOL SetUserClipPlane(CKDWORD ClipPlaneIndex, const VxPlane &PlaneEquation) override;
    CKBOOL GetUserClipPlane(CKDWORD ClipPlaneIndex, VxPlane &PlaneEquation) override;
    CKRenderObject *Pick(int x, int y, CKPICKRESULT *oRes, CKBOOL iIgnoreUnpickable = FALSE) override;
    CKRenderObject *Pick(CKPOINT pt, CKPICKRESULT *oRes, CKBOOL iIgnoreUnpickable = FALSE) override;
    CKERROR RectPick(const VxRect &r, XObjectPointerArray &oObjects, CKBOOL Intersect = TRUE) override;
    void AttachViewpointToCamera(CKCamera *cam) override;
    void DetachViewpointFromCamera() override;
    CKCamera *GetAttachedCamera() override;
    CK3dEntity *GetViewpoint() override;
    CKMaterial *GetBackgroundMaterial() override;
    void GetBoundingBox(VxBbox *BBox) override;
    void GetStats(VxStats *stats) override;
    void SetCurrentMaterial(CKMaterial *mat, CKBOOL Lit = TRUE) override;
    void Activate(CKBOOL active = TRUE) override;
    int DumpToMemory(const VxRect *iRect, VXBUFFER_TYPE buffer, VxImageDescEx &desc) override;
    int CopyToVideo(const VxRect *iRect, VXBUFFER_TYPE buffer, VxImageDescEx &desc) override;
    CKERROR DumpToFile(CKSTRING filename, const VxRect *rect, VXBUFFER_TYPE buffer) override;
    VxDirectXData *GetDirectXInfo() override;
    void WarnEnterThread() override;
    void WarnExitThread() override;
    CK2dEntity *Pick2D(const Vx2DVector &v) override;
    CKBOOL SetRenderTarget(CKTexture *texture = NULL, int CubeMapFace = 0) override;
    void AddRemoveSequence(CKBOOL Start) override;
    void SetTransparentMode(CKBOOL Trans) override;
    void AddDirtyRect(CKRECT *Rect) override;
    void RestoreScreenBackup() override;
    CKDWORD GetStencilFreeMask() override;
    void UsedStencilBits(CKDWORD stencilBits) override;
    int GetFirstFreeStencilBits() override;
    VxDrawPrimitiveData *LockCurrentVB(CKDWORD VertexCount) override;
    CKBOOL ReleaseCurrentVB() override;
    void SetTextureMatrix(const VxMatrix &M, int Stage = 0) override;
    void SetStereoParameters(float EyeSeparation, float FocalLength) override;
    void GetStereoParameters(float &EyeSeparation, float &FocalLength) override;

    // Internal methods
    CKBOOL DestroyDevice();
    void ClearCallbacks();
    void SetClipRect(VxRect *rect);
    void SetFullViewport(CKViewportData *vp, int width, int height);
    void UpdateProjection(CKBOOL forceUpdate);
    void AddSprite3DBatch(RCKSprite3D *sprite);
    void CallSprite3DBatches();
    void FlushSprite3DBatchesIfNeeded();  // IDA: sub_1000D2F0
    void AddExtents2D(const VxRect &rect, CKObject *obj);
    void CheckObjectExtents();
    void RenderTransparents(CKDWORD flags);
    
    // Internal pick methods
    CK3dEntity *Pick3D(const Vx2DVector &pt, VxIntersectionDesc *desc, CK3dEntity *filter, CKBOOL ignoreUnpickable);
    CK2dEntity *_Pick2D(const Vx2DVector &pt, CKBOOL ignoreUnpickable);

    CKERROR Create(void *Window, int Driver, CKRECT *rect, CKBOOL Fullscreen, int Bpp, int Zbpp, int StencilBpp, int RefreshRate);
    VxStats &GetStats() {
        return m_Stats;
    }

    explicit RCKRenderContext(CKContext *Context, CKSTRING name = nullptr);
    ~RCKRenderContext() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;
    void PostLoad() override;

    void PreDelete() override;
    void CheckPreDeletion() override;

    int GetMemoryOccupation() override;
    CKBOOL IsObjectUsed(CKObject *obj, CK_CLASSID cid) override;

    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKObject *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

private:
    CK_RENDER_FLAGS ResolveRenderFlags(CK_RENDER_FLAGS Flags) const;
    void ExecutePreRenderCallbacks();
    void ExecutePostRenderCallbacks(CKBOOL beforeTransparent);
    void ExecutePostSpriteCallbacks();
    void LoadPVInformationTexture();
    void DrawPVInformationWatermark();
    void FillStateString();

    static int ClassifyTransparentOrder(const RCK3dEntity *a, const RCK3dEntity *b, const VxVector &cam);

public:
    // =====================================================================
    // Binary-aligned member layout - matches IDA RCKRenderContext (956 bytes)
    // Base class CKRenderContext at offset 0x00 (20 bytes)
    // =====================================================================
    
    CKDWORD m_WinHandle;                    // 0x14 (4 bytes)
    CKDWORD m_AppHandle;                    // 0x18 (4 bytes) - WIN_HANDLE
    CKRECT m_WinRect;                       // 0x1C (16 bytes)
    CKDWORD m_RenderFlags;                  // 0x2C (4 bytes) - CK_RENDER_FLAGS stored as DWORD
    CKRenderedScene *m_RenderedScene;       // 0x30 (4 bytes)
    CKBOOL m_Fullscreen;                    // 0x34 (4 bytes)
    CKBOOL m_Active;                        // 0x38 (4 bytes)
    CKBOOL m_Perspective;                   // 0x3C (4 bytes) - renamed from m_PerspectiveOrOrthographic
    CKBOOL m_ProjectionUpdated;             // 0x40 (4 bytes)
    CKBOOL m_Start;                         // 0x44 (4 bytes)
    CKBOOL m_TransparentMode;               // 0x48 (4 bytes)
    CKBOOL m_DeviceValid;                   // 0x4C (4 bytes)
    CKCallbacksContainer m_PreRenderCallBacks;      // 0x50 (28 bytes)
    CKCallbacksContainer m_PostRenderCallBacks;  // 0x6C (28 bytes)
    CKCallbacksContainer m_PostSpriteRenderCallBacks;     // 0x88 (28 bytes)
    RCKRenderManager *m_RenderManager;      // 0xA4 (4 bytes)
    CKRasterizerContext *m_RasterizerContext; // 0xA8 (4 bytes)
    CKRasterizerDriver *m_RasterizerDriver; // 0xAC (4 bytes)
    int m_DriverIndex;                      // 0xB0 (4 bytes) - NOTE: m_Driver removed, only m_DriverIndex exists
    CKDWORD m_Shading;                      // 0xB4 (4 bytes)
    CKDWORD m_TextureEnabled;               // 0xB8 (4 bytes)
    CKDWORD m_DisplayWireframe;             // 0xBC (4 bytes)
    VxFrustum m_Frustum;                    // 0xC0 (172 bytes)
    float m_Fov;                            // 0x16C (4 bytes)
    float m_Zoom;                           // 0x170 (4 bytes)
    float m_NearPlane;                      // 0x174 (4 bytes)
    float m_FarPlane;                       // 0x178 (4 bytes)
    VxMatrix m_ProjectionMatrix;            // 0x17C (64 bytes)
    CKViewportData m_ViewportData;          // 0x1BC (24 bytes)
    CKRenderContextSettings m_Settings;     // 0x1D4 (28 bytes)
    CKRenderContextSettings m_FullscreenSettings; // 0x1F0 (28 bytes)
    VxRect m_CurrentExtents;                // 0x20C (16 bytes)
    int m_FpsFrameCount;                    // 0x21C (4 bytes) - frame counter used for FPS smoothing
    CKDWORD m_TimeFpsCalc;                  // 0x220 (4 bytes)
    VxTimeProfiler m_RenderTimeProfiler;    // 0x224 (16 bytes)
    float m_SmoothedFps;                    // 0x234 (4 bytes)
    VxStats m_Stats;                        // 0x238 (72 bytes)
    VxTimeProfiler m_DevicePreCallbacksTimeProfiler;    // 0x280 (16 bytes)
    VxTimeProfiler m_DevicePostCallbacksTimeProfiler;   // 0x290 (16 bytes)
    VxTimeProfiler m_ObjectsCallbacksTimeProfiler;      // 0x2A0 (16 bytes)
    VxTimeProfiler m_SpriteCallbacksTimeProfiler;       // 0x2B0 (16 bytes)
    VxTimeProfiler m_ObjectsRenderTimeProfiler;         // 0x2C0 (16 bytes)
    VxTimeProfiler m_SceneTraversalTimeProfiler;        // 0x2D0 (16 bytes)
    VxTimeProfiler m_SkinTimeProfiler;                  // 0x2E0 (16 bytes)
    VxTimeProfiler m_SpriteTimeProfiler;                // 0x2F0 (16 bytes)
    VxTimeProfiler m_TransparentObjectsSortTimeProfiler; // 0x300 (16 bytes)
    RCK3dEntity *m_Current3dEntity;         // 0x310 (4 bytes)
    RCKTexture *m_TargetTexture;            // 0x314 (4 bytes)
    CKRST_CUBEFACE m_CubeMapFace;           // 0x318 (4 bytes) - changed type to match IDA
    float m_FocalLength;                    // 0x31C (4 bytes)
    float m_EyeSeparation;                  // 0x320 (4 bytes)
    CKDWORD m_Flags;                        // 0x324 (4 bytes)
    CKDWORD m_FpsInterval;                  // 0x328 (4 bytes)
    XString m_CurrentObjectDesc;            // 0x32C (8 bytes)
    XString m_StateString;                  // 0x334 (8 bytes)
    CKDWORD m_SceneTraversalCalls;          // 0x33C (4 bytes)
    CKDWORD m_DrawSceneCalls;               // 0x340 (4 bytes)
    CKDWORD m_SortTransparentObjects;       // 0x344 (4 bytes)
    XArray<RCKMaterial*> m_Sprite3DBatches; // 0x348 (12 bytes) - Materials for sprite batch rendering
    XArray<RCK3dEntity*> m_TransparentObjects; // 0x354 (12 bytes) - Transparent entities for sorting/rendering
    int m_StencilFreeMask;                  // 0x360 (4 bytes)
    UserDrawPrimitiveDataClass *m_UserDrawPrimitiveData; // 0x364 (4 bytes)
    CKDWORD m_MaskFree;                     // 0x368 (4 bytes)
    CKDWORD m_VertexBufferIndex;            // 0x36C (4 bytes)
    int m_StartIndex;                       // 0x370 (4 bytes)
    CKDWORD m_DpFlags;                      // 0x374 (4 bytes)
    CKDWORD m_VertexBufferCount;            // 0x378 (4 bytes)
    XArray<CKObjectExtents> m_ObjectExtents; // 0x37C (12 bytes) - Object screen extents for picking
    XArray<CKRenderExtents> m_Extents;      // 0x388 (12 bytes) - Additional extents array
    XObjectArray m_RootObjects;             // 0x394 (12 bytes) - single array for both 3D and 2D roots
    RCKCamera *m_Camera;                    // 0x3A0 (4 bytes)
    RCKTexture *m_NCUTex;                   // 0x3A4 (4 bytes)
    VxTimeProfiler m_PVTimeProfiler;        // 0x3A8 (16 bytes)
    CKDWORD m_PVInformation;                // 0x3B8 (4 bytes)
    // Total: 956 bytes (0x3BC)

    void OnClearAll();
};

#endif // RCKRENDERCONTEXT_H