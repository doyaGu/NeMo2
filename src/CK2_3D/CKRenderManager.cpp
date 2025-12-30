#include "RCKRenderManager.h"

#include "CKLevel.h"
#include "CKRasterizer.h"
#include "CKMaterial.h"
#include "CKParameterManager.h"
#include "RCKRenderContext.h"
#include "RCK3dEntity.h"
#include "RCK2dEntity.h"
#include "RCKMesh.h"
#include "RCKTexture.h"
#include "RCKSprite.h"
#include "RCKSpriteText.h"
#include "RCKVertexBuffer.h"

// External reference to rasterizer info array from CK2_3D.cpp
extern XClassArray<CKRasterizerInfo> g_RasterizersInfo;

// Helper function to update driver description from rasterizer driver
static void UpdateDriverDescCaps(VxDriverDescEx *drvDesc) {
    CKRasterizerDriver *rstDriver = drvDesc->RasterizerDriver;

    if (!rstDriver) {
        // NULL rasterizer case
        memset(&drvDesc->DriverDesc, 0, sizeof(drvDesc->DriverDesc));
        strcpy(drvDesc->DriverDesc, "NULL Rasterizer");
        strcpy(drvDesc->DriverDesc2, "NULL Rasterizer");
        drvDesc->CapsUpToDate = TRUE;
        drvDesc->Hardware = FALSE;
        drvDesc->DisplayModeCount = 1;
        drvDesc->DisplayModes = new VxDisplayMode[1];
        drvDesc->DisplayModes[0].Width = 640;
        drvDesc->DisplayModes[0].Height = 480;
        drvDesc->DisplayModes[0].Bpp = 32;
        drvDesc->DisplayModes[0].RefreshRate = 60; // Changed from 0 to 60
        drvDesc->Caps2D.Caps = 7;
        return;
    }

    // Copy caps from rasterizer driver
    drvDesc->CapsUpToDate = rstDriver->m_CapsUpToDate;
    strncpy(drvDesc->DriverDesc, rstDriver->m_Desc.CStr(), sizeof(drvDesc->DriverDesc) - 1);
    strncpy(drvDesc->DriverDesc2, rstDriver->m_Desc.CStr(), sizeof(drvDesc->DriverDesc2) - 1);
    drvDesc->Hardware = rstDriver->m_Hardware;

    // Copy 3D and 2D caps
    memcpy(&drvDesc->Caps3D, &rstDriver->m_3DCaps, sizeof(Vx3DCapsDesc));
    drvDesc->Caps2D = rstDriver->m_2DCaps;

    // Copy texture formats
    int texFormatCount = rstDriver->m_TextureFormats.Size();
    drvDesc->TextureFormats.Resize(texFormatCount);
    for (int i = 0; i < texFormatCount; ++i) {
        drvDesc->TextureFormats[i] = rstDriver->m_TextureFormats[i].Format;
    }

    // Copy display modes
    int displayModeCount = rstDriver->m_DisplayModes.Size();
    drvDesc->DisplayModeCount = displayModeCount;
    if (drvDesc->DisplayModes) {
        delete[] drvDesc->DisplayModes;
    }
    drvDesc->DisplayModes = new VxDisplayMode[displayModeCount];
    memcpy(drvDesc->DisplayModes, rstDriver->m_DisplayModes.Begin(), displayModeCount * sizeof(VxDisplayMode));
}

RCKRenderManager::RCKRenderManager(CKContext *context) : CKRenderManager(context, (CKSTRING) "Render Manager") {
    // Initialize options
    m_TextureVideoFormat.Set("TextureVideoFormat", _16_ARGB1555);
    m_SpriteVideoFormat.Set("SpriteVideoFormat", _16_ARGB1555);

    m_EnableScreenDump.Set("EnableScreenDump", FALSE);
    m_Options.PushBack(&m_EnableScreenDump);

    m_EnableDebugMode.Set("EnableDebugMode", FALSE);
    m_Options.PushBack(&m_EnableDebugMode);

    m_VertexCache.Set("VertexCache", 16);
    m_Options.PushBack(&m_VertexCache);

    m_SortTransparentObjects.Set("SortTransparentObjects", TRUE);
    m_Options.PushBack(&m_SortTransparentObjects);

    m_TextureCacheManagement.Set("TextureCacheManagement", TRUE);
    m_Options.PushBack(&m_TextureCacheManagement);

    m_UseIndexBuffers.Set("UseIndexBuffers", FALSE);
    m_Options.PushBack(&m_UseIndexBuffers);

    m_ForceLinearFog.Set("ForceLinearFog", FALSE);
    m_Options.PushBack(&m_ForceLinearFog);

    m_EnsureVertexShader.Set("EnsureVertexShader", FALSE);
    m_Options.PushBack(&m_EnsureVertexShader);

    m_ForceSoftware.Set("ForceSoftware", FALSE);
    m_Options.PushBack(&m_ForceSoftware);

    m_DisableFilter.Set("DisableFilter", FALSE);
    m_Options.PushBack(&m_DisableFilter);

    m_DisableDithering.Set("DisableDithering", FALSE);
    m_Options.PushBack(&m_DisableDithering);

    m_Antialias.Set("Antialias", 0);
    m_Options.PushBack(&m_Antialias);

    m_DisableMipmap.Set("DisableMipmap", FALSE);
    m_Options.PushBack(&m_DisableMipmap);

    m_DisableSpecular.Set("DisableSpecular", FALSE);
    m_Options.PushBack(&m_DisableSpecular);

    m_DisablePerspectiveCorrection.Set("DisablePerspectiveCorrection", FALSE);
    m_Options.PushBack(&m_DisablePerspectiveCorrection);

    m_RenderContextMaskFree = -1;
    m_Context->RegisterNewManager(this);

    // Initialize driver-related fields
    m_DriverCount = 0;
    m_Drivers = nullptr;
    m_DefaultMat = nullptr;
    m_2DRootFore = nullptr;
    m_2DRootBack = nullptr;

    // Get main window for rasterizer initialization
    WIN_HANDLE mainWindow = m_Context->GetMainWindow();

    // Start rasterizers and count available drivers
    int rstInfoCount = g_RasterizersInfo.Size();

    for (CKRasterizerInfo *rstInfo = g_RasterizersInfo.Begin(); rstInfo != g_RasterizersInfo.End();) {
        CKRasterizer *rasterizer = nullptr;

        if (rstInfo->StartFct) {
            rasterizer = rstInfo->StartFct(mainWindow);
        }

        if (rasterizer) {
            int driverCount = rasterizer->GetDriverCount();
            m_DriverCount += driverCount;
            m_Rasterizers.PushBack(rasterizer);
            ++rstInfo;
        } else {
            // Remove failed rasterizer info - move to next without incrementing
            rstInfo = g_RasterizersInfo.Remove(rstInfo);
        }
    }

    // Link rasterizers together
    int rasterizerCount = m_Rasterizers.Size();
    for (int i = 0; i < rasterizerCount; ++i) {
        for (int j = 0; j < rasterizerCount; ++j) {
            if (i != j) {
                m_Rasterizers[i]->LinkRasterizer(m_Rasterizers[j]);
            }
        }
    }

    // Allocate driver description array
    if (m_DriverCount > 0) {
        m_Drivers = new VxDriverDescEx[m_DriverCount];
        memset(m_Drivers, 0, sizeof(VxDriverDescEx) * m_DriverCount);
    }

    CKDWORD driverId = 0;

    // First pass: enumerate hardware drivers
    for (int i = 0; i < rasterizerCount; ++i) {
        CKRasterizer *rasterizer = m_Rasterizers[i];
        int drvCount = rasterizer->GetDriverCount();

        for (int k = 0; k < drvCount; ++k) {
            CKRasterizerDriver *rstDriver = rasterizer->GetDriver(k);
            if (rstDriver && rstDriver->m_Hardware) {
                VxDriverDescEx *drvDesc = &m_Drivers[driverId];
                drvDesc->Rasterizer = rasterizer;
                drvDesc->RasterizerDriver = rstDriver;
                drvDesc->DriverId = driverId;
                UpdateDriverDescCaps(drvDesc);
                ++driverId;
            }
        }
    }

    // Second pass: enumerate software drivers
    for (int i = 0; i < rasterizerCount; ++i) {
        CKRasterizer *rasterizer = m_Rasterizers[i];
        int drvCount = rasterizer->GetDriverCount();

        for (int m = 0; m < drvCount; ++m) {
            CKRasterizerDriver *rstDriver = rasterizer->GetDriver(m);
            if (!rstDriver || !rstDriver->m_Hardware) {
                VxDriverDescEx *drvDesc = &m_Drivers[driverId];
                drvDesc->Rasterizer = rasterizer;
                drvDesc->RasterizerDriver = rstDriver;
                drvDesc->DriverId = driverId;
                UpdateDriverDescCaps(drvDesc);
                ++driverId;
            }
        }
    }

    // Create default material
    m_DefaultMat = (CKMaterial *) m_Context->CreateObject(CKCID_MATERIAL, (CKSTRING) "Default Mat", CK_OBJECTCREATION_NONAMECHECK);
    if (m_DefaultMat) {
        m_DefaultMat->ModifyObjectFlags(CK_OBJECT_NOTTOBELISTEDANDSAVED, 0);
    }
    // Create 2D root entities
    m_2DRootFore = (CK2dEntity *) m_Context->CreateObject(CKCID_2DENTITY, (CKSTRING) "2DRootFore", CK_OBJECTCREATION_NONAMECHECK);
    if (m_2DRootFore) {
        m_2DRootFore->ModifyObjectFlags(CK_OBJECT_NOTTOBELISTEDANDSAVED, 0);
        ((RCK2dEntity *) m_2DRootFore)->ModifyFlags(0, CK_2DENTITY_RATIOOFFSET | CK_2DENTITY_CLIPTOCAMERAVIEW);
        m_2DRootFore->Show(CKHIDE);
        m_2DRootForeId = m_2DRootFore->GetID();
    }

    m_2DRootBack = (CK2dEntity *) m_Context->CreateObject(CKCID_2DENTITY, (CKSTRING) "2DRootBack", CK_OBJECTCREATION_NONAMECHECK);
    if (m_2DRootBack) {
        m_2DRootBack->ModifyObjectFlags(CK_OBJECT_NOTTOBELISTEDANDSAVED, 0);
        ((RCK2dEntity *) m_2DRootBack)->ModifyFlags(0, CK_2DENTITY_RATIOOFFSET | CK_2DENTITY_CLIPTOCAMERAVIEW);
        m_2DRootBack->Show(CKHIDE);
        m_2DRootBackId = m_2DRootBack->GetID();
    }

    // Register default material effects
    RegisterDefaultEffects();
}

RCKRenderManager::~RCKRenderManager() {
    // Clean up drivers
    for (int i = 0; i < m_DriverCount; ++i) {
        delete[] m_Drivers[i].DisplayModes;
        m_Drivers[i].TextureFormats.Clear();
    }
    delete[] m_Drivers;
    m_Drivers = nullptr;

    // Close rasterizers
    int rstInfoCount = g_RasterizersInfo.Size();
    for (int i = 0; i < rstInfoCount; ++i) {
        CKRasterizerInfo &info = g_RasterizersInfo[i];
        if (info.CloseFct && m_Rasterizers[i]) {
            info.CloseFct(m_Rasterizers[i]);
        }
    }
}

CKERROR RCKRenderManager::PreClearAll() {
    m_SceneGraphRootNode.Clear();
    DetachAllObjects();
    ClearTemporaryCallbacks();
    DeleteAllVertexBuffers();

    for (int i = 0; i < CKGetClassCount(); ++i) {
        if (CKIsChildClassOf(i, CKCID_RENDERCONTEXT)) {
            int count = m_Context->GetObjectsCountByClassID(i);
            CK_ID *ids = m_Context->GetObjectsListByClassID(i);
            for (int j = 0; j < count; ++j) {
                RCKRenderContext *ctx = (RCKRenderContext *) m_Context->GetObject(ids[j]);
                if (ctx)
                    ctx->OnClearAll();
            }
        } else if (CKIsChildClassOf(i, CKCID_3DENTITY)) {
            int count = m_Context->GetObjectsCountByClassID(i);
            CK_ID *ids = m_Context->GetObjectsListByClassID(i);
            for (int j = 0; j < count; ++j) {
                RCK3dEntity *entity = (RCK3dEntity *) m_Context->GetObject(ids[j]);
                entity->RemoveAllCallbacks();
            }
        } else if (CKIsChildClassOf(i, CKCID_MESH)) {
            int count = m_Context->GetObjectsCountByClassID(i);
            CK_ID *ids = m_Context->GetObjectsListByClassID(i);
            for (int j = 0; j < count; ++j) {
                RCKMesh *mesh = (RCKMesh *) m_Context->GetObject(ids[j]);
                mesh->RemoveAllCallbacks();
            }
        }
    }

    m_DefaultMat = nullptr;
    return CK_OK;
}

CKERROR RCKRenderManager::PreProcess() {
    SaveLastFrameMatrix();
    CleanMovedEntities();
    RemoveAllTemporaryCallbacks();
    return CK_OK;
}

CKERROR RCKRenderManager::PostProcess() {
    // Set moved flag on all moved entities
    for (RCK3dEntity **it = (RCK3dEntity **) m_MovedEntities.Begin();
         it != (RCK3dEntity **) m_MovedEntities.End(); ++it) {
        (*it)->m_MoveableFlags |= VX_MOVEABLE_RESERVED2;
    }

    // Clear extents for all render contexts
    int ctxCount = GetRenderContextCount();
    for (int j = 0; j < ctxCount; ++j) {
        RCKRenderContext *ctx = (RCKRenderContext *) GetRenderContext(j);
        if (ctx) {
            ctx->m_Extents.Resize(0);
        }
    }

    return CK_OK;
}

CKERROR RCKRenderManager::SequenceAddedToScene(CKScene *scn, CK_ID *objids, int count) {
    CKLevel *level = m_Context->GetCurrentLevel();
    if (!level)
        return CK_OK;

    if (level->GetCurrentScene() != scn)
        return CK_OK;

    int ctxCount = level->GetRenderContextCount();
    for (int i = 0; i < ctxCount; ++i) {
        CKRenderContext *ctx = level->GetRenderContext(i);
        if (ctx) {
            for (int j = 0; j < count; ++j) {
                CKObject *obj = m_Context->GetObject(objids[j]);
                if (obj && CKIsChildClassOf(obj, CKCID_RENDEROBJECT)) {
                    ctx->AddObject((CKRenderObject *) obj);
                }
            }
        }
    }

    return CK_OK;
}

CKERROR RCKRenderManager::SequenceRemovedFromScene(CKScene *scn, CK_ID *objids, int count) {
    CKLevel *level = m_Context->GetCurrentLevel();
    if (!level)
        return CK_OK;

    if (level->GetCurrentScene() != scn)
        return CK_OK;

    int ctxCount = level->GetRenderContextCount();
    for (int i = 0; i < ctxCount; ++i) {
        CKRenderContext *ctx = level->GetRenderContext(i);
        if (ctx) {
            for (int j = 0; j < count; ++j) {
                CKObject *obj = m_Context->GetObject(objids[j]);
                if (obj && CKIsChildClassOf(obj, CKCID_RENDEROBJECT))
                    ctx->RemoveObject((CKRenderObject *) obj);
            }
        }
    }

    return CK_OK;
}

CKERROR RCKRenderManager::OnCKEnd() {
    CKObject *obj = m_Context->GetObject(m_2DRootForeId);
    if (obj && CKIsChildClassOf(obj, CKCID_2DENTITY)) {
        m_Context->DestroyObject(m_2DRootBack);
    }

    obj = m_Context->GetObject(m_2DRootBackId);
    if (obj && CKIsChildClassOf(obj, CKCID_2DENTITY)) {
        m_Context->DestroyObject(m_2DRootFore);
    }

    m_2DRootBack = nullptr;
    m_2DRootFore = nullptr;
    m_2DRootForeId = 0;
    m_2DRootBackId = 0;

    return CK_OK;
}

CKERROR RCKRenderManager::OnCKPause() {
    RemoveAllTemporaryCallbacks();

    int ctxCount = GetRenderContextCount();
    for (int i = 0; i < ctxCount; ++i) {
        RCKRenderContext *ctx = (RCKRenderContext *) GetRenderContext(i);
        if (ctx)
            ctx->m_PVInformation = 0;
    }

    return CK_OK;
}

CKERROR RCKRenderManager::SequenceToBeDeleted(CK_ID *objids, int count) {
    m_Entities.Check();
    m_MovedEntities.Check();
    m_SceneGraphRootNode.Check();
    return CK_OK;
}

CKERROR RCKRenderManager::SequenceDeleted(CK_ID *objids, int count) {
    m_RenderContexts.Check(m_Context);
    return CK_OK;
}

CKDWORD RCKRenderManager::GetValidFunctionsMask() {
    return CKMANAGER_FUNC_OnSequenceToBeDeleted |
           CKMANAGER_FUNC_OnSequenceDeleted |
           CKMANAGER_FUNC_PreProcess |
           CKMANAGER_FUNC_PostProcess |
           CKMANAGER_FUNC_PreClearAll |
           CKMANAGER_FUNC_OnCKEnd |
           CKMANAGER_FUNC_OnCKPause |
           CKMANAGER_FUNC_OnSequenceAddedToScene |
           CKMANAGER_FUNC_OnSequenceRemovedFromScene;
}

int RCKRenderManager::GetRenderDriverCount() {
    return m_DriverCount;
}

// Static array to hold VxDriverDesc structures - use fixed size to avoid dynamic allocation issues
static VxDriverDesc s_DriverDescCache[16];
static int s_DriverDescCacheCount = 0;

VxDriverDesc *RCKRenderManager::GetRenderDriverDescription(int Driver) {
    if (Driver < 0 || Driver >= m_DriverCount || Driver >= 16) {
        return nullptr;
    }

    VxDriverDescEx *drv = &m_Drivers[Driver];

    if (!drv->CapsUpToDate) {
        UpdateDriverDescCaps(drv);
    }

    // Copy data to VxDriverDesc - directly copy without XSArray issues
    VxDriverDesc *desc = &s_DriverDescCache[Driver];

    strncpy(desc->DriverDesc, drv->DriverDesc, sizeof(desc->DriverDesc) - 1);
    desc->DriverDesc[sizeof(desc->DriverDesc) - 1] = '\0';
    strncpy(desc->DriverName, drv->DriverDesc2, sizeof(desc->DriverName) - 1);
    desc->DriverName[sizeof(desc->DriverName) - 1] = '\0';
    desc->IsHardware = drv->Hardware;
    desc->DisplayModeCount = drv->DisplayModeCount;
    desc->DisplayModes = drv->DisplayModes;
    // Point to the same TextureFormats - no copy needed
    // Note: We can't easily copy XSArray, so we reference the original
    // desc->TextureFormats stays as-is (caller should use drv->TextureFormats)
    desc->Caps2D = drv->Caps2D;
    desc->Caps3D = drv->Caps3D;

    return desc;
}

void RCKRenderManager::GetDesiredTexturesVideoFormat(VxImageDescEx &VideoFormat) {
    VxPixelFormat2ImageDesc((VX_PIXELFORMAT) m_TextureVideoFormat.Value, VideoFormat);
}

void RCKRenderManager::SetDesiredTexturesVideoFormat(VxImageDescEx &VideoFormat) {
    m_TextureVideoFormat.Value = VxImageDesc2PixelFormat(VideoFormat);
}

CKRenderContext *RCKRenderManager::GetRenderContext(int pos) {
    return (CKRenderContext *) m_RenderContexts.GetObject(m_Context, pos);
}

CKRenderContext *RCKRenderManager::GetRenderContextFromPoint(CKPOINT &pt) {
    for (CK_ID *it = m_RenderContexts.Begin(); it != m_RenderContexts.End(); ++it) {
        RCKRenderContext *ctx = (RCKRenderContext *) m_Context->GetObject(*it);
        if (ctx) {
            WIN_HANDLE win = ctx->GetWindowHandle();
            if (win) {
                CKRECT rect;
                VxGetWindowRect(win, &rect);
                if (VxPtInRect(&rect, &pt))
                    return ctx;
            }
        }
    }
    return nullptr;
}

int RCKRenderManager::GetRenderContextCount() {
    return m_RenderContexts.Size();
}

void RCKRenderManager::Process() {
    for (CK_ID *it = m_RenderContexts.Begin(); it != m_RenderContexts.End(); ++it) {
        CKRenderContext *dev = (CKRenderContext *) m_Context->GetObject(*it);
        if (dev)
            dev->Render(CK_RENDER_USECURRENTSETTINGS);
    }
}

void RCKRenderManager::FlushTextures() {
    int textureCount = m_Context->GetObjectsCountByClassID(CKCID_TEXTURE);
    CK_ID *textureIds = m_Context->GetObjectsListByClassID(CKCID_TEXTURE);
    for (int i = 0; i < textureCount; ++i) {
        CKTexture *texture = (CKTexture *) m_Context->GetObject(textureIds[i]);
        if (texture)
            texture->FreeVideoMemory();
    }

    int spriteCount = m_Context->GetObjectsCountByClassID(CKCID_SPRITE);
    CK_ID *spriteIds = m_Context->GetObjectsListByClassID(CKCID_SPRITE);
    for (int i = 0; i < spriteCount; ++i) {
        CKSprite *sprite = (CKSprite *) m_Context->GetObject(spriteIds[i]);
        if (sprite)
            sprite->FreeVideoMemory();
    }

    int spriteTextCount = m_Context->GetObjectsCountByClassID(CKCID_SPRITETEXT);
    CK_ID *spriteTextIds = m_Context->GetObjectsListByClassID(CKCID_SPRITETEXT);
    for (int i = 0; i < spriteTextCount; ++i) {
        RCKSpriteText *spriteText = (RCKSpriteText *) m_Context->GetObject(spriteTextIds[i]);
        if (spriteText)
            spriteText->FreeVideoMemory();
    }
}

CKRenderContext *RCKRenderManager::CreateRenderContext(void *Window, int Driver, CKRECT *rect, CKBOOL Fullscreen, int Bpp, int Zbpp, int StencilBpp, int RefreshRate) {
    RCKRenderContext *dev = (RCKRenderContext *) m_Context->CreateObject(CKCID_RENDERCONTEXT, nullptr, CK_OBJECTCREATION_NONAMECHECK);
    if (!dev)
        return nullptr;

    if (dev->Create(Window, Driver, rect, Fullscreen, Bpp, Zbpp, StencilBpp, RefreshRate) != CK_OK) {
        m_Context->DestroyObject(dev);
        return nullptr;
    }

    m_RenderContexts.PushBack(dev->GetID());
    return dev;
}

CKERROR RCKRenderManager::DestroyRenderContext(CKRenderContext *context) {
    if (!context)
        return CKERR_INVALIDPARAMETER;

    CKLevel *level = m_Context->GetCurrentLevel();
    if (level)
        level->RemoveRenderContext(context);

    if (!m_RenderContexts.RemoveObject(context))
        return CKERR_INVALIDPARAMETER;

    m_Context->DestroyObject(context);
    return CK_OK;
}

void RCKRenderManager::RemoveRenderContext(CKRenderContext *context) {
    if (context)
        m_RenderContexts.Remove(context->GetID());
}

CKVertexBuffer *RCKRenderManager::CreateVertexBuffer() {
    RCKVertexBuffer *vertexBuffer = new RCKVertexBuffer(m_Context);
    m_VertexBuffers.PushBack(vertexBuffer);
    return vertexBuffer;
}

void RCKRenderManager::DestroyVertexBuffer(CKVertexBuffer *VB) {
    if (VB) {
        if (m_VertexBuffers.Remove(VB))
            delete VB;
    }
}

void RCKRenderManager::SetRenderOptions(CKSTRING RenderOptionString, CKDWORD Value) {
    for (int i = 0; i < m_Options.Size(); ++i) {
        VxOption *option = m_Options[i];
        if (stricmp(option->Key.CStr(), RenderOptionString) == 0) {
            option->Value = Value;
            return;
        }
    }
}

const VxEffectDescription &RCKRenderManager::GetEffectDescription(int EffectIndex) {
    return m_Effects[EffectIndex];
}

int RCKRenderManager::GetEffectCount() {
    return m_Effects.Size();
}

int RCKRenderManager::AddEffect(const VxEffectDescription &NewEffect) {
    int size = m_Effects.Size();
    m_Effects.PushBack(NewEffect);
    m_Effects[size].EffectIndex = (VX_EFFECT) size;
    return size;
}

CKDWORD RCKRenderManager::CreateObjectIndex(CKRST_OBJECTTYPE type) {
    if (m_DriverCount == 0)
        return 0;
    return m_Drivers->Rasterizer->CreateObjectIndex(type, TRUE);
}

CKBOOL RCKRenderManager::ReleaseObjectIndex(CKDWORD index, CKRST_OBJECTTYPE type) {
    if (m_DriverCount == 0)
        return true;
    return m_Drivers->Rasterizer->ReleaseObjectIndex(index, type, TRUE);
}

CKMaterial *RCKRenderManager::GetDefaultMaterial() {
    if (!m_DefaultMat) {
        m_DefaultMat = (CKMaterial *) m_Context->CreateObject(CKCID_MATERIAL, (CKSTRING) "Default Mat", CK_OBJECTCREATION_NONAMECHECK);
        if (m_DefaultMat) {
            m_DefaultMat->ModifyObjectFlags(CK_OBJECT_NOTTOBELISTEDANDSAVED, 0);
        }
    }
    return m_DefaultMat;
}

void RCKRenderManager::DetachAllObjects() {
    m_MovedEntities.Clear();
    m_Entities.Clear();

    for (CK_ID *it = m_RenderContexts.Begin(); it != m_RenderContexts.End(); ++it) {
        CKRenderContext *ctx = (CKRenderContext *) m_Context->GetObject(*it);
        if (ctx) {
            ctx->DetachAll();
            ctx->SetCurrentRenderOptions(255);
        }
    }
}

void RCKRenderManager::DestroyingDevice(CKRenderContext *ctx) {
    RCKRenderContext *rctx = (RCKRenderContext *) ctx;
    CKRasterizerContext *rstCtx = rctx->m_RasterizerContext;

    for (int i = 0; i < CKGetClassCount(); ++i) {
        if (CKIsChildClassOf(i, CKCID_TEXTURE)) {
            int count = m_Context->GetObjectsCountByClassID(i);
            CK_ID *ids = m_Context->GetObjectsListByClassID(i);
            for (int j = 0; j < count; ++j) {
                RCKTexture *tex = (RCKTexture *) m_Context->GetObject(ids[j]);
                if (tex && tex->m_RasterizerContext == rstCtx) {
                    tex->m_RasterizerContext = nullptr;
                }
            }
        } else if (CKIsChildClassOf(i, CKCID_SPRITE)) {
            int count = m_Context->GetObjectsCountByClassID(i);
            CK_ID *ids = m_Context->GetObjectsListByClassID(i);
            for (int j = 0; j < count; ++j) {
                RCKSprite *sprite = (RCKSprite *) m_Context->GetObject(ids[j]);
                if (sprite && sprite->m_RasterizerContext == rstCtx) {
                    sprite->m_RasterizerContext = nullptr;
                }
            }
        }
    }
}

void RCKRenderManager::DeleteAllVertexBuffers() {
    for (CKVertexBuffer **it = (CKVertexBuffer **) m_VertexBuffers.Begin();
         it < (CKVertexBuffer **) m_VertexBuffers.End(); ++it) {
        if (*it) {
            delete *it;
        }
    }
    m_VertexBuffers.Resize(0);
}

void RCKRenderManager::SaveLastFrameMatrix() {
    for (RCK3dEntity **it = (RCK3dEntity **) m_Entities.Begin();
         it != (RCK3dEntity **) m_Entities.End(); ++it) {
        (*it)->SaveLastFrameMatrix();
    }
}

void RCKRenderManager::RegisterLastFrameEntity(RCK3dEntity *entity) {
    if (!entity) return;
    m_Entities.AddIfNotHere(entity);
}

void RCKRenderManager::UnregisterLastFrameEntity(RCK3dEntity *entity) {
    if (!entity) return;
    m_Entities.Remove(entity);
}

void RCKRenderManager::CleanMovedEntities() {
    int count = 0;
    for (RCK3dEntity **it = (RCK3dEntity **) m_MovedEntities.Begin();
         it != (RCK3dEntity **) m_MovedEntities.End(); ++it) {
        RCK3dEntity *entity = *it;
        if ((entity->GetMoveableFlags() & VX_MOVEABLE_RESERVED2) != 0) {
            // Entity was moved this frame, clear both flags
            entity->m_MoveableFlags &= ~(VX_MOVEABLE_HASMOVED | VX_MOVEABLE_RESERVED2);
        } else {
            // Entity was not moved, just clear the flag and keep in list
            entity->m_MoveableFlags &= ~VX_MOVEABLE_RESERVED2;
            m_MovedEntities[count++] = entity;
        }
    }
    m_MovedEntities.Resize(count);
}

void RCKRenderManager::AddTemporaryCallback(CKCallbacksContainer *callbacks, void *Function, void *Argument, CKBOOL preOrPost) {
    VxCallBack cb;
    cb.callback = callbacks;
    cb.argument = Function;
    cb.arg2 = Argument;

    if (preOrPost)
        m_TemporaryPreRenderCallbacks.PushBack(cb);
    else
        m_TemporaryPostRenderCallbacks.PushBack(cb);
}

void RCKRenderManager::RemoveTemporaryCallback(CKCallbacksContainer *callbacks) {
    // Remove from pre-render callbacks array
    for (int i = m_TemporaryPreRenderCallbacks.Size() - 1; i >= 0; --i) {
        if (m_TemporaryPreRenderCallbacks[i].callback == callbacks) {
            m_TemporaryPreRenderCallbacks.RemoveAt(i);
        }
    }

    // Remove from post-render callbacks array
    for (int i = m_TemporaryPostRenderCallbacks.Size() - 1; i >= 0; --i) {
        if (m_TemporaryPostRenderCallbacks[i].callback == callbacks) {
            m_TemporaryPostRenderCallbacks.RemoveAt(i);
        }
    }
}

void RCKRenderManager::ClearTemporaryCallbacks() {
    m_TemporaryPreRenderCallbacks.Resize(0);
    m_TemporaryPostRenderCallbacks.Resize(0);
}

void RCKRenderManager::RemoveAllTemporaryCallbacks() {
    // Remove all pre-render callbacks from their containers
    for (VxCallBack *it = m_TemporaryPreRenderCallbacks.Begin();
         it != m_TemporaryPreRenderCallbacks.End(); ++it) {
        CKCallbacksContainer *container = (CKCallbacksContainer *) it->callback;
        if (container) {
            container->RemovePreCallback(it->argument, it->arg2);
        }
    }

    // Remove all post-render callbacks from their containers
    for (VxCallBack *it = m_TemporaryPostRenderCallbacks.Begin();
         it != m_TemporaryPostRenderCallbacks.End(); ++it) {
        CKCallbacksContainer *container = (CKCallbacksContainer *) it->callback;
        if (container) {
            container->RemovePostCallback(it->argument, it->arg2);
        }
    }

    ClearTemporaryCallbacks();
}

void RCKRenderManager::RegisterDefaultEffects() {
    // Effect 0: None
    VxEffectDescription effectNone;
    effectNone.Summary = "None";
    AddEffect(effectNone);

    // Effect 1: TexGen
    VxEffectDescription effectTexGen;
    effectTexGen.Summary = "TexGen";
    effectTexGen.Description = "Generate texture coordinates.\r\n"
        "The mapping parameter determines which type of coordinates to generate.\r\n"
        "The TexGen parameter controls the ways texture coordinates are generated or transformed:\r\n"
        "None : Mesh texture coordinates.\r\n"
        "Transform : Mesh texture coordinates multiplied by the referential matrix.\r\n"
        "Reflect : Generate texture coordinates that simulate a reflection .\r\n"
        "Planar: Generate texture coordinates projected as projeted from the referential plane.\r\n"
        "Cube Reflect: Generate texture coordinates using reflection vector for a cube map.\r\n"
        "It use the current viewpoint as referential";
    effectTexGen.DescImage = "Effect_CubeReflMap.jpg";
    effectTexGen.MaxTextureCount = 0;
    effectTexGen.NeededTextureCoordsCount = 0;
    effectTexGen.ParameterType = CKPGUID_TEXGENEFFECT;
    effectTexGen.ParameterDescription = "TexGen Type";
    effectTexGen.ParameterDefaultValue = "Reflect";
    AddEffect(effectTexGen);

    // Effect 2: TexGen with referential
    VxEffectDescription effectTexGenRef;
    effectTexGenRef.Summary = "TexGen with referential";
    effectTexGenRef.Description = "Generate texture coordinates.\r\n"
        "The TexGen parameter controls the ways texture coordinates are generated or transformed:\r\n"
        "None : Mesh texture coordinates.\r\n"
        "Transform : Mesh texture coordinates multiplied by the referential matrix.\r\n"
        "Reflect : Generate texture coordinates that simulate a reflection .\r\n"
        "Planar: Generate texture coordinates projected as projeted from the referential plane.\r\n"
        "Cube Reflect: Generate texture coordinates using reflection vector for a cube map.\r\n"
        "This works as Tex coordinates generation effect but an additionnal referential can be used instead of the viewpoint.\r\n";
    effectTexGenRef.DescImage = "Effect_CubeReflMap.jpg";
    effectTexGenRef.MaxTextureCount = 0;
    effectTexGenRef.NeededTextureCoordsCount = 0;
    effectTexGenRef.ParameterType = CKPGUID_TEXGENREFEFFECT;
    effectTexGenRef.ParameterDescription = "TexGen Params";
    effectTexGenRef.ParameterDefaultValue = "Reflect,NULL";
    AddEffect(effectTexGenRef);

    // Effect 3: Environment Bump Map
    VxEffectDescription effectBumpEnv;
    effectBumpEnv.Summary = "Environment Bump Map";
    effectBumpEnv.Description = "The default amplitude of the bump effect is 2.0f, this can be amplified or reduced"
        "by an offset given in the amplitude parameter.\r\n"
        "The Env. Texture can be either a cube map or a normal texture and the way it is combined with the base texture can be given in the parameter\r\n"
        "See the Tex coords generation effect for details on the TexGen param\r\n"
        "See the Combine 2 Textures effect for details on the Combine param";
    effectBumpEnv.DescImage = "Effect_BumpMapSmooth.jpg";
    effectBumpEnv.MaxTextureCount = 2;
    effectBumpEnv.NeededTextureCoordsCount = 2;
    effectBumpEnv.Tex1Description = "Bump Texture";
    effectBumpEnv.Tex2Description = "Env. Texture";
    effectBumpEnv.ParameterType = CKPGUID_BUMPMAPPARAM;
    effectBumpEnv.ParameterDescription = "Params";
    effectBumpEnv.ParameterDefaultValue = "0,Add,Reflect,NULL";
    AddEffect(effectBumpEnv);

    // Effect 4: Floor DotProduct3 Lighting
    VxEffectDescription effectDP3;
    effectDP3.Summary = "Floor DotProduct3 Lighting";
    effectDP3.Description = "The Bump texture should be in normal format. "
        "When this effect is not valid because of video card limitation, the texture will simply be modulated with lighting.";
    effectDP3.DescImage = "Effect_DP3.jpg";
    effectDP3.MaxTextureCount = 1;
    effectDP3.NeededTextureCoordsCount = 1;
    effectDP3.Tex1Description = "Bump Texture (Normals)";
    effectDP3.ParameterType = CKPGUID_3DENTITY;
    effectDP3.ParameterDescription = "Light";
    effectDP3.ParameterDefaultValue = "NULL";
    AddEffect(effectDP3);

    // Effect 5: Combine 2 Textures
    VxEffectDescription effectCombine2;
    effectCombine2.Summary = "Combine 2 Textures";
    effectCombine2.Description =
        "Blends two textures, the base material one and the one given in the effect, with a set of"
        "texture coordinates generated by a method similar to the one in the TexGen effect."
        "If you do not select a generation method here it will use the texture coordinates set in the channel (if available).\r\n";
    effectCombine2.DescImage = "Effect_Blend2Textures.jpg";
    effectCombine2.MaxTextureCount = 1;
    effectCombine2.NeededTextureCoordsCount = 0;
    effectCombine2.ParameterType = CKPGUID_COMBINE2TEX;
    effectCombine2.ParameterDescription = "Params";
    effectCombine2.ParameterDefaultValue = "Modulate,None,NULL";
    AddEffect(effectCombine2);

    // Effect 6: Combine 3 Textures
    VxEffectDescription effectCombine3;
    effectCombine3.Summary = "Combine 3 Textures";
    effectCombine3.Description = "Effect similar than the Combine 2 Textures except than it will work on 3.\r\n";
    effectCombine3.DescImage = "Effect_Blend3Textures.jpg";
    effectCombine3.MaxTextureCount = 2;
    effectCombine3.NeededTextureCoordsCount = 0;
    effectCombine3.ParameterType = CKPGUID_COMBINE3TEX;
    effectCombine3.ParameterDescription = "Params";
    effectCombine3.ParameterDefaultValue = "Modulate,None,NULL,Modulate,None,NULL";
    AddEffect(effectCombine3);

    // Build enumeration string for Material Effect
    XString effectEnum;
    for (int i = 0; i < m_Effects.Size(); ++i) {
        effectEnum << m_Effects[i].Summary;
        effectEnum << "=";
        effectEnum << m_Effects[i].EffectIndex;
        if (i < m_Effects.Size() - 1)
            effectEnum << ",";
    }

    // Register parameter types with ParameterManager
    CKParameterManager *pm = m_Context->GetParameterManager();
    if (!pm)
        return;

    // Register Material Effect enum
    pm->RegisterNewEnum(CKPGUID_MATERIALEFFECT, (CKSTRING) "Material Effect", (CKSTRING) effectEnum.CStr());
    CKParameterTypeDesc *typeDesc = pm->GetParameterTypeDescription(CKPGUID_MATERIALEFFECT);
    if (typeDesc)
        typeDesc->dwFlags |= CKPARAMETERTYPE_HIDDEN;

    // Register Tex Coords Generator enum
    pm->RegisterNewEnum(CKPGUID_TEXGENEFFECT, (CKSTRING) "Tex Coords Generator",
                        (CKSTRING)
                        "None=0,Transform=1,Reflect=2,Chrome=3,Planar=4,CubeMap Reflect=31,CubeMap SkyMap=32,CubeMap Normals=33,CubeMap Positions=34");
    typeDesc = pm->GetParameterTypeDescription(CKPGUID_TEXGENEFFECT);
    if (typeDesc)
        typeDesc->dwFlags |= CKPARAMETERTYPE_HIDDEN;

    // Register Texture Blending enum
    pm->RegisterNewEnum(CKPGUID_TEXCOMBINE, (CKSTRING) "Texture Blending",
                        (CKSTRING)
                        "None=0,Modulate=4,Modulate 2X=5,Modulate 4X=6,Add=7,Add Signed=8,Add Signed 2X=9,Subtract=10,Add Smooth=11,"
                        "Blend Using Diffuse Alpha=12,Blend Using Texture Alpha=13,Blend Using Current Alpha=16,"
                        "Modulate Alpha Add Color=18,Modulate Color Add Alpha=19,Modulate InvAlpha Add Color=20,Modulate InvColor Add Alpha=21");
    typeDesc = pm->GetParameterTypeDescription(CKPGUID_TEXCOMBINE);
    if (typeDesc)
        typeDesc->dwFlags |= CKPARAMETERTYPE_HIDDEN;

    // Register TexgenReferential structure: (TexGen, Referential)
    pm->RegisterNewStructure(CKPGUID_TEXGENREFEFFECT, (CKSTRING) "TexgenReferential",
                             (CKSTRING) "TexGen,Referential",
                             CKPGUID_TEXGENEFFECT, CKPGUID_3DENTITY);

    // Register Combine2Textures structure: (Combine, TexGen, Referential)
    pm->RegisterNewStructure(CKPGUID_COMBINE2TEX, (CKSTRING) "Combine 2 Textures",
                             (CKSTRING) "Combine,TexGen,Referential",
                             CKPGUID_TEXCOMBINE, CKPGUID_TEXGENEFFECT, CKPGUID_3DENTITY);

    // Register Combine3Textures structure: (Combine1, TexGen1, Ref1, Combine2, TexGen2, Ref2)
    pm->RegisterNewStructure(CKPGUID_COMBINE3TEX, (CKSTRING) "Combine 3 Textures",
                             (CKSTRING) "Combine1,TexGen1,Ref1,Combine2,TexGen2,Ref2",
                             CKPGUID_TEXCOMBINE, CKPGUID_TEXGENEFFECT, CKPGUID_3DENTITY,
                             CKPGUID_TEXCOMBINE, CKPGUID_TEXGENEFFECT, CKPGUID_3DENTITY);

    // Register BumpmapParameters structure: (Amplitude, EnvMap Combine, EnvMap TexGen, EnvMap Referential)
    pm->RegisterNewStructure(CKPGUID_BUMPMAPPARAM, (CKSTRING) "Bumpmap Parameters",
                             (CKSTRING) "Amplitude,EnvMap Combine,EnvMap TexGen,EnvMap Referential",
                             CKPGUID_FLOAT, CKPGUID_TEXCOMBINE, CKPGUID_TEXGENEFFECT, CKPGUID_3DENTITY);
}

// =====================================================
// Scene Graph Node Management
// =====================================================

CKSceneGraphNode *RCKRenderManager::CreateNode(RCK3dEntity *entity) {
    CKSceneGraphNode *node = new CKSceneGraphNode(entity);
    if (node) {
        m_SceneGraphRootNode.AddNode(node);
    }
    return node;
}

void RCKRenderManager::DeleteNode(CKSceneGraphNode *node) {
    if (!node)
        return;

    // Remove from parent
    if (node->m_Parent) {
        node->m_Parent->RemoveNode(node);
    }

    // Delete the node
    delete node;
}

// =====================================================
// Driver Management
// =====================================================

CKRasterizerDriver *RCKRenderManager::GetDriver(int DriverIndex) {
    // IDA: 0x1006f7f0
    if (DriverIndex >= m_DriverCount)
        return nullptr;
    return m_Drivers[DriverIndex].RasterizerDriver;
}

CKRasterizerContext *RCKRenderManager::GetFullscreenContext() {
    // IDA: 0x10074f8a
    for (int i = 0; i < m_Rasterizers.Size(); ++i) {
        CKRasterizer *rasterizer = m_Rasterizers[i];
        if (rasterizer && rasterizer->m_FullscreenContext) {
            return rasterizer->m_FullscreenContext;
        }
    }
    return nullptr;
}

int RCKRenderManager::GetPreferredSoftwareDriver() {
    // IDA: 0x100733f0
    // First pass: prefer OpenGL software driver
    for (int i = 0; i < m_DriverCount; ++i) {
        CKRasterizerDriver *driver = GetDriver(i);
        if (driver && !driver->m_Hardware && driver->m_2DCaps.Family == CKRST_OPENGL) {
            return i;
        }
    }

    // Second pass: any software driver
    for (int i = 0; i < m_DriverCount; ++i) {
        CKRasterizerDriver *driver = GetDriver(i);
        if (driver && !driver->m_Hardware) {
            return i;
        }
    }

    return 0;
}
