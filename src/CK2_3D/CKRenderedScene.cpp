#include "CKRenderedScene.h"

#include "VxMatrix.h"
#include "CKRenderContext.h"
#include "CKRasterizer.h"
#include "CK3dEntity.h"
#include "CKMaterial.h"
#include "CKLight.h"
#include "CKCamera.h"
#include "CKSceneGraph.h"
#include "RCKRenderManager.h"
#include "RCKRenderContext.h"
#include "RCK3dEntity.h"
#include "RCK2dEntity.h"
#include "RCKCamera.h"
#include "RCKLight.h"

extern int g_UpdateTransparency;
extern int g_FogProjectionMode;

CKRenderedScene::CKRenderedScene(CKRenderContext *rc) {
    // IDA: 0x1006f830
    m_RenderContext = rc;
    m_Context = rc->m_Context;
    m_AmbientLight = 0xFF0F0F0F;
    m_FogMode = 0;
    m_FogStart = 1.0f;
    m_FogEnd = 100.0f;
    m_FogColor = 0;
    m_FogDensity = 1.0f;
    m_LightCount = 0;
    m_BackgroundMaterial = nullptr;
    m_RootEntity = nullptr;
    m_AttachedCamera = nullptr;

    m_RootEntity = (CK3dEntity *) m_Context->CreateObject(CKCID_3DENTITY, nullptr, CK_OBJECTCREATION_NONAMECHECK);
    m_RootEntity->ModifyObjectFlags(CK_OBJECT_NOTTOBELISTEDANDSAVED, 0);
    AddObject(m_RootEntity);

    m_BackgroundMaterial = (CKMaterial *) m_Context->CreateObject(CKCID_MATERIAL, (CKSTRING) "Background Material", CK_OBJECTCREATION_NONAMECHECK);
    m_BackgroundMaterial->ModifyObjectFlags(CK_OBJECT_NOTTOBELISTEDANDSAVED, 0);
    m_BackgroundMaterial->SetDiffuse(VxColor(0.0f, 0.0f, 0.0f));
    m_BackgroundMaterial->SetAmbient(VxColor(0.0f, 0.0f, 0.0f));
    m_BackgroundMaterial->SetSpecular(VxColor(0.0f, 0.0f, 0.0f));
    m_BackgroundMaterial->SetEmissive(VxColor(0.0f, 0.0f, 0.0f));
}

CKRenderedScene::~CKRenderedScene() {
    if (!m_Context->IsInClearAll()) {
        CKDestroyObject(m_BackgroundMaterial);
        CKDestroyObject(m_RootEntity);
    }
}

void CKRenderedScene::AddObject(CKRenderObject *obj) {
    // IDA: 0x1006fae5
    int classId = obj->GetClassID();

    if (CKIsChildClassOf(classId, CKCID_3DENTITY)) {
        m_3DEntities.PushBack((CK3dEntity *) obj);

        if (CKIsChildClassOf(obj->GetClassID(), CKCID_CAMERA)) {
            m_Cameras.PushBack((CKCamera *) obj);
        } else if (CKIsChildClassOf(obj->GetClassID(), CKCID_LIGHT)) {
            m_Lights.PushBack((CKLight *) obj);
        }
    } else if (CKIsChildClassOf(obj->GetClassID(), CKCID_2DENTITY)) {
        m_2DEntities.PushBack((CK2dEntity *) obj);

        CK2dEntity *ent2d = (CK2dEntity *) obj;
        if (!ent2d->GetParent()) {
            RCKRenderManager *renderManager = ((RCKRenderContext *) m_RenderContext)->m_RenderManager;
            if (ent2d->IsBackground()) {
                ent2d->SetParent(renderManager->m_2DRootBack);
            } else {
                ent2d->SetParent(renderManager->m_2DRootFore);
            }
        }
    }
}

void CKRenderedScene::RemoveObject(CKRenderObject *obj) {
    // IDA: 0x1006fc21
    int classId = obj->GetClassID();

    if (CKIsChildClassOf(classId, CKCID_3DENTITY)) {
        m_3DEntities.Remove((CK3dEntity *) obj);

        if (CKIsChildClassOf(obj->GetClassID(), CKCID_CAMERA)) {
            m_Cameras.Remove((CKCamera *) obj);
            if (m_AttachedCamera == (CKCamera *) obj) {
                m_AttachedCamera = nullptr;
            }
        }
        if (CKIsChildClassOf(obj->GetClassID(), CKCID_LIGHT)) {
            m_Lights.Remove((CKLight *) obj);
        }
    } else if (CKIsChildClassOf(obj->GetClassID(), CKCID_2DENTITY)) {
        m_2DEntities.Remove((CK2dEntity *) obj);

        CK2dEntity *ent2d = (CK2dEntity *) obj;
        if (!ent2d->GetParent()) {
            CK2dEntity *root2D = m_RenderContext->Get2dRoot(ent2d->IsBackground());
            ((RCK2dEntity *) root2D)->m_Children.Remove(ent2d);
        }
    }
}

void CKRenderedScene::DetachAll() {
    // IDA: 0x1006fd4f
    RCKRenderContext *rc = (RCKRenderContext *) m_RenderContext;
    RCKRenderManager *rm = rc->m_RenderManager;

    // Clear 2D root children if in ClearAll or if render context not found in list
    if (m_Context->IsInClearAll() || !rm->m_RenderContexts.IsHere(rc->m_ID)) {
        if (rm->m_2DRootBack) {
            ((RCK2dEntity *) rm->m_2DRootBack)->m_Children.Clear();
        }
        if (rm->m_2DRootFore) {
            ((RCK2dEntity *) rm->m_2DRootFore)->m_Children.Clear();
        }
    }

    // Remove from render context if not in ClearAll
    if (!m_Context->IsInClearAll()) {
        for (CK2dEntity **it = m_2DEntities.Begin(); it != m_2DEntities.End(); ++it) {
            RCK2dEntity *entity = (RCK2dEntity *) *it;
            if (entity) {
                entity->RemoveFromRenderContext(rc);
            }
        }

        for (CKObject **it = m_3DEntities.Begin(); it != m_3DEntities.End(); ++it) {
            RCK3dEntity *entity = (RCK3dEntity *) *it;
            if (entity) {
                entity->RemoveFromRenderContext(rc);
            }
        }
    }

    m_2DEntities.Clear();
    m_3DEntities.Clear();
    m_Cameras.Clear();
    m_Lights.Clear();
    m_AttachedCamera = nullptr;
}

CKERROR CKRenderedScene::Draw(CK_RENDER_FLAGS Flags) {
    // IDA: 0x100704ae
    RCKRenderContext *rc = (RCKRenderContext *) m_RenderContext;
    RCKRenderManager *rm = rc->m_RenderManager;
    CKRasterizerContext *rst = rc->m_RasterizerContext;
    RCK3dEntity *rootEntity = (RCK3dEntity *) m_RootEntity;

    SetDefaultRenderStates(rst);

    rc->m_SpriteTimeProfiler.Reset();

    // Render background 2D sprites
    if ((Flags & CK_RENDER_BACKGROUNDSPRITES) != 0 &&
        rm->m_2DRootBack->GetChildrenCount() > 0) {
        VxRect viewRect;
        rc->GetViewRect(viewRect);

        rc->SetFullViewport(&rst->m_ViewportData,
                            rc->m_Settings.m_Rect.right,
                            rc->m_Settings.m_Rect.bottom);
        rst->SetViewport(&rst->m_ViewportData);

        ((RCK2dEntity *) rm->m_2DRootBack)->Render((CKRenderContext *) rc);

        ResizeViewport(viewRect);
    }

    rc->m_Stats.SpriteTime += rc->m_SpriteTimeProfiler.Current();

    // Render 3D scene
    if (!(Flags & CK_RENDER_SKIP3D)) {
        RCKCamera *camera = rc->m_Camera;

        if (camera) {
            // Copy camera world matrix to root entity (3x4 part = 0x30 bytes)
            VxMatrix *camMatrix = (VxMatrix *) &camera->m_WorldMatrix;
            VxMatrix *rootMatrix = (VxMatrix *) &rootEntity->m_WorldMatrix;
            memcpy(rootMatrix, camMatrix, sizeof(VxMatrix) - sizeof(VxVector4));
            rootEntity->WorldMatrixChanged(FALSE, TRUE);

            // Compute perspective projection based on camera bounding box
            VxVector camPos;
            VxVector rootPos;
            rootEntity->GetPosition(&rootPos, nullptr);
            camera->InverseTransform(&camPos, &rootPos, nullptr);

            float z = rc->m_NearPlane / fabsf(camPos.z);
            float right = (camera->m_LocalBoundingBox.Max.x - camPos.x) * z;
            float left = (camera->m_LocalBoundingBox.Min.x - camPos.x) * z;
            float top = (camera->m_LocalBoundingBox.Max.y - camPos.y) * z;
            float bottom = (camera->m_LocalBoundingBox.Min.y - camPos.y) * z;

            rc->m_ProjectionMatrix.PerspectiveRect(left, right, top, bottom, rc->m_NearPlane, rc->m_FarPlane);
            rst->SetTransformMatrix(VXMATRIX_PROJECTION, rc->m_ProjectionMatrix);
            rst->SetViewport(&rc->m_ViewportData);

            // Update 2D root rects
            VxRect fullRect(0.0f, 0.0f,
                            (float) rc->m_Settings.m_Rect.right,
                            (float) rc->m_Settings.m_Rect.bottom);
            CK2dEntity *root2DBack = rc->Get2dRoot(TRUE);
            CK2dEntity *root2DFore = rc->Get2dRoot(FALSE);
            if (root2DBack) root2DBack->SetRect(fullRect);
            if (root2DFore) root2DFore->SetRect(fullRect);
        }

        // Build view frustum
        float aspectRatio = (float) rc->m_ViewportData.ViewWidth / (float) rc->m_ViewportData.ViewHeight;
        VxVector *origin = (VxVector *) &rootEntity->m_WorldMatrix[3];
        VxVector *vright = (VxVector *) &rootEntity->m_WorldMatrix[0];
        VxVector *up = (VxVector *) &rootEntity->m_WorldMatrix[1];
        VxVector *dir = (VxVector *) &rootEntity->m_WorldMatrix[2];

        VxFrustum frustum(*origin, *vright, *up, *dir,
                          rc->m_NearPlane, rc->m_FarPlane, rc->m_Fov, aspectRatio);
        rc->m_Frustum = frustum;

        rst->SetTransformMatrix(VXMATRIX_WORLD, VxMatrix::Identity());
        rst->SetTransformMatrix(VXMATRIX_VIEW, rootEntity->GetInverseWorldMatrix());

        SetupLights(rst);

        if (!camera) {
            rc->UpdateProjection(FALSE);
        }

        // Execute pre-render callbacks (m_PreCallBacks)
        rc->m_DevicePreCallbacksTimeProfiler.Reset();
        if (rc->m_PreRenderCallBacks.m_PreCallBacks.Size() > 0) {
            VxCallBack *it = rc->m_PreRenderCallBacks.m_PreCallBacks.Begin();
            rst->SetVertexShader(0);
            while (it < rc->m_PreRenderCallBacks.m_PreCallBacks.End()) {
                ((CK_RENDERCALLBACK) it->callback)((CKRenderContext *) rc, it->argument);
                ++it;
            }
        }
        rc->m_Stats.DevicePreCallbacks += rc->m_DevicePreCallbacksTimeProfiler.Current();

        rst->SetVertexShader(0);
        m_Context->ExecuteManagersOnPreRender((CKRenderContext *) rc);

        // Compute render flags for scene traversal (matches original CK2_3D.dll behavior)
        // a3 = CK_RENDER_DEFAULTSETTINGS + ((Flags & CK_RENDER_DONOTUPDATEEXTENTS) ? 1 : 0)
        CKDWORD renderFlags = (Flags & CK_RENDER_DONOTUPDATEEXTENTS)
                                  ? CK_RENDER_CLEARVIEWPORT
                                  : CK_RENDER_DEFAULTSETTINGS;

        rc->m_ObjectsRenderTimeProfiler.Reset();
        rc->m_TransparentObjects.Resize(0);
        rc->m_Stats.SceneTraversalTime = 0.0f;
        rc->m_SceneTraversalTimeProfiler.Reset();

        rm->m_SceneGraphRootNode.RenderTransparentObjects(rc, renderFlags);

        rc->m_Stats.SceneTraversalTime += rc->m_SceneTraversalTimeProfiler.Current();

        rc->CallSprite3DBatches();

        // Execute post-render temp callbacks (m_PostRenderCallBacks.m_PostCallBacks)
        rc->m_DevicePostCallbacksTimeProfiler.Reset();
        if (rc->m_PostRenderCallBacks.m_PostCallBacks.Size() > 0) {
            VxCallBack *it = rc->m_PostRenderCallBacks.m_PostCallBacks.Begin();
            rst->SetVertexShader(0);
            while (it < rc->m_PostRenderCallBacks.m_PostCallBacks.End()) {
                ((CK_RENDERCALLBACK) it->callback)((CKRenderContext *) rc, it->argument);
                ++it;
            }
        }
        rc->m_Stats.DevicePostCallbacks += rc->m_DevicePostCallbacksTimeProfiler.Current();

        // Sort and render transparent objects
        rc->m_SortTransparentObjects = TRUE;
        rm->m_SceneGraphRootNode.SortTransparentObjects(rc, renderFlags);
        rc->CallSprite3DBatches();
        rc->m_SortTransparentObjects = FALSE;

        rc->m_Stats.ObjectsRenderTime = rc->m_ObjectsRenderTimeProfiler.Current();

        // Execute post-render callbacks (m_PreRenderCallBacks.m_PostCallBacks)
        rc->m_DevicePostCallbacksTimeProfiler.Reset();
        if (rc->m_PreRenderCallBacks.m_PostCallBacks.Size() > 0) {
            VxCallBack *it = rc->m_PreRenderCallBacks.m_PostCallBacks.Begin();
            rst->SetVertexShader(0);
            while (it < rc->m_PreRenderCallBacks.m_PostCallBacks.End()) {
                ((CK_RENDERCALLBACK) it->callback)((CKRenderContext *) rc, it->argument);
                ++it;
            }
        }
        rc->m_Stats.DevicePostCallbacks += rc->m_DevicePostCallbacksTimeProfiler.Current();

        m_Context->ExecuteManagersOnPostRender(rc);
    }

    rc->m_SpriteTimeProfiler.Reset();

    // Render foreground 2D sprites
    if ((Flags & CK_RENDER_FOREGROUNDSPRITES) != 0 && rm->m_2DRootFore->GetChildrenCount() > 0) {
        VxRect viewRect;
        rc->GetViewRect(viewRect);

        rc->SetFullViewport(&rst->m_ViewportData,
                            rc->m_Settings.m_Rect.right,
                            rc->m_Settings.m_Rect.bottom);
        rst->SetViewport(&rst->m_ViewportData);

        ((RCK2dEntity *) rm->m_2DRootFore)->Render(rc);

        ResizeViewport(viewRect);
    }

    rc->m_Stats.SpriteTime += rc->m_SpriteTimeProfiler.Current();

    rst->SetVertexShader(0);
    m_Context->ExecuteManagersOnPostSpriteRender(rc);

    // Execute final post-render callbacks (m_PostSpriteRenderCallBacks.m_PostCallBacks)
    rc->m_DevicePostCallbacksTimeProfiler.Reset();
    if (rc->m_PostSpriteRenderCallBacks.m_PostCallBacks.Size() > 0) {
        VxCallBack *it = rc->m_PostSpriteRenderCallBacks.m_PostCallBacks.Begin();
        rst->SetVertexShader(0);
        while (it < rc->m_PostSpriteRenderCallBacks.m_PostCallBacks.End()) {
            ((CK_RENDERCALLBACK) it->callback)((CKRenderContext *) rc, it->argument);
            ++it;
        }
    }
    rc->m_Stats.DevicePostCallbacks += rc->m_DevicePostCallbacksTimeProfiler.Current();

    rc->m_Stats.ObjectsRenderTime -= (rc->m_Stats.SceneTraversalTime +
        rc->m_Stats.TransparentObjectsSortTime +
        rc->m_Stats.ObjectsCallbacksTime +
        rc->m_Stats.SkinTime);
    rc->m_Stats.SpriteTime -= rc->m_Stats.SpriteCallbacksTime;

    g_UpdateTransparency = FALSE;

    return CK_OK;
}

void CKRenderedScene::SetupLights(CKRasterizerContext *rst) {
    // IDA: 0x1006fea0
    for (CKDWORD i = 0; i < m_LightCount; ++i) {
        rst->EnableLight(i, FALSE);
    }
    m_LightCount = 0;

    for (CKObject **it = m_Lights.Begin(); it < m_Lights.End(); ++it) {
        RCKLight *light = (RCKLight *) *it;
        if (light->Setup(rst, m_LightCount)) {
            ++m_LightCount;
        }
    }

    rst->SetRenderState(VXRENDERSTATE_AMBIENT, m_AmbientLight);
}

void CKRenderedScene::ResizeViewport(const VxRect &rect) {
    CKRasterizerContext *rst = ((RCKRenderContext *) m_RenderContext)->m_RasterizerContext;
    rst->m_ViewportData.ViewX = (int) rect.left;
    rst->m_ViewportData.ViewY = (int) rect.top;
    rst->m_ViewportData.ViewWidth = (int) rect.GetWidth();
    rst->m_ViewportData.ViewHeight = (int) rect.GetHeight();
    rst->SetViewport(&rst->m_ViewportData);
}

void CKRenderedScene::SetDefaultRenderStates(CKRasterizerContext *rst) {
    RCKRenderContext *rc = (RCKRenderContext *) m_RenderContext;
    RCKRenderManager *rm = rc->m_RenderManager;

    rst->InvalidateStateCache(VXRENDERSTATE_FOGVERTEXMODE);
    rst->InvalidateStateCache(VXRENDERSTATE_FOGENABLE);
    rst->InvalidateStateCache(VXRENDERSTATE_FOGPIXELMODE);
    rst->SetRenderState(VXRENDERSTATE_FOGENABLE, m_FogMode != VXFOG_NONE);

    if (m_FogMode) {
        if (rm->m_ForceLinearFog.Value != 0) {
            m_FogMode = VXFOG_LINEAR;
        }

        if ((rc->m_RasterizerDriver->m_3DCaps.RasterCaps & (CKRST_RASTERCAPS_FOGRANGE | CKRST_RASTERCAPS_FOGPIXEL)) == (CKRST_RASTERCAPS_FOGRANGE | CKRST_RASTERCAPS_FOGPIXEL)) {
            rst->SetRenderState(VXRENDERSTATE_FOGPIXELMODE, m_FogMode);
        } else {
            m_FogMode = VXFOG_LINEAR;
            rst->SetRenderState(VXRENDERSTATE_FOGVERTEXMODE, m_FogMode);
        }

        const VxMatrix &projMat = rst->m_ProjectionMatrix;

        float endZ = projMat[2][2] * m_FogEnd + projMat[3][2];
        float endW = projMat[2][3] * m_FogEnd + projMat[3][3];

        float startZ = projMat[2][2] * m_FogStart + projMat[3][2];
        float startW = projMat[2][3] * m_FogStart + projMat[3][3];

        float projFogEnd = endZ / endW;
        float projFogStart = startZ / startW;
        float recipStartW = 1.0f / startW;

        if (g_FogProjectionMode == 0) {
            rst->SetRenderState(VXRENDERSTATE_FOGEND, *reinterpret_cast<CKDWORD *>(&m_FogEnd));
            rst->SetRenderState(VXRENDERSTATE_FOGSTART, *reinterpret_cast<CKDWORD *>(&m_FogStart));
        } else if (g_FogProjectionMode == 1) {
            rst->SetRenderState(VXRENDERSTATE_FOGEND, *reinterpret_cast<CKDWORD *>(&projFogEnd));
            rst->SetRenderState(VXRENDERSTATE_FOGSTART, *reinterpret_cast<CKDWORD *>(&projFogStart));
        } else if (g_FogProjectionMode == 2) {
            rst->SetRenderState(VXRENDERSTATE_FOGEND, *reinterpret_cast<CKDWORD *>(&projFogStart));
            rst->SetRenderState(VXRENDERSTATE_FOGSTART, *reinterpret_cast<CKDWORD *>(&recipStartW));
        }

        rst->SetRenderState(VXRENDERSTATE_FOGDENSITY, *reinterpret_cast<CKDWORD *>(&m_FogDensity));
        rst->SetRenderState(VXRENDERSTATE_FOGCOLOR, m_FogColor);
    }

    if (rm->m_DisableSpecular.Value != 0) {
        rst->SetRenderStateFlags(VXRENDERSTATE_SPECULARENABLE, FALSE);
        rst->SetRenderState(VXRENDERSTATE_SPECULARENABLE, FALSE);
        rst->SetRenderStateFlags(VXRENDERSTATE_SPECULARENABLE, TRUE);
    } else {
        rst->SetRenderStateFlags(VXRENDERSTATE_SPECULARENABLE, FALSE);
        rst->SetRenderState(VXRENDERSTATE_SPECULARENABLE, TRUE);
    }

    if (rm->m_DisableDithering.Value != 0) {
        rst->SetRenderStateFlags(VXRENDERSTATE_DITHERENABLE, FALSE);
        rst->SetRenderState(VXRENDERSTATE_DITHERENABLE, FALSE);
        rst->SetRenderStateFlags(VXRENDERSTATE_DITHERENABLE, TRUE);
    } else {
        rst->SetRenderStateFlags(VXRENDERSTATE_DITHERENABLE, FALSE);
        rst->InvalidateStateCache(VXRENDERSTATE_DITHERENABLE);
        rst->SetRenderState(VXRENDERSTATE_DITHERENABLE, TRUE);
    }

    if (rm->m_DisablePerspectiveCorrection.Value != 0) {
        rst->SetRenderStateFlags(VXRENDERSTATE_TEXTUREPERSPECTIVE, FALSE);
        rst->SetRenderState(VXRENDERSTATE_TEXTUREPERSPECTIVE, FALSE);
        rst->SetRenderStateFlags(VXRENDERSTATE_TEXTUREPERSPECTIVE, TRUE);
    } else {
        rst->SetRenderStateFlags(VXRENDERSTATE_TEXTUREPERSPECTIVE, FALSE);
        rst->SetRenderState(VXRENDERSTATE_TEXTUREPERSPECTIVE, TRUE);
    }

    rst->m_PresentInterval = rm->m_DisableFilter.Value;
    rst->m_CurrentPresentInterval = rm->m_DisableMipmap.Value;
    rst->SetRenderState(VXRENDERSTATE_NORMALIZENORMALS, TRUE);
    rst->SetRenderState(VXRENDERSTATE_ZENABLE, TRUE);
    rst->SetRenderState(VXRENDERSTATE_CULLMODE, VXCULL_CCW);
    rst->SetRenderState(VXRENDERSTATE_ZFUNC, VXCMP_LESSEQUAL);

    if (rc->m_Shading < 2) {
        if (rc->m_Shading) {
            rst->SetRenderStateFlags(VXRENDERSTATE_FILLMODE, FALSE);
            rst->SetRenderStateFlags(VXRENDERSTATE_SHADEMODE, FALSE);
            rst->SetRenderState(VXRENDERSTATE_SHADEMODE, rc->m_Shading);
            rst->SetRenderStateFlags(VXRENDERSTATE_SHADEMODE, TRUE);
        } else {
            rst->SetRenderStateFlags(VXRENDERSTATE_SHADEMODE, FALSE);
            rst->SetRenderStateFlags(VXRENDERSTATE_FILLMODE, FALSE);
            rst->SetRenderState(VXRENDERSTATE_FILLMODE, VXFILL_WIREFRAME);
            rst->SetRenderStateFlags(VXRENDERSTATE_FILLMODE, TRUE);
        }
    } else {
        rst->SetRenderStateFlags(VXRENDERSTATE_SHADEMODE, FALSE);
        rst->SetRenderStateFlags(VXRENDERSTATE_FILLMODE, FALSE);
    }
}

void CKRenderedScene::PrepareCameras(CK_RENDER_FLAGS flags) {
    RCKRenderContext *rc = (RCKRenderContext *) m_RenderContext;

    for (CKObject **it = m_Cameras.Begin(); it != m_Cameras.End(); ++it) {
        CKCamera *camera = (CKCamera *) *it;
        if (camera && camera->GetClassID() == CKCID_TARGETCAMERA) {
            CK3dEntity *target = camera->GetTarget();
            if (target) {
                VxVector origin(0.0f, 0.0f, 0.0f);
                camera->LookAt(&origin, target, FALSE);
            }
        }
    }

    for (CKObject **it = m_Lights.Begin(); it != m_Lights.End(); ++it) {
        CKLight *light = (CKLight *) *it;
        if (light && light->GetClassID() == CKCID_TARGETLIGHT && light->GetType() != VX_LIGHTPOINT) {
            CK3dEntity *target = light->GetTarget();
            if (target) {
                VxVector origin(0.0f, 0.0f, 0.0f);
                light->LookAt(&origin, target, FALSE);
            }
        }
    }

    if (m_AttachedCamera && m_RootEntity) {
        RCKCamera *cam = (RCKCamera *) m_AttachedCamera;
        RCK3dEntity *rootEntity = (RCK3dEntity *) m_RootEntity;

        std::memcpy(&rootEntity->m_WorldMatrix, &cam->m_WorldMatrix, sizeof(rootEntity->m_WorldMatrix));
        ((RCK3dEntity *) m_RootEntity)->WorldMatrixChanged(TRUE, FALSE);

        if (!cam->IsUpToDate()) {
            rc->m_Perspective = (cam->GetProjectionType() == CK_PERSPECTIVEPROJECTION);
            rc->m_Zoom = cam->GetOrthographicZoom();
            rc->m_Fov = cam->GetFov();
            rc->m_NearPlane = cam->GetFrontPlane();
            rc->m_FarPlane = cam->GetBackPlane();

            if ((flags & CK_RENDER_USECAMERARATIO) != 0) {
                UpdateViewportSize(TRUE, flags);
            } else {
                rc->UpdateProjection(TRUE);
            }

            cam->ModifyObjectFlags(CK_OBJECT_UPTODATE, 0);
        } else {
            rc->UpdateProjection(FALSE);
        }
    } else {
        rc->UpdateProjection(FALSE);
    }
}

void CKRenderedScene::UpdateViewportSize(int forceUpdate, CK_RENDER_FLAGS Flags) {
    // IDA: 0x10071381
    RCKRenderContext *rc = (RCKRenderContext *) m_RenderContext;

    VxRect windowRect;
    rc->GetWindowRect(windowRect, FALSE);

    if (Flags == 0) {
        Flags = (CK_RENDER_FLAGS) rc->GetCurrentRenderOptions();
    }

    int width = 4;
    int height = 3;
    if (m_AttachedCamera) {
        m_AttachedCamera->GetAspectRatio(width, height);
    }

    int right = rc->m_Settings.m_Rect.right;
    int bottom = rc->m_Settings.m_Rect.bottom;

    int viewX = 0;
    int viewY = 0;
    int viewWidth = right;
    int viewHeight = bottom;

    double rcAspect = static_cast<double>(right) / static_cast<double>(bottom);
    double camAspect = static_cast<double>(width) / static_cast<double>(height);

    if ((Flags & CK_RENDER_USECAMERARATIO) != 0) {
        if (rcAspect >= camAspect) {
            viewHeight = bottom;
            viewWidth = width * bottom / height;
        } else {
            viewWidth = right;
            viewHeight = height * right / width;
        }

        viewX = static_cast<int>(static_cast<double>(right - viewWidth) * 0.5);
        viewY = static_cast<int>(static_cast<double>(bottom - viewHeight) * 0.5);
    } else {
        viewX = 0;
        viewY = 0;
        viewWidth = right;
        viewHeight = bottom;
    }

    if ((rc->m_RenderFlags & CK_RENDER_USECAMERARATIO) != 0) {
        if (m_AttachedCamera) {
            if ((m_AttachedCamera->GetFlags() & CK_3DENTITY_CAMERAIGNOREASPECT) != 0) {
                if (forceUpdate) {
                    rc->UpdateProjection(TRUE);
                }
            } else if (rc->m_ViewportData.ViewHeight == viewHeight &&
                rc->m_ViewportData.ViewWidth == viewWidth &&
                rc->m_ViewportData.ViewX == viewX &&
                rc->m_ViewportData.ViewY == viewY) {
                if (forceUpdate) {
                    rc->UpdateProjection(TRUE);
                }
            } else {
                rc->m_ViewportData.ViewHeight = viewHeight;
                rc->m_ViewportData.ViewWidth = viewWidth;
                rc->m_ViewportData.ViewX = viewX;
                rc->m_ViewportData.ViewY = viewY;
                rc->UpdateProjection(TRUE);
            }
        } else if (forceUpdate) {
            rc->UpdateProjection(TRUE);
        }
    } else if (forceUpdate) {
        rc->UpdateProjection(TRUE);
    }
}

void CKRenderedScene::ForceCameraSettingsUpdate() {
    // IDA: 0x10070ffd
    if (!m_AttachedCamera)
        return;
    if (!m_RootEntity)
        return;

    RCKRenderContext *rc = (RCKRenderContext *) m_RenderContext;
    RCKCamera *cam = (RCKCamera *) m_AttachedCamera;

    // Set root entity's world matrix from camera
    m_RootEntity->SetWorldMatrix(cam->GetWorldMatrix(), FALSE);

    // Update render context settings from camera
    rc->m_Perspective = (cam->GetProjectionType() == 1);
    rc->m_Zoom = cam->GetOrthographicZoom();
    rc->m_Fov = cam->GetFov();
    rc->m_NearPlane = cam->GetFrontPlane();
    rc->m_FarPlane = cam->GetBackPlane();

    // Update viewport size
    UpdateViewportSize(FALSE, CK_RENDER_USECURRENTSETTINGS);
}
