#include "CKScene.h"

#include "CKAttributeManager.h"
#include "CKStateChunk.h"
#include "CKLevel.h"
#include "CKBehavior.h"
#include "CKRenderContext.h"
#include "CKTexture.h"
#include "CKMaterial.h"
#include "CKCamera.h"
#include "CKPlace.h"
#include "CKFile.h"

CK_CLASSID CKScene::m_ClassID = CKCID_SCENE;

void CKScene::AddObjectToScene(CKSceneObject *o, CKBOOL dependencies) {
    if (!o)
        return;
    if (CKIsChildClassOf(o, CKCID_BEOBJECT)) {
        CKBeObject *beo = (CKBeObject *) o;
        beo->AddToScene(this, dependencies);
    } else {
        AddObject(o);
    }
}

void CKScene::RemoveObjectFromScene(CKSceneObject *o, CKBOOL dependencies) {
    if (!o)
        return;
    if (CKIsChildClassOf(o, CKCID_BEOBJECT)) {
        CKBeObject *beo = (CKBeObject *) o;
        beo->RemoveFromScene(this, dependencies);
    } else {
        RemoveObject(o);
    }
}

CKBOOL CKScene::IsObjectHere(CKObject *o) {
    if (!o)
        return FALSE;
    if (CKIsChildClassOf(o, CKCID_SCENEOBJECT)) {
        CKSceneObject *so = (CKSceneObject *) o;
        return so->IsInScene(this);
    }
    return FALSE;
}

void CKScene::BeginAddSequence(CKBOOL Begin) {
    if (Begin) {
        if (m_AddObjectCount == 0) {
            m_AddObjectList.Resize(0);
        }
        ++m_AddObjectCount;
    } else if (--m_AddObjectCount <= 0) {
        const int objectCount = m_AddObjectList.Size();
        if (objectCount > 0) {
            m_Context->ExecuteManagersOnSequenceAddedToScene(this, m_AddObjectList.Begin(), objectCount);
        }

        m_AddObjectList.Resize(0);
        m_AddObjectCount = 0;
    }
}

void CKScene::BeginRemoveSequence(CKBOOL Begin) {
    if (Begin) {
        if (m_RemoveObjectCount == 0) {
            m_RemoveObjectList.Resize(0);
        }
        ++m_RemoveObjectCount;
    } else if (--m_RemoveObjectCount <= 0) {
        const int objectCount = m_RemoveObjectList.Size();
        if (objectCount > 0) {
            m_Context->ExecuteManagersOnSequenceRemovedFromScene(this, m_RemoveObjectList.Begin(), objectCount);
        }

        m_RemoveObjectList.Resize(0);
        m_RemoveObjectCount = 0;
    }
}

int CKScene::GetObjectCount() {
    return m_SceneObjects.Size();
}

void CKScene::AddObject(CKSceneObject *o) {
    if (!o)
        return;

    // Do not add scenes to themselves or if they are already in the scene.
    if (o->GetClassID() == CKCID_SCENE || GetSceneObjectDesc(o))
        return;

    CKSceneObjectDesc *desc = AddObjectDesc(o);
    if (!desc)
        return;

    // For single-activity objects, inherit flags from the other scene's descriptor.
    // m_SingleObjectActivities stores the flags from the scene where the object is already active.
    CK_ID flags = 0;
    if (m_Context->m_ObjectManager->GetSingleObjectActivity(o, flags)) {
        // If INTERNAL_IC flag is set, save initial state and clear the flag
        if (flags & CK_SCENEOBJECT_INTERNAL_IC) {
            flags &= ~CK_SCENEOBJECT_INTERNAL_IC;
            desc->m_InitialValue = CKSaveObjectState(o, CK_STATESAVE_ALL);
        }

        // Store flags with ACTIVE bit cleared (object starts inactive until explicitly activated)
        desc->m_Flags = flags & ~CK_SCENEOBJECT_ACTIVE;

        // Handle initial activation state based on inherited flags
        if (flags & CK_SCENEOBJECT_START_ACTIVATE) {
            Activate(o, (flags & CK_SCENEOBJECT_START_RESET) != 0);
        } else if (flags & CK_SCENEOBJECT_START_DEACTIVATE) {
            DeActivate(o);
        }
    }

    if (m_AddObjectCount > 0) {
        m_AddObjectList.AddIfNotHere(o->GetID());
    } else {
        CK_ID objId = o->GetID();
        m_Context->ExecuteManagersOnSequenceAddedToScene(this, &objId, 1);
    }
}

void CKScene::RemoveObject(CKSceneObject *o) {
    if (!o)
        return;

    CKSceneObjectDesc *desc = GetSceneObjectDesc(o);
    if (!desc)
        return;

    desc->Clear();
    m_SceneObjects.Remove(o->GetID());

    o->RemoveSceneIn(this);

    if (m_RemoveObjectCount > 0) {
        m_RemoveObjectList.PushBack(o->GetID());
    } else {
        CK_ID id = o->GetID();
        m_Context->ExecuteManagersOnSequenceRemovedFromScene(this, &id, 1);
    }
}

void CKScene::RemoveAllObjects() {
    // RemoveAllObjects does not access CKContext nor touch scene membership.
    // Relationship cleanup (RemoveSceneIn) is handled in PreDelete().
    for (CKSODHashIt it = m_SceneObjects.Begin(); it != m_SceneObjects.End(); ++it) {
        CKSceneObjectDesc &desc = *it;
        desc.Clear();
    }
    m_SceneObjects.Clear();
}

const XObjectPointerArray &CKScene::ComputeObjectList(CK_CLASSID cid, CKBOOL derived) {
    m_ObjectList.Resize(0);

    for (CKSODHashIt it = m_SceneObjects.Begin(); it != m_SceneObjects.End(); ++it) {
        CKObject *obj = m_Context->GetObject(it.GetKey());
        if (obj) {
            if (derived ? CKIsChildClassOf(obj, cid) : (obj->GetClassID() == cid)) {
                m_ObjectList.PushBack(obj);
            }
        }
    }
    return m_ObjectList;
}

CKERROR CKScene::ComputeObjectList(CKObjectArray *array, CK_CLASSID cid, CKBOOL derived) {
    if (!array)
        return CKERR_INVALIDPARAMETER;
    for (CKSODHashIt it = m_SceneObjects.Begin(); it != m_SceneObjects.End(); ++it) {
        CKObject *obj = m_Context->GetObject(it.GetKey());
        if (obj) {
            if (derived ? CKIsChildClassOf(obj, cid) : (obj->GetClassID() == cid)) {
                array->InsertRear(obj);
            }
        }
    }
    return CK_OK;
}

CKSceneObjectIterator CKScene::GetObjectIterator() {
    return CKSceneObjectIterator(m_SceneObjects.Begin());
}

CKSceneObjectDesc *CKScene::GetSceneObjectDesc(CKSceneObject *o) {
    if (!o || !o->IsInScene(this))
        return nullptr;

    auto it = m_SceneObjects.Find(o->GetID());
    if (it == m_SceneObjects.End())
        return nullptr;
    return &(*it);
}

void CKScene::Activate(CKSceneObject *o, CKBOOL Reset) {
    if (!o) return;

    if (CKIsChildClassOf(o, CKCID_BEHAVIOR)) {
        ActivateScript((CKBehavior *) o, TRUE, Reset);
    } else if (CKIsChildClassOf(o, CKCID_BEOBJECT)) {
        ActivateBeObject((CKBeObject *) o, TRUE, Reset);
    }
}

void CKScene::DeActivate(CKSceneObject *o) {
    if (!o) return;

    if (CKIsChildClassOf(o, CKCID_BEHAVIOR)) {
        ActivateScript((CKBehavior *) o, FALSE, FALSE);
    } else if (CKIsChildClassOf(o, CKCID_BEOBJECT)) {
        ActivateBeObject((CKBeObject *) o, FALSE, FALSE);
    }
}

void CKScene::ActivateBeObject(CKBeObject *beo, CKBOOL activate, CKBOOL reset) {
    if (!beo) return;

    const int scriptCount = beo->GetScriptCount();
    CKSceneObjectDesc *desc = GetSceneObjectDesc(beo);
    if (!desc) return;

    CKScene *currentScene = m_Context->GetCurrentScene();

    if (activate) {
        // If not current scene OR already active, just set the flag
        if (currentScene != this || (desc->m_Flags & CK_SCENEOBJECT_ACTIVE)) {
            desc->m_Flags |= CK_SCENEOBJECT_ACTIVE;
        } else {
            // Current scene AND not yet active: set flag FIRST, then process
            desc->m_Flags |= CK_SCENEOBJECT_ACTIVE;

            if (scriptCount > 0 || CKIsChildClassOf(beo, CKCID_CHARACTER))
                m_Context->GetBehaviorManager()->AddObjectNextFrame(beo);

            if (reset && (desc->m_Flags & CK_SCENEOBJECT_START_RESET)) {
                CKStateChunk *initialValue = desc->m_InitialValue;
                if (initialValue)
                    beo->Load(initialValue, nullptr);
            }

            for (int i = 0; i < scriptCount; ++i) {
                CKBehavior *script = beo->GetScript(i);
                if (script)
                    script->CallSubBehaviorsCallbackFunction(CKM_BEHAVIORACTIVATESCRIPT);
            }
        }
    } else {
        // Deactivate
        if (currentScene == this && (desc->m_Flags & CK_SCENEOBJECT_ACTIVE)) {
            // Current scene AND currently active: clear flag FIRST, then process
            desc->m_Flags &= ~CK_SCENEOBJECT_ACTIVE;

            if (scriptCount > 0 || CKIsChildClassOf(beo, CKCID_CHARACTER))
                m_Context->GetBehaviorManager()->RemoveObjectNextFrame(beo);

            for (int i = 0; i < scriptCount; ++i) {
                CKBehavior *script = beo->GetScript(i);
                if (script)
                    script->CallSubBehaviorsCallbackFunction(CKM_BEHAVIORDEACTIVATESCRIPT);
            }
        } else {
            // Not current scene OR not active: just clear the flag
            desc->m_Flags &= ~CK_SCENEOBJECT_ACTIVE;
        }
    }
}

void CKScene::ActivateScript(CKBehavior *beh, CKBOOL activate, CKBOOL reset) {
    if (!beh || beh->GetType() != CKBEHAVIORTYPE_SCRIPT) return;

    CKSceneObjectDesc *desc = GetSceneObjectDesc(beh);
    if (!desc)
        return;

    if (activate) {
        if (m_Context->GetCurrentScene() == this) {
            CKBeObject *owner = beh->GetOwner();
            if (owner && (owner->IsActiveInCurrentScene() || owner->GetClassID() == CKCID_LEVEL))
                m_Context->GetBehaviorManager()->AddObjectNextFrame(owner);
            CKDWORD addFlags = CKBEHAVIOR_ACTIVATENEXTFRAME;
            if (reset)
                addFlags |= CKBEHAVIOR_RESETNEXTFRAME;
            beh->ModifyFlags(addFlags, CKBEHAVIOR_DEACTIVATENEXTFRAME);
            // Only set flag and call callback if not already active
            if (!(desc->m_Flags & CK_SCENEOBJECT_ACTIVE)) {
                desc->m_Flags |= CK_SCENEOBJECT_ACTIVE;
                beh->CallSubBehaviorsCallbackFunction(CKM_BEHAVIORACTIVATESCRIPT);
            }
        } else {
            // Not current scene - just mark as active, no callback
            desc->m_Flags |= CK_SCENEOBJECT_ACTIVE;
        }
    } else {
        if (m_Context->GetCurrentScene() == this) {
            beh->ModifyFlags(CKBEHAVIOR_DEACTIVATENEXTFRAME, CKBEHAVIOR_ACTIVATENEXTFRAME | CKBEHAVIOR_RESETNEXTFRAME);
            // Only clear flag and call callback if currently active
            if (desc->m_Flags & CK_SCENEOBJECT_ACTIVE) {
                desc->m_Flags &= ~CK_SCENEOBJECT_ACTIVE;
                beh->CallSubBehaviorsCallbackFunction(CKM_BEHAVIORDEACTIVATESCRIPT);
            }
        } else {
            // Not current scene - just mark as inactive, no callback
            desc->m_Flags &= ~CK_SCENEOBJECT_ACTIVE;
        }
    }
}

void CKScene::SetObjectFlags(CKSceneObject *o, CK_SCENEOBJECT_FLAGS flags) {
    CKSceneObjectDesc *desc = GetSceneObjectDesc(o);
    if (desc)
        desc->m_Flags = flags;
}

CK_SCENEOBJECT_FLAGS CKScene::GetObjectFlags(CKSceneObject *o) {
    CKSceneObjectDesc *desc = GetSceneObjectDesc(o);
    if (desc)
        return (CK_SCENEOBJECT_FLAGS) desc->m_Flags;
    return (CK_SCENEOBJECT_FLAGS) 0;
}

CK_SCENEOBJECT_FLAGS CKScene::ModifyObjectFlags(CKSceneObject *o, CKDWORD Add, CKDWORD Remove) {
    CKSceneObjectDesc *desc = GetSceneObjectDesc(o);
    if (!desc)
        return (CK_SCENEOBJECT_FLAGS) 0;
    CKDWORD flags = desc->m_Flags;
    flags |= Add;
    flags &= ~Remove;
    desc->m_Flags = flags;
    return (CK_SCENEOBJECT_FLAGS) flags;
}

CKBOOL CKScene::SetObjectInitialValue(CKSceneObject *o, CKStateChunk *chunk) {
    CKSceneObjectDesc *desc = GetSceneObjectDesc(o);
    if (!desc)
        return FALSE;
    delete desc->m_InitialValue;
    if (chunk) chunk->CloseChunk();
    desc->m_InitialValue = chunk;
    return TRUE;
}

CKStateChunk *CKScene::GetObjectInitialValue(CKSceneObject *o) {
    CKSceneObjectDesc *desc = GetSceneObjectDesc(o);
    if (!desc)
        return nullptr;
    return desc->m_InitialValue;
}

CKBOOL CKScene::IsObjectActive(CKSceneObject *o) {
    if (o == this)
        return TRUE;
    CKSceneObjectDesc *desc = GetSceneObjectDesc(o);
    if (!desc)
        return FALSE;
    return desc->IsActive();
}

void CKScene::ApplyEnvironmentSettings(XObjectPointerArray *renderContexts) {
    XObjectPointerArray *tempArray = nullptr;

    if (!renderContexts) {
        CKLevel *level = m_Context->GetCurrentLevel();
        if (!level)
            return;

        tempArray = new XObjectPointerArray();

        const int count = level->GetRenderContextCount();
        for (int i = 0; i < count; ++i) {
            CKRenderContext *rc = level->GetRenderContext(i);
            if (rc) {
                tempArray->PushBack(rc);
            }
        }

        renderContexts = tempArray;
    }

    for (XObjectPointerArray::Iterator it = renderContexts->Begin(); it != renderContexts->End(); ++it) {
        CKRenderContext *rc = (CKRenderContext *) *it;
        if (!rc) continue;

        // 1. Attach viewpoint to starting camera
        CKCamera *startCam = GetStartingCamera();
        if (startCam) rc->AttachViewpointToCamera(startCam);

        // 2. Set background properties
        CKMaterial *bgMaterial = rc->GetBackgroundMaterial();
        if (bgMaterial) {
            VxColor bgColor(m_BackgroundColor);
            bgMaterial->SetDiffuse(bgColor);
            bgMaterial->SetTexture0(GetBackgroundTexture());
        }

        // 3. Configure fog settings
        rc->SetFogColor(m_FogColor);
        rc->SetFogMode(m_FogMode);
        rc->SetFogStart(m_FogStart);
        rc->SetFogEnd(m_FogEnd);
        rc->SetFogDensity(m_FogDensity);

        // 4. Set ambient lighting
        rc->SetAmbientLight(m_AmbientLightColor);
    }

    if (tempArray) delete tempArray;
}

void CKScene::UseEnvironmentSettings(CKBOOL use) {
    if (use)
        m_EnvironmentSettings |= CK_SCENE_USEENVIRONMENTSETTINGS;
    else
        m_EnvironmentSettings &= ~CK_SCENE_USEENVIRONMENTSETTINGS;
}

CKBOOL CKScene::EnvironmentSettings() {
    return (m_EnvironmentSettings & CK_SCENE_USEENVIRONMENTSETTINGS) != 0;
}

void CKScene::SetAmbientLight(CKDWORD Color) {
    m_AmbientLightColor = Color;
}

CKDWORD CKScene::GetAmbientLight() {
    return m_AmbientLightColor;
}

void CKScene::SetFogMode(VXFOG_MODE Mode) {
    m_FogMode = Mode;
}

void CKScene::SetFogStart(float Start) {
    m_FogStart = Start;
}

void CKScene::SetFogEnd(float End) {
    m_FogEnd = End;
}

void CKScene::SetFogDensity(float Density) {
    m_FogDensity = Density;
}

void CKScene::SetFogColor(CKDWORD Color) {
    m_FogColor = Color;
}

VXFOG_MODE CKScene::GetFogMode() {
    return m_FogMode;
}

float CKScene::GetFogStart() {
    return m_FogStart;
}

float CKScene::GetFogEnd() {
    return m_FogEnd;
}

float CKScene::GetFogDensity() {
    return m_FogDensity;
}

CKDWORD CKScene::GetFogColor() {
    return m_FogColor;
}

void CKScene::SetBackgroundColor(CKDWORD Color) {
    m_BackgroundColor = Color;
}

CKDWORD CKScene::GetBackgroundColor() {
    return m_BackgroundColor;
}

void CKScene::SetBackgroundTexture(CKTexture *texture) {
    m_BackgroundTexture = texture ? texture->GetID() : 0;
}

CKTexture *CKScene::GetBackgroundTexture() {
    return (CKTexture *) m_Context->GetObject(m_BackgroundTexture);
}

void CKScene::SetStartingCamera(CKCamera *camera) {
    if (camera && camera->IsInScene(this))
        m_StartingCamera = camera->GetID();
    else
        m_StartingCamera = 0;
}

CKCamera *CKScene::GetStartingCamera() {
    return (CKCamera *) m_Context->GetObject(m_StartingCamera);
}

CKLevel *CKScene::GetLevel() {
    return (CKLevel *) m_Context->GetObject(m_Level);
}

void CKScene::SetLevel(CKLevel *level) {
    m_Level = level ? level->GetID() : 0;
}

CKERROR CKScene::Merge(CKScene *mergedScene, CKLevel *fromLevel) {
    if (!mergedScene)
        return CKERR_INVALIDPARAMETER;

    CKMoveAllScripts(mergedScene, this);
    CKCopyAllAttributes(mergedScene, this);

    for (CKSODHashIt it = mergedScene->m_SceneObjects.Begin(); it != mergedScene->m_SceneObjects.End(); ++it) {
        CKSceneObjectDesc &desc = *it;
        CKSceneObject *obj = (CKSceneObject *) m_Context->GetObject(it.GetKey());

        if (obj && !obj->IsInScene(this)) {
            CKSceneObjectDesc *newDesc = AddObjectDesc(obj);
            if (!newDesc) continue;

            newDesc->m_Flags = desc.m_Flags;
            newDesc->m_InitialValue = desc.m_InitialValue;
            desc.m_InitialValue = nullptr;

            if (fromLevel) {
                CKLevel *currentLevel = m_Context->GetCurrentLevel();
                if (currentLevel && currentLevel != fromLevel && newDesc->m_InitialValue) {
                    newDesc->m_InitialValue->RemapObject(fromLevel->GetID(), currentLevel->GetID());
                }
            }
            if (newDesc->m_InitialValue) {
                newDesc->m_InitialValue->RemapObject(mergedScene->GetID(), GetID());
            }

            m_AddObjectList.PushBack(newDesc->m_Object);
        }
    }

    CKRemapObjectParameterValue(m_Context, mergedScene->GetID(), GetID());

    if (m_AddObjectCount == 0 && !m_AddObjectList.IsEmpty()) {
        m_Context->ExecuteManagersOnSequenceAddedToScene(this, m_AddObjectList.Begin(), m_AddObjectList.Size());
    }

    m_Context->DestroyObject(mergedScene);
    return CK_OK;
}

void CKScene::Init(XObjectPointerArray &renderContexts, CK_SCENEOBJECTACTIVITY_FLAGS activityFlags, CK_SCENEOBJECTRESET_FLAGS resetFlags) {
    if (GetScriptCount() > 0)
        m_Context->GetBehaviorManager()->AddObject(this);

    if (EnvironmentSettings())
        ApplyEnvironmentSettings(&renderContexts);

    for (XObjectPointerArray::Iterator rit = renderContexts.Begin(); rit != renderContexts.End(); ++rit) {
        CKRenderContext *rc = (CKRenderContext *)*rit;
        if (rc)
            rc->AddRemoveSequence(TRUE);
    }

    for (CKSceneObjectIterator it = GetObjectIterator(); !it.End(); it++) {
        CKSceneObjectDesc *desc = it.GetObjectDesc();
        CKSceneObject *obj = (CKSceneObject *) m_Context->GetObject(desc->m_Object);
        if (!obj) continue;

        if (CKIsChildClassOf(obj, CKCID_RENDEROBJECT)) {
            CKRenderObject *renderObj = (CKRenderObject *)obj;
            for (XObjectPointerArray::Iterator rit = renderContexts.Begin(); rit != renderContexts.End(); ++rit) {
                CKRenderContext *rc = (CKRenderContext *)*rit;
                if (rc)
                    rc->AddObject(renderObj);
            }
        }

        CKBOOL active = (activityFlags == CK_SCENEOBJECTACTIVITY_ACTIVATE);
        CKBOOL doNothing = (activityFlags == CK_SCENEOBJECTACTIVITY_DONOTHING);
        CKBOOL reset = (resetFlags == CK_SCENEOBJECTRESET_RESET);

        if (resetFlags == CK_SCENEOBJECTRESET_SCENEDEFAULT)
            reset = (desc->m_Flags & CK_SCENEOBJECT_START_RESET) != 0;

        if (activityFlags == CK_SCENEOBJECTACTIVITY_SCENEDEFAULT) {
            active = (desc->m_Flags & CK_SCENEOBJECT_START_ACTIVATE) != 0;
            doNothing = (desc->m_Flags & CK_SCENEOBJECT_START_LEAVE) != 0;
        }

        if (reset && desc->m_InitialValue)
            obj->Load(desc->m_InitialValue, nullptr);

        if (!doNothing) {
            if (active) desc->m_Flags |= CK_SCENEOBJECT_ACTIVE;
            else desc->m_Flags &= ~CK_SCENEOBJECT_ACTIVE;
        }

        if (CKIsChildClassOf(obj, CKCID_BEHAVIOR)) {
            CKBehavior *beh = (CKBehavior *) obj;
            if (beh->IsActive()) desc->m_Flags |= CK_SCENEOBJECT_ACTIVE;
            else desc->m_Flags &= ~CK_SCENEOBJECT_ACTIVE;
            beh->ModifyFlags(0, CKBEHAVIOR_ACTIVATENEXTFRAME | CKBEHAVIOR_RESETNEXTFRAME | CKBEHAVIOR_DEACTIVATENEXTFRAME);
            beh->Activate((desc->m_Flags & CK_SCENEOBJECT_ACTIVE) != 0, reset);
        }

        if (desc->m_Flags & CK_SCENEOBJECT_ACTIVE) {
            if (CKIsChildClassOf(obj, CKCID_BEOBJECT)) {
                CKBeObject *beo = (CKBeObject *) obj;
                if (beo->GetScriptCount() > 0)
                    m_Context->GetBehaviorManager()->AddObject(beo);
            }
        }
    }

    for (XObjectPointerArray::Iterator rit = renderContexts.Begin(); rit != renderContexts.End(); ++rit) {
        CKRenderContext *rc = (CKRenderContext *)*rit;
        rc->AddRemoveSequence(FALSE);
    }

    m_EnvironmentSettings |= CK_SCENE_LAUNCHEDONCE;
}

void CKScene::Stop(XObjectPointerArray &renderContexts, CKBOOL reset) {
    for (XObjectPointerArray::Iterator rit = renderContexts.Begin(); rit != renderContexts.End(); ++rit) {
        CKRenderContext *rc = (CKRenderContext *)*rit;
        if (rc)
            rc->AddRemoveSequence(TRUE);
    }

    for (CKSceneObjectIterator it = GetObjectIterator(); !it.End(); it++) {
        CKSceneObject *obj = (CKSceneObject *) m_Context->GetObject(it.GetObjectID());
        if (!obj) continue;
        if (CKIsChildClassOf(obj, CKCID_RENDEROBJECT)) {
            CKRenderObject *renderObj = (CKRenderObject *) obj;
            for (XObjectPointerArray::Iterator rit = renderContexts.Begin(); rit != renderContexts.End(); ++rit) {
                CKRenderContext *rc = (CKRenderContext *)*rit;
                if (rc)
                    rc->RemoveObject(renderObj);
            }

            if (obj->GetClassID() == CKCID_PLACE) {
                CKPlace *place = (CKPlace *) obj;
                for (int i = 0; i < place->GetChildrenCount(); ++i) {
                    CK3dEntity *child = place->GetChild(i);
                    for (XObjectPointerArray::Iterator rit = renderContexts.Begin(); rit != renderContexts.End(); ++rit) {
                        CKRenderContext *rc = (CKRenderContext *)*rit;
                        if (rc)
                            rc->RemoveObject(child);
                    }
                }
            }
        }
    }

    if (reset) {
        for (XObjectPointerArray::Iterator rit = renderContexts.Begin(); rit != renderContexts.End(); ++rit) {
            CKRenderContext *rc = (CKRenderContext *)*rit;
            if (rc)
                rc->AddRemoveSequence(FALSE);
        }
    }

}

void CKScene::Launch() {
    for (CKSceneObjectIterator it = GetObjectIterator(); !it.End(); it++) {
        CKObject *obj = m_Context->GetObject(it.GetObjectID());
        if (!obj) {
            CheckSceneObjectList();
            continue;
        }

        CKSceneObjectDesc *desc = it.GetObjectDesc();
        if (desc->m_Flags & CK_SCENEOBJECT_START_ACTIVATE) {
            desc->m_Flags |= CK_SCENEOBJECT_ACTIVE;
            if (CKIsChildClassOf(obj, CKCID_BEOBJECT)) {
                CKBeObject *beo = (CKBeObject *) obj;
                if (beo->GetScriptCount() > 0) m_Context->GetBehaviorManager()->AddObject(beo);
            }
        } else if (desc->m_Flags & CK_SCENEOBJECT_START_DEACTIVATE) {
            desc->m_Flags &= ~CK_SCENEOBJECT_ACTIVE;
        } else if ((desc->m_Flags & CK_SCENEOBJECT_START_LEAVE) && (desc->m_Flags & CK_SCENEOBJECT_ACTIVE)) {
            if (CKIsChildClassOf(obj, CKCID_BEOBJECT)) {
                CKBeObject *beo = (CKBeObject *) obj;
                if (beo->GetScriptCount() > 0) m_Context->GetBehaviorManager()->AddObject(beo);
            }
        }

        if ((desc->m_Flags & CK_SCENEOBJECT_START_RESET) && desc->m_InitialValue)
            obj->Load(desc->m_InitialValue, nullptr);
    }
}

CKScene::CKScene(CKContext *Context, CKSTRING name) : CKBeObject(Context, name) {
    m_Level = 0;
    m_Priority = 30000;
    m_EnvironmentSettings = CK_SCENE_USEENVIRONMENTSETTINGS;
    m_SceneGlobalIndex = m_Context->m_ObjectManager->GetSceneGlobalIndex();
    m_Scenes.Set(m_SceneGlobalIndex);
    m_BackgroundColor = 0;
    m_BackgroundTexture = 0;
    m_StartingCamera = 0;
    m_FogStart = 1.0;
    m_FogMode = VXFOG_NONE;
    m_FogColor = 0;
    m_FogDensity = 1.0;
    m_AddObjectCount = 0;
    m_RemoveObjectCount = 0;
    m_AmbientLightColor = 0x0F0F0F;
    m_FogEnd = 100.0;
}

CKScene::~CKScene() {
    RemoveAllObjects();
    m_Context->m_ObjectManager->ReleaseSceneGlobalIndex(m_SceneGlobalIndex);
    if (m_Context->m_BehaviorContext.CurrentScene == this)
        m_Context->m_BehaviorContext.CurrentScene = nullptr;
    if (m_Context->m_BehaviorContext.PreviousScene == this)
        m_Context->m_BehaviorContext.PreviousScene = nullptr;
}

CK_CLASSID CKScene::GetClassID() {
    return m_ClassID;
}

void CKScene::PreSave(CKFile *file, CKDWORD flags) {
    CKBeObject::PreSave(file, flags);
    for (CKSODHashIt it = m_SceneObjects.Begin(); it != m_SceneObjects.End(); ++it) {
        CKObject *obj = m_Context->GetObject(it.GetKey());
        if (obj) file->SaveObject(obj);
    }
}

CKStateChunk *CKScene::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_SCENE, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(CK_STATESAVE_SCENENEWDATA);

    CKLevel *level = GetLevel();
    chunk->WriteObject(level);

    XArray<CKSceneObjectDesc *> descArray;
    CKSceneObjectIterator oit = GetObjectIterator();
    int missingObjects = 0;

    while (!oit.End()) {
        CK_ID objID = oit.GetObjectID();
        CKObject *obj = m_Context->GetObject(objID);
        if (obj) {
            // Filter out hidden and non-persistent objects
            if (!obj->IsDynamic() && !obj->IsPrivate()) {
                descArray.PushBack(oit.GetObjectDesc());
            }
        } else {
            missingObjects++;
        }
        oit++;
    }

    int descCount = descArray.Size();
    chunk->WriteInt(descCount);
    if (descCount > 0) {
        chunk->StartObjectIDSequence(descCount);
        for (auto it = descArray.Begin(); it != descArray.End(); ++it) {
            CKObject *obj = m_Context->GetObject((*it)->m_Object);
            chunk->WriteObjectSequence(obj);
        }

        chunk->StartSubChunkSequence(descCount * 2);
        for (auto it = descArray.Begin(); it != descArray.End(); ++it) {
            CKSceneObjectDesc *desc = *it;
            chunk->WriteSubChunkSequence(desc ? desc->m_InitialValue : nullptr);
            chunk->WriteSubChunkSequence(nullptr);
        }

        for (auto it = descArray.Begin(); it != descArray.End(); ++it) {
            CKSceneObjectDesc *desc = *it;
            chunk->WriteDword(desc ? desc->m_Flags : 0);
        }
    }

    chunk->WriteIdentifier(CK_STATESAVE_SCENELAUNCHED);
    chunk->WriteDword(m_EnvironmentSettings);

    chunk->WriteIdentifier(CK_STATESAVE_SCENERENDERSETTINGS);
    chunk->WriteDword(m_BackgroundColor);
    chunk->WriteDword(m_AmbientLightColor);

    chunk->WriteDword(m_FogMode);
    chunk->WriteDword(m_FogColor);
    chunk->WriteFloat(m_FogStart);
    chunk->WriteFloat(m_FogEnd);
    chunk->WriteFloat(m_FogDensity);

    CKTexture *bgTex = GetBackgroundTexture();
    chunk->WriteObject(bgTex);

    CKCamera *startCam = GetStartingCamera();
    chunk->WriteObject(startCam);

    if (missingObjects > 0) {
        CheckSceneObjectList();
    }

    chunk->CloseChunk();
    return chunk;
}

CKERROR CKScene::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) return CKERR_INVALIDPARAMETER;
    chunk->StartRead();
    m_EnvironmentSettings &= ~CK_SCENE_LAUNCHEDONCE;

    const int dataVersion = chunk->GetDataVersion();
    if (dataVersion >= 1) {
        // Load base class data
        CKERROR err = CKBeObject::Load(chunk, file);
        if (err != CK_OK)
            return err;

        if (chunk->SeekIdentifier(CK_STATESAVE_SCENENEWDATA)) {
            m_Level = chunk->ReadObjectID();

            RemoveAllObjects();

            // Load scene objects
            const int descCount = chunk->ReadInt();
            CKSceneObjectDesc *descList = new CKSceneObjectDesc[descCount];

            // Read object IDs
            chunk->StartReadSequence();
            for (int i = 0; i < descCount; ++i) {
                CKSceneObjectDesc *desc = &descList[i];
                desc->Init(nullptr);
                desc->m_Object = chunk->ReadObjectID();
            }

            CKAttributeManager *am = m_Context->GetAttributeManager();

            chunk->StartReadSequence();
            for (int i = 0; i < descCount; ++i) {
                CKStateChunk *objChunk = chunk->ReadSubChunk();
                descList[i].m_InitialValue = objChunk;
                if (objChunk && CKIsChildClassOf(objChunk->GetChunkClassID(), CKCID_BEOBJECT)) {
                    if (objChunk->GetChunkVersion() < 6 && !objChunk->GetManagers()) {
                        objChunk->StartRead();
                        if (objChunk->SeekIdentifier(CK_STATESAVE_NEWATTRIBUTES)) {
                            am->PatchRemapBeObjectStateChunk(objChunk);
                        }
                    }
                }

                CKStateChunk *subChunk = chunk->ReadSubChunk();
                if (subChunk) {
                    delete subChunk;
                }
            }

            if (dataVersion >= 8) {
                for (int i = 0; i < descCount; ++i) {
                    CKSceneObjectDesc *desc = &descList[i];
                    desc->m_Flags = chunk->ReadDword();
                }
            } else {
                for (int i = 0; i < descCount; ++i) {
                    CKSceneObjectDesc *desc = &descList[i];
                    CKDWORD flags = chunk->ReadDword();
                    desc->m_Flags = flags & CK_SCENEOBJECT_ACTIVE;
                    if (flags & 2) {
                        desc->m_Flags |= CK_SCENEOBJECT_START_RESET;
                    }
                    if (flags & 1) {
                        desc->m_Flags |= CK_SCENEOBJECT_START_ACTIVATE;
                    } else {
                        desc->m_Flags |= CK_SCENEOBJECT_START_DEACTIVATE;
                    }
                }
            }

            if (descCount > 0) {
                for (int i = 0; i < descCount; ++i) {
                    CKSceneObjectDesc *desc = &descList[i];
                    CKSceneObject *obj = (CKSceneObject *)m_Context->GetObject(desc->m_Object);
                    if (obj && CKIsChildClassOf(obj, CKCID_SCENEOBJECT)) {
                        m_SceneObjects.InsertUnique(obj->GetID(), *desc);
                        obj->AddSceneIn(this);
                    }
                }
            }

            delete[] descList;
            AddObjectToScene(GetLevel());
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_SCENELAUNCHED)) {
            m_EnvironmentSettings = chunk->ReadDword();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_SCENERENDERSETTINGS)) {
            m_BackgroundColor = chunk->ReadDword();
            m_AmbientLightColor = chunk->ReadDword();

            m_FogMode = (VXFOG_MODE)chunk->ReadDword();
            m_FogColor = chunk->ReadDword();
            m_FogStart = chunk->ReadFloat();
            m_FogEnd = chunk->ReadFloat();
            m_FogDensity = chunk->ReadFloat();
            m_BackgroundTexture = chunk->ReadObjectID();
            m_StartingCamera = chunk->ReadObjectID();

            if (file && file->m_FileInfo.ProductBuild < 0x2010000) {
                if (m_FogMode == VXFOG_EXP || m_FogMode == VXFOG_EXP2) {
                    m_FogMode = VXFOG_LINEAR;
                }
            }
        }
    } else {
        // Old data version is not supported yet
        return CKERR_OBSOLETEVIRTOOLS;
    }

    // Add scripts to the scene object list if they are not already present
    for (int i = 0; i < GetScriptCount(); ++i) {
        if (!GetSceneObjectDesc(GetScript(i)))
            AddObjectDesc(GetScript(i));
    }

    return CK_OK;
}

void CKScene::PreDelete() {
    for (CKSODHashIt it = m_SceneObjects.Begin(); it != m_SceneObjects.End(); ++it) {
        CKSceneObject *obj = (CKSceneObject *) m_Context->GetObject(it.GetKey());
        if (obj) obj->RemoveSceneIn(this);
        CKSceneObjectDesc &desc = *it;
        desc.Clear();
    }
}

void CKScene::CheckPostDeletion() {
    CKObject::CheckPostDeletion();
    CheckSceneObjectList();

    if (!m_Context->GetObject(m_BackgroundTexture))
        m_BackgroundTexture = 0;
    if (!m_Context->GetObject(m_StartingCamera))
        m_StartingCamera = 0;
}

int CKScene::GetMemoryOccupation() {
    int size = CKBeObject::GetMemoryOccupation() + (int) (sizeof(CKScene) - sizeof(CKBeObject));
    size += m_SceneObjects.GetMemoryOccupation(FALSE);
    return size;
}

int CKScene::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    if (m_SceneObjects.IsHere(o->GetID())) return TRUE;
    return CKBeObject::IsObjectUsed(o, cid);
}

CKERROR CKScene::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::PrepareDependencies(context);
    if (err != CK_OK) return err;

    if (context.IsInMode(CK_DEPENDENCIES_COPY)) {
        context.GetClassDependencies(m_ClassID);
        for (CKSODHashIt it = m_SceneObjects.Begin(); it != m_SceneObjects.End(); ++it) {
            CKObject *obj = m_Context->GetObject(it.GetKey());
            if (obj) obj->PrepareDependencies(context);
        }
        CKTexture *bgTexture = GetBackgroundTexture();
        if (bgTexture) bgTexture->PrepareDependencies(context);
    }
    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKScene::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::RemapDependencies(context);
    if (err != CK_OK) return err;

    m_Level = context.RemapID(m_Level);
    return CK_OK;
}

CKERROR CKScene::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKBeObject::Copy(o, context);
    if (err != CK_OK) return err;

    CKScene &scene = (CKScene &)o;
    if (&m_SceneObjects != &scene.m_SceneObjects) {
        m_SceneObjects = scene.m_SceneObjects;
    }

    m_EnvironmentSettings = scene.m_EnvironmentSettings;
    m_BackgroundColor = scene.m_BackgroundColor;
    m_BackgroundTexture = scene.m_BackgroundTexture;
    m_StartingCamera = scene.m_StartingCamera;
    m_AmbientLightColor = scene.m_AmbientLightColor;
    m_FogMode = scene.m_FogMode;
    m_FogStart = scene.m_FogStart;
    m_FogEnd = scene.m_FogEnd;
    m_FogColor = scene.m_FogColor;
    m_FogDensity = scene.m_FogDensity;
    return CK_OK;
}

CKSTRING CKScene::GetClassName() {
    return "Scene";
}

int CKScene::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKScene::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKScene::Register() {
    CKCLASSNOTIFYFROM(CKScene, CKSceneObject);
    CKPARAMETERFROMCLASS(CKScene, CKPGUID_SCENE);
}

CKScene *CKScene::CreateInstance(CKContext *Context) {
    return new CKScene(Context);
}

void CKScene::CheckSceneObjectList() {
    // Do not remove from the hash table while iterating it.
    // XHashTable uses pool compaction (FastRemove + RematEntry) which can remap
    // entries and cause iterator-based removal to revisit elements or loop.
    XArray<CK_ID> toRemove;
    const int count = m_SceneObjects.Size();
    if (count > 0)
        toRemove.Reserve(count);

    for (CKSODHashIt it = m_SceneObjects.Begin(); it != m_SceneObjects.End(); ++it) {
        const CK_ID objID = it.GetKey();
        if (!m_Context->GetObject(objID)) {
            toRemove.PushBack(objID);
        }
    }

    for (int i = 0; i < toRemove.Size(); ++i) {
        const CK_ID objID = toRemove[i];
        CKSODHashIt it = m_SceneObjects.Find(objID);
        if (it != m_SceneObjects.End()) {
            CKSceneObjectDesc &desc = *it;
            desc.Clear();
            m_SceneObjects.Remove(objID);
        }
    }
}

CKSceneObjectDesc *CKScene::AddObjectDesc(CKSceneObject *o) {
    if (!o) return nullptr;

    CKDWORD removeFlags = 0;
    if (o->GetClassID() == CKCID_BEHAVIOR) {
        CKBehavior *beh = (CKBehavior *) o;
        if (beh->GetType() != CKBEHAVIORTYPE_SCRIPT) return nullptr;
        CKBeObject *owner = beh->GetOwner();
        if (!owner) return nullptr;
        // For scripts owned by a level, check if this scene is NOT the default scene
        // of the current level. If so, clear the START_RESET flag.
        if (CKIsChildClassOf(owner, CKCID_LEVEL)) {
            CKLevel *currentLevel = m_Context->GetCurrentLevel();
            CKScene *defaultScene = currentLevel ? currentLevel->GetLevelScene() : nullptr;
            if (!defaultScene || m_ID != defaultScene->GetID())
                removeFlags = CK_SCENEOBJECT_START_RESET;
        }
    }

    CKSceneObjectDesc desc;
    desc.Init(o);
    desc.m_Flags &= ~removeFlags;
    CKSODHashIt it = m_SceneObjects.InsertUnique(o->GetID(), desc);
    o->AddSceneIn(this);
    return it;
}
