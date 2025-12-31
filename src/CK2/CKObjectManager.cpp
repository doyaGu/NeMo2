#include "CKObjectManager.h"

#include "CKContext.h"
#include "CKSceneObject.h"
#include "CKRenderManager.h"
#include "CKRenderContext.h"
#include "CKMaterial.h"
#include "CK2dEntity.h"
#include "CK3dEntity.h"
#include "CKBehavior.h"
#include "CKGlobals.h"

extern int g_MaxClassID;
extern XClassInfoArray g_CKClassInfo;

int CKObjectManager::ObjectsByClass(CK_CLASSID cid, CKBOOL derived, CK_ID *obj_ids) {
    if (cid < 0 || cid >= g_MaxClassID)
        return 0;

    int count = 0;
    if (derived) {
        // Iterate through all class IDs to find derived classes
        for (CK_CLASSID currentCid = CKCID_OBJECT; currentCid < g_MaxClassID; ++currentCid) {
            if (!CKIsChildClassOf(currentCid, cid))
                continue;

            XObjectArray &objArray = m_ClassLists[currentCid];
            const int classCount = objArray.Size();
            if (classCount > 0) {
                if (obj_ids) {
                    memcpy(obj_ids + count, objArray.Begin(), classCount * sizeof(CK_ID));
                }
                count += classCount;
            }
        }
    } else {
        // Direct class match
        XObjectArray &objArray = m_ClassLists[cid];
        count = objArray.Size();
        if (obj_ids && count > 0) {
            memcpy(obj_ids, objArray.Begin(), count * sizeof(CK_ID));
        }
    }

    return count;
}

int CKObjectManager::GetObjectsCount() {
    return m_ObjectCount;
}

CKObject *CKObjectManager::GetObject(CK_ID id) {
    if (!m_Objects)
        return nullptr;
    if (id > static_cast<CK_ID>(m_ObjectCount))
        return nullptr;
    return m_Objects[id];
}

CKERROR CKObjectManager::DeleteAllObjects() {
    for (CK_CLASSID cid = 0; cid < g_MaxClassID; ++cid) {
        m_ClassLists[cid].Clear();
    }

    if (m_Objects) {
        for (CK_ID id = 0; id <= static_cast<CK_ID>(m_ObjectCount); ++id) {
            CKObject *obj = m_Objects[id];
            if (obj)
                delete obj;
            m_Objects[id] = nullptr;
        }
    }

    m_ObjectCount = 0;
    m_FreeObjectIDs.Clear();
    m_ObjectAppData.Clear();

    return CK_OK;
}

CKERROR CKObjectManager::ClearAllObjects() {
    // Bit array to track protected objects during cleanup
    XBitArray objectsToKeep;
    objectsToKeep.CheckSize(m_ObjectCount + 1);

    m_Context->m_InClearAll = TRUE;

    // Initialize: Mark all objects as to-be-deleted
    for (CK_ID id = static_cast<CK_ID>(m_ObjectCount); id > 0; --id) {
        CKObject *obj = m_Objects[id];
        if (obj)
            obj->ModifyObjectFlags(CK_OBJECT_TOBEDELETED, 0);
    }

    // Protect render-related objects from deletion
    if (m_Context->m_RenderManager) {
        const int rcCount = m_Context->m_RenderManager->GetRenderContextCount();
        for (int i = 0; i < rcCount; ++i) {
            CKRenderContext *dev = m_Context->m_RenderManager->GetRenderContext(i);
            if (!dev)
                continue;

            // Protect render context
            objectsToKeep.Set(dev->GetID());
            dev->ModifyObjectFlags(0, CK_OBJECT_TOBEDELETED);

            // Protect viewpoint
            CK3dEntity *viewpoint = dev->GetViewpoint();
            if (viewpoint) {
                objectsToKeep.Set(viewpoint->GetID());
                viewpoint->ModifyObjectFlags(0, CK_OBJECT_TOBEDELETED);
            }

            // Protect background material
            CKMaterial *bgMat = dev->GetBackgroundMaterial();
            if (bgMat) {
                bgMat->SetTexture(0, nullptr);
                objectsToKeep.Set(bgMat->GetID());
                bgMat->ModifyObjectFlags(0, CK_OBJECT_TOBEDELETED);
            }

            // Protect 2D roots
            CK2dEntity *bg2dRoot = dev->Get2dRoot(TRUE);
            if (bg2dRoot) {
                objectsToKeep.Set(bg2dRoot->GetID());
                bg2dRoot->ModifyObjectFlags(0, CK_OBJECT_TOBEDELETED);
            }
            CK2dEntity *fg2dRoot = dev->Get2dRoot(FALSE);
            if (fg2dRoot) {
                objectsToKeep.Set(fg2dRoot->GetID());
                fg2dRoot->ModifyObjectFlags(0, CK_OBJECT_TOBEDELETED);
            }
        }
    }

    // Destroy class lists and free arrays
    for (CK_CLASSID cid = 0; cid < g_MaxClassID; ++cid) {
        m_ClassLists[cid].Clear();
    }
    m_FreeObjectIDs.Clear();
    m_DynamicObjects.Clear();

    // First pass: Delete non-protected behaviors with notification
    for (CK_ID id = static_cast<CK_ID>(m_ObjectCount); id > 0; --id) {
        if (!objectsToKeep.IsSet(id)) {
            CKObject *obj = m_Objects[id];
            if (obj && CKIsChildClassOf(obj, CKCID_BEHAVIOR)) {
                CKBehavior *beh = (CKBehavior *) obj;
                beh->CallCallbackFunction(CKM_BEHAVIORDETACH);
                delete beh;
                m_Objects[id] = nullptr;
            }
        }
    }

    // Second pass: Delete remaining non-protected objects
    // Note: CK2.dll uses a clever algorithm to conditionally decrement m_ObjectCount
    // CK2.dll semantics:
    // - decrement m_ObjectCount while deleting objects
    // - once we encounter any protected object ID, stop decrementing for the rest of the scan
    CKBOOL shouldDecrement = TRUE;
    for (CK_ID id = static_cast<CK_ID>(m_ObjectCount); id > 0; --id) {
        if (objectsToKeep.IsSet(id)) {
            shouldDecrement = FALSE;
            continue;
        }

        CKObject *obj = m_Objects[id];
        if (obj) {
            delete obj;
            m_Objects[id] = nullptr;
        }

        if (shouldDecrement) {
            --m_ObjectCount;
        }
    }

    m_Context->m_InClearAll = FALSE;

    // Finalize protected objects - rebuild class lists
    for (CK_ID id = 1; id <= static_cast<CK_ID>(m_ObjectCount); ++id) {
        CKObject *obj = m_Objects[id];
        if (!obj)
            continue;

        // Render contexts need post-deletion
        CK_CLASSID cid = obj->GetClassID();
        if (cid == CKCID_RENDERCONTEXT)
            obj->CheckPostDeletion();

        // Re-add to class lists
        m_ClassLists[cid].PushBack(id);
    }

    // Cleanup auxiliary data
    m_ObjectAppData.Clear();
    m_SingleObjectActivities.Clear();

    return CK_OK;
}

CKBOOL CKObjectManager::IsObjectSafe(CKObject *iObject) {
    if (!iObject)
        return FALSE;

    // CK2.dll: starts from 0, loops while v3 <= m_ObjectsCount
    for (CK_ID id = 0; id <= static_cast<CK_ID>(m_ObjectCount); ++id) {
        if (m_Objects[id] == iObject)
            return TRUE;
    }

    return FALSE;
}

CKERROR CKObjectManager::DeleteObjects(CK_ID *obj_ids, int Count, CK_CLASSID cid, CKDWORD Flags) {
    if (m_Context->m_InClearAll)
        return CK_OK;

    if (Count == 0)
        return CKERR_INVALIDPARAMETER;

    CKDWORD flags = CK_OBJECT_TOBEDELETED;
    if (Flags & CK_DESTROY_FREEID)
        flags |= CK_OBJECT_FREEID;

    XBitArray notifiedClasses;
    for (int i = 0; i < Count; ++i) {
        CK_ID id = obj_ids[i];
        if (id == 0 || id > static_cast<CK_ID>(m_ObjectCount))
            continue;

        CKObject *obj = m_Objects[id];
        if (obj) {
            notifiedClasses.Set(obj->GetClassID());
            obj->ModifyObjectFlags(flags, 0);
        }
    }

    const int classCount = m_ClassLists.Size();
    for (CK_CLASSID classId = 0; classId < classCount; ++classId) {
        if (notifiedClasses.IsSet(classId)) {
            XObjectArray &classList = m_ClassLists[classId];
            const int newSize = CheckIDArrayPredeleted(classList.Begin(), classList.Size());
            classList.Resize(newSize);
        }
    }

    if (!(Flags & CK_DESTROY_NONOTIFY)) {
        m_Context->ExecuteManagersOnSequenceToBeDeleted(obj_ids, Count);
        if (classCount > CKCID_OBJECT) {
            for (CK_CLASSID classId = CKCID_OBJECT; classId < g_CKClassInfo.Size(); ++classId) {
                if (notifiedClasses.CheckCommon(g_CKClassInfo[classId].CommonToBeNotify)) {
                    XObjectArray &classList = m_ClassLists[classId];
                    for (int i = 0; i < classList.Size(); ++i) {
                        CKObject *obj = m_Objects[classList[i]];
                        if (obj) {
                            obj->CheckPreDeletion();
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < Count; ++i) {
        CK_ID id = obj_ids[i];
        if (id == 0 || id > static_cast<CK_ID>(m_ObjectCount))
            continue;

        CKObject *obj = m_Objects[id];
        if (obj) {
            obj->PreDelete();
        }
    }

    for (int i = 0; i < Count; ++i) {
        CK_ID id = obj_ids[i];
        if (id == 0 || id > static_cast<CK_ID>(m_ObjectCount))
            continue;

        CKObject *obj = m_Objects[id];
        if (obj) {
            delete obj;
        }
    }

    m_DynamicObjects.Check(m_Context);
    if (!(Flags & CK_DESTROY_NONOTIFY)) {
        if (classCount > CKCID_OBJECT) {
            for (CK_CLASSID classId = CKCID_OBJECT; classId < g_CKClassInfo.Size(); ++classId) {
                if (notifiedClasses.CheckCommon(g_CKClassInfo[classId].CommonToBeNotify)) {
                    XObjectArray &classList = m_ClassLists[classId];
                    for (int i = 0; i < classList.Size(); ++i) {
                        CK_ID id = classList[i];
                        if (id >= static_cast<CK_ID>(m_AllocatedObjectCount)) {
                            continue;
                        }
                        CKObject *obj = m_Objects[id];
                        if (obj) {
                            obj->CheckPostDeletion();
                        }
                    }
                }
            }
        }

        m_Context->ExecuteManagersOnSequenceDeleted(obj_ids, Count);
    }

    return CK_OK;
}

CKERROR CKObjectManager::GetRootEntities(XObjectPointerArray &array) {
    if (g_MaxClassID == 0)
        return CK_OK;

    for (CK_CLASSID cid = CKCID_OBJECT; cid <= g_MaxClassID; ++cid) {
        if (CKIsChildClassOf(cid, CKCID_3DENTITY)) {
            XObjectArray &classList = m_ClassLists[cid];
            for (XObjectArray::Iterator it = classList.Begin(); it != classList.End(); ++it) {
                CK3dEntity *ent = (CK3dEntity *) m_Objects[*it];
                if (ent && ent->GetParent() == nullptr)
                    array.PushBack(ent);
            }
        }
    }

    return CK_OK;
}

int CKObjectManager::GetObjectsCountByClassID(CK_CLASSID cid) {
    if (cid < 0 || cid >= g_MaxClassID) return 0;
    return m_ClassLists[cid].Size();
}

CK_ID *CKObjectManager::GetObjectsListByClassID(CK_CLASSID cid) {
    if (cid < 0 || cid >= g_MaxClassID) return nullptr;
    return m_ClassLists[cid].Begin();
}

CK_ID CKObjectManager::RegisterObject(CKObject *iObject) {
    CK_ID id = 0;
    if (!m_FreeObjectIDs.IsEmpty()) {
        id = m_FreeObjectIDs.PopBack();
    } else {
        ++m_ObjectCount;
        if (m_ObjectCount >= m_AllocatedObjectCount) {
            int newCount = m_AllocatedObjectCount + 100;
            CKObject **newObjs = new CKObject *[newCount];
            memset(newObjs, 0, newCount * sizeof(CKObject *));
            if (m_Objects) {
                memcpy(newObjs, m_Objects, m_AllocatedObjectCount * sizeof(CKObject *));
                delete[] m_Objects;
            }
            m_Objects = newObjs;
            m_AllocatedObjectCount = newCount;
        }
        id = m_ObjectCount;
    }
    m_Objects[id] = iObject;
    return id;
}

void CKObjectManager::FinishRegisterObject(CKObject *iObject) {
    CK_ID id = iObject->GetID();
    if (iObject->IsDynamic())
        m_DynamicObjects.PushBack(id);
    m_ClassLists[iObject->GetClassID()].PushBack(id);
}

void CKObjectManager::UnRegisterObject(CK_ID id) {
    // CK2.dll: checks id < m_AllocatedObjectCount && id != 0
    if (id >= static_cast<CK_ID>(m_AllocatedObjectCount))
        return;
    if (id == 0)
        return;

    CKObject *obj = m_Objects[id];

    // CK2.dll only adds to free ID list if FREEID flag (0x8) is set and not in ClearAll
    if (obj && (obj->GetObjectFlags() & CK_OBJECT_FREEID) && !m_Context->m_InClearAll) {
        m_FreeObjectIDs.PushBack(id);
    }

    // CK2.dll nulls the object pointer
    m_Objects[id] = nullptr;

    // CK2.dll removes from hash tables if they have entries
    if (m_ObjectAppData.Size() > 0)
        m_ObjectAppData.Remove(id);
    if (m_SingleObjectActivities.Size() > 0)
        m_SingleObjectActivities.Remove(id);
}

CKObject *CKObjectManager::GetObjectByName(CKSTRING name, CKObject *previous) {
    if (!name) return nullptr;
    CK_ID startId = previous ? previous->GetID() + 1 : 1;

    for (CK_ID id = startId; id <= static_cast<CK_ID>(m_ObjectCount); ++id) {
        CKObject *obj = m_Objects[id];
        if (obj && obj->GetName() && strcmp(obj->GetName(), name) == 0) {
            return obj;
        }
    }
    return nullptr;
}

CKObject *CKObjectManager::GetObjectByName(CKSTRING name, CK_CLASSID cid, CKObject *previous) {
    if (!name || cid < 0 || cid >= g_MaxClassID) return nullptr;

    XObjectArray &classList = m_ClassLists[cid];
    int startIndex = 0;
    if (previous) {
        int pos = classList.GetPosition(previous->GetID());
        if (pos != -1) startIndex = pos + 1;
    }

    for (int i = startIndex; i < classList.Size(); ++i) {
        CKObject *obj = GetObject(classList[i]);
        if (obj && obj->GetName() && strcmp(obj->GetName(), name) == 0) {
            return obj;
        }
    }
    return nullptr;
}

CKObject *CKObjectManager::GetObjectByNameAndParentClass(CKSTRING name, CK_CLASSID pcid, CKObject *previous) {
    if (!name || pcid < 0 || pcid >= g_MaxClassID) return nullptr;

    CK_CLASSID startCid = previous ? previous->GetClassID() : 0;
    int startIndex = 0;
    if (previous) {
        startIndex = m_ClassLists[startCid].GetPosition(previous->GetID()) + 1;
    }

    for (CK_CLASSID cid = startCid; cid < g_MaxClassID; ++cid) {
        if (!CKIsChildClassOf(cid, pcid)) continue;

        XObjectArray &classList = m_ClassLists[cid];
        for (int i = (cid == startCid) ? startIndex : 0; i < classList.Size(); ++i) {
            CKObject *obj = GetObject(classList[i]);
            if (obj && obj->GetName() && strcmp(obj->GetName(), name) == 0) {
                return obj;
            }
        }
    }
    return nullptr;
}

CKERROR CKObjectManager::GetObjectListByType(CK_CLASSID cid, XObjectPointerArray &array, CKBOOL derived) {
    if (cid < 0 || cid >= g_MaxClassID) return CKERR_INVALIDPARAMETER;

    if (derived) {
        for (CK_CLASSID classId = 0; classId < g_MaxClassID; ++classId) {
            if (CKIsChildClassOf(classId, cid)) {
                XObjectArray &classList = m_ClassLists[classId];
                for (int i = 0; i < classList.Size(); ++i) {
                    CKObject *obj = GetObject(classList[i]);
                    if (obj) array.PushBack(obj);
                }
            }
        }
    } else {
        XObjectArray &classList = m_ClassLists[cid];
        for (int i = 0; i < classList.Size(); ++i) {
            CKObject *obj = GetObject(classList[i]);
            if (obj) array.PushBack(obj);
        }
    }
    return CK_OK;
}

CKBOOL CKObjectManager::InLoadSession() {
    return m_InLoadSession;
}

void CKObjectManager::StartLoadSession(int MaxObjectID) {
    if (!m_InLoadSession) {
        if (m_LoadSession)
            delete[] m_LoadSession;
        m_MaxObjectID = MaxObjectID;
        m_LoadSession = new CK_ID[MaxObjectID];
        memset(m_LoadSession, 0, sizeof(CK_ID) * MaxObjectID);
        m_InLoadSession = TRUE;
    }
}

void CKObjectManager::EndLoadSession() {
    delete[] m_LoadSession;
    m_LoadSession = nullptr;
    m_InLoadSession = FALSE;
}

void CKObjectManager::RegisterLoadObject(CKObject *iObject, CK_ID ObjectID) {
    if (m_LoadSession && ObjectID >= 0 && static_cast<CKDWORD>(ObjectID) < m_MaxObjectID) {
        m_LoadSession[ObjectID] = iObject ? iObject->GetID() : 0;
    }
}

CK_ID CKObjectManager::RealId(CK_ID id) {
    if (!m_LoadSession)
        return id;
    if (id < m_MaxObjectID)
        return m_LoadSession[id];
    return 0;
}

int CKObjectManager::CheckIDArray(CK_ID *obj_ids, int Count) {
    if (!obj_ids || Count <= 0)
        return 0;

    int writeCursor = 0;
    int readCursor = 0;

    // Phase 1: Fast path - original only checks if m_Objects[id] is non-null
    while (readCursor < Count) {
        CK_ID id = obj_ids[readCursor];
        if (id >= static_cast<CK_ID>(m_AllocatedObjectCount) || !m_Objects[id]) {
            break;
        }
        ++readCursor;
        ++writeCursor;
    }

    // Phase 2: Slow path - skip the invalid entry and continue
    readCursor++;
    while (readCursor < Count) {
        CK_ID id = obj_ids[readCursor];
        if (id < static_cast<CK_ID>(m_AllocatedObjectCount) && m_Objects[id]) {
            obj_ids[writeCursor] = id;
            ++writeCursor;
        }
        ++readCursor;
    }

    return writeCursor;
}

int CKObjectManager::CheckIDArrayPredeleted(CK_ID *obj_ids, int Count) {
    if (!obj_ids || Count <= 0)
        return 0;

    int writeCursor = 0;
    int readCursor = 0;

    // Phase 1: Fast path - check object exists and ToBeDeleted flag
    while (readCursor < Count) {
        CK_ID id = obj_ids[readCursor];
        CKObject *obj = (id < static_cast<CK_ID>(m_AllocatedObjectCount)) ? m_Objects[id] : nullptr;
        if (!obj || obj->IsToBeDeleted()) {
            break;
        }
        ++readCursor;
        ++writeCursor;
    }

    // Phase 2: Slow path
    readCursor++;
    while (readCursor < Count) {
        CK_ID id = obj_ids[readCursor];
        CKObject *obj = (id < static_cast<CK_ID>(m_AllocatedObjectCount)) ? m_Objects[id] : nullptr;
        if (obj && !obj->IsToBeDeleted()) {
            obj_ids[writeCursor] = id;
            ++writeCursor;
        }
        ++readCursor;
    }

    return writeCursor;
}

CKDeferredDeletion *CKObjectManager::MatchDeletion(CKDependencies *depoptions, CKDWORD Flags) {
    if (Flags >= 4)
        return nullptr;
    XArray<CKDeferredDeletion *> &deletions = m_DeferredDeletions[Flags];
    for (XArray<CKDeferredDeletion *>::Iterator it = deletions.Begin(); it != deletions.End(); ++it) {
        if ((*it)->m_DependenciesPtr == depoptions)
            return *it;
    }
    return nullptr;
}

void CKObjectManager::RegisterDeletion(CKDeferredDeletion *deletion) {
    if (deletion && deletion->m_Flags < 4) {
        m_DeferredDeletions[deletion->m_Flags].PushBack(deletion);
    }
}

int CKObjectManager::GetDynamicIDCount() {
    return m_DynamicObjects.Size();
}

CK_ID CKObjectManager::GetDynamicID(int index) {
    if (index < 0 || index >= m_DynamicObjects.Size())
        return 0;
    return m_DynamicObjects[index];
}

void CKObjectManager::DeleteAllDynamicObjects() {
    m_NeedDeleteAllDynamicObjects = TRUE;
}

void CKObjectManager::SetDynamic(CKObject *iObject) {
    if (iObject) {
        if (!iObject->IsDynamic()) {
            iObject->ModifyObjectFlags(CK_OBJECT_DYNAMIC, 0);
            m_DynamicObjects.PushBack(iObject->GetID());
        }
    }
}

void CKObjectManager::UnSetDynamic(CKObject *iObject) {
    if (iObject) {
        if (iObject->IsDynamic()) {
            iObject->ModifyObjectFlags(0, CK_OBJECT_DYNAMIC);
            m_DynamicObjects.Remove(iObject->GetID());
        }
    }
}

CKERROR CKObjectManager::PostProcess() {
    // Process deferred deletions
    for (CKDWORD flag = 0; flag < 4; ++flag) {
        XArray<CKDeferredDeletion *> &deletions = m_DeferredDeletions[flag];
        if (!deletions.IsEmpty()) {
            for (CKDeferredDeletion **it = deletions.Begin(); it != deletions.End(); ++it) {
                CKDeferredDeletion *deletion = *it;
                m_Context->PrepareDestroyObjects(deletion->m_Dependencies.Begin(),
                                                 deletion->m_Dependencies.Size(),
                                                 flag,
                                                 deletion->m_DependenciesPtr);
            }

            m_Context->FinishDestroyObjects(flag);

            for (CKDeferredDeletion **it = deletions.Begin(); it != deletions.End(); ++it) {
                CKDeferredDeletion *deletion = *it;
                if (deletion) {
                    if (deletion->m_DependenciesPtr) {
                        deletion->m_DependenciesPtr->Clear();
                        delete deletion->m_DependenciesPtr;
                    }
                    deletion->m_Dependencies.Clear();
                    delete deletion;
                }
            }
            deletions.Clear();
        }
    }

    // Process deletion of all dynamic objects if requested
    if (m_NeedDeleteAllDynamicObjects) {
        if (!m_DynamicObjects.IsEmpty()) {
            // Swap out the dynamic objects array before destroying
            XObjectArray tempArray;
            tempArray.Swap(m_DynamicObjects);
            m_Context->DestroyObjects(tempArray.Begin(), tempArray.Size());
        }
        m_NeedDeleteAllDynamicObjects = FALSE;
    }

    return CK_OK;
}

CKERROR CKObjectManager::OnCKReset() {
    if (!m_DynamicObjects.IsEmpty()) {
        // Swap out the dynamic objects array before destroying
        XObjectArray tempArray;
        tempArray.Swap(m_DynamicObjects);
        m_Context->DestroyObjects(tempArray.Begin(), tempArray.Size());
    }

    return CK_OK;
}

CKObjectManager::~CKObjectManager() {
    delete[] m_Objects;
    delete[] m_LoadSession;
}

CKObjectManager::CKObjectManager(CKContext *Context) : CKBaseManager(Context, OBJECT_MANAGER_GUID, "Object Manager") {
    m_MaxObjectID = 0;
    m_ObjectCount = 0;
    m_AllocatedObjectCount = 1000;
    m_Objects = new CKObject *[1000];
    memset(m_Objects, 0, m_AllocatedObjectCount * sizeof(CKObject *));
    m_LoadSession = nullptr;
    m_InLoadSession = FALSE;
    m_NeedDeleteAllDynamicObjects = FALSE;
    m_ClassLists.Resize(g_MaxClassID);
    m_Context->RegisterNewManager(this);
}

CKDWORD CKObjectManager::GetGroupGlobalIndex() {
    int index = m_GroupGlobalIndex.GetUnsetBitPosition(0);
    m_GroupGlobalIndex.Set(index);
    return index;
}

void CKObjectManager::ReleaseGroupGlobalIndex(CKDWORD index) {
    m_GroupGlobalIndex.Unset(index);
}

int CKObjectManager::GetSceneGlobalIndex() {
    int index = m_SceneGlobalIndex.GetUnsetBitPosition(0);
    m_SceneGlobalIndex.Set(index);
    return index;
}

void CKObjectManager::ReleaseSceneGlobalIndex(int index) {
    m_SceneGlobalIndex.Unset(index);
}

void *CKObjectManager::GetObjectAppData(CK_ID id) {
    XObjectAppDataTable::Iterator it = m_ObjectAppData.Find(id);
    if (it == m_ObjectAppData.End()) {
        return nullptr;
    }
    return *it;
}

void CKObjectManager::SetObjectAppData(CK_ID id, void *arg) {
    m_ObjectAppData[id] = arg;
}

void CKObjectManager::AddSingleObjectActivity(CKSceneObject *o, CK_ID id) {
    if (o) {
        m_SingleObjectActivities[o->GetID()] = id;
    }
}

int CKObjectManager::GetSingleObjectActivity(CKSceneObject *o, CK_ID &id) {
    if (!o) {
        return 0;
    }

    XHashItID it = m_SingleObjectActivities.Find(o->GetID());
    if (it == m_SingleObjectActivities.End()) {
        return 0;
    }

    id = *it;
    return 1;
}
