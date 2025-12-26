#include "CKGroup.h"

#include "CKFile.h"

CK_CLASSID CKGroup::m_ClassID = CKCID_GROUP;

CKERROR CKGroup::AddObject(CKBeObject *o) {
    if (!o || o == this || !CKIsChildClassOf(o, CKCID_BEOBJECT))
        return CKERR_INVALIDPARAMETER;
    if (o->IsInGroup(this))
        return CKERR_ALREADYPRESENT;
    o->AddToGroup(this);
    m_ObjectArray.PushBack(o);
    m_ClassIdUpdated = FALSE;
    return CK_OK;
}

CKERROR CKGroup::AddObjectFront(CKBeObject *o) {
    if (!o || o == this || !CKIsChildClassOf(o, CKCID_BEOBJECT))
        return CKERR_INVALIDPARAMETER;
    if (o->IsInGroup(this))
        return CKERR_ALREADYPRESENT;
    o->AddToGroup(this);
    m_ObjectArray.PushFront(o);
    m_ClassIdUpdated = FALSE;
    return CK_OK;
}

CKERROR CKGroup::InsertObjectAt(CKBeObject *o, int pos) {
    if (!o || o == this || !CKIsChildClassOf(o, CKCID_BEOBJECT))
        return CKERR_INVALIDPARAMETER;
    if (pos < 0 || pos > m_ObjectArray.Size())
        return CKERR_INVALIDPARAMETER;
    if (o->IsInGroup(this))
        return CKERR_ALREADYPRESENT;
    o->AddToGroup(this);
    m_ObjectArray.Insert(pos, o);
    m_ClassIdUpdated = FALSE;
    return CK_OK;
}

CKBeObject *CKGroup::RemoveObject(int pos) {
    if (pos < 0 || pos >= m_ObjectArray.Size())
        return nullptr;
    CKBeObject *o = (CKBeObject *)m_ObjectArray[pos];
    if (o) {
        o->RemoveFromGroup(this);
    }
    m_ObjectArray.RemoveAt(pos);
    m_ClassIdUpdated = FALSE;
    return o;
}

void CKGroup::RemoveObject(CKBeObject *obj) {
    if (!obj)
        return;

    if (m_ObjectArray.Remove(obj)) {
        obj->RemoveFromGroup(this);
        m_ClassIdUpdated = FALSE;
    }
}

void CKGroup::Clear() {
    for (int i = m_ObjectArray.Size() - 1; i >= 0; --i) {
        CKBeObject *o = (CKBeObject *)m_ObjectArray[i];
        if (o) {
            o->RemoveFromGroup(this);
        }
    }
    m_ObjectArray.Resize(0);
    m_ClassIdUpdated = FALSE;
}

void CKGroup::MoveObjectUp(CKBeObject *o) {
    if (o) {
        for (int i = 1; i < m_ObjectArray.Size(); ++i) {
            if (m_ObjectArray[i] == o) {
                m_ObjectArray.Swap(i, i - 1);
                break;
            }
        }
    }
}

void CKGroup::MoveObjectDown(CKBeObject *o) {
    if (o) {
        for (int i = 0; i < m_ObjectArray.Size() - 1; ++i) {
            if (m_ObjectArray[i] == o) {
                m_ObjectArray.Swap(i, i + 1);
                break;
            }
        }
    }
}

CKBeObject *CKGroup::GetObject(int pos) {
    if (pos < 0 || pos >= m_ObjectArray.Size())
        return nullptr;
    return (CKBeObject *)m_ObjectArray[pos];
}

int CKGroup::GetObjectCount() {
    return m_ObjectArray.Size();
}

CK_CLASSID CKGroup::GetCommonClassID() {
    if (!m_ClassIdUpdated) {
        ComputeClassID();
    }
    return m_CommonClassId;
}

int CKGroup::CanBeHide() {
    return TRUE;
}

void CKGroup::Show(CK_OBJECT_SHOWOPTION Show) {
    CKObject::Show(Show);
    for (int i = 0; i < m_ObjectArray.Size(); ++i) {
        CKObject *o = m_ObjectArray[i];
        if (o) {
            o->Show(Show);
        }
    }
}

CKGroup::CKGroup(CKContext *Context, CKSTRING Name) : CKBeObject(Context, Name) {
    m_ClassIdUpdated = FALSE;
    m_ObjectFlags = CK_OBJECT_VISIBLE;
    m_CommonClassId = CKCID_BEOBJECT;
    m_GroupIndex = m_Context->m_ObjectManager->GetGroupGlobalIndex();
}

CKGroup::~CKGroup() {
    m_Context->m_ObjectManager->ReleaseGroupGlobalIndex(m_GroupIndex);
}

CK_CLASSID CKGroup::GetClassID() {
    return m_ClassID;
}

void CKGroup::AddToScene(CKScene *scene, CKBOOL dependencies) {
    if (!scene) return;
    CKBeObject::AddToScene(scene, dependencies);
    if (dependencies) {
        for (int i = 0; i < m_ObjectArray.Size(); ++i) {
            CKBeObject *o = (CKBeObject *)m_ObjectArray[i];
            if (o) o->AddToScene(scene, dependencies);
        }
    }
}

void CKGroup::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    if (!scene) return;
    if (dependencies) {
        for (int i = 0; i < m_ObjectArray.Size(); ++i) {
            CKBeObject *o = (CKBeObject *)m_ObjectArray[i];
            if (o) o->RemoveFromScene(scene, dependencies);
        }
    }
    CKBeObject::RemoveFromScene(scene, dependencies);
}

void CKGroup::PreSave(CKFile *file, CKDWORD flags) {
    CKBeObject::PreSave(file, flags);
    if (!file) return;
    for (XObjectPointerArray::Iterator it = m_ObjectArray.Begin(); it != m_ObjectArray.End(); ++it) {
        CKBeObject *o = (CKBeObject *)*it;
        if (o && !CKIsChildClassOf(o, CKCID_LEVEL) && !CKIsChildClassOf(o, CKCID_SCENE)) {
            file->SaveObject(o, flags);
        }
    }
}

CKStateChunk *CKGroup::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_GROUP, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if ((flags & CK_STATESAVE_GROUPALL) && !m_ObjectArray.IsEmpty()) {
        chunk->WriteIdentifier(CK_STATESAVE_GROUPALL);
        m_ObjectArray.Save(chunk);
    }

    if (GetClassID() == CKCID_GROUP)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();
    return chunk;
}

CKERROR CKGroup::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) return CKERR_INVALIDPARAMETER;
    CKBeObject::Load(chunk, file);
    Clear();

    if (chunk->SeekIdentifier(CK_STATESAVE_GROUPALL)) {
        m_ObjectArray.Load(m_Context, chunk);
        if (!file) {
            for (XObjectPointerArray::Iterator it = m_ObjectArray.Begin(); it != m_ObjectArray.End();) {
                CKBeObject *o = (CKBeObject *)*it;
                if (o) {
                    if (!o->IsInGroup(this)) {
                        o->AddToGroup(this);
                    }
                    ++it;
                } else {
                    it = m_ObjectArray.Remove(it);
                }
            }
        }
    }

    m_ClassIdUpdated = FALSE;
    return CK_OK;
}

void CKGroup::PostLoad() {
    for (XObjectPointerArray::Iterator it = m_ObjectArray.Begin(); it != m_ObjectArray.End();) {
        CKBeObject *o = (CKBeObject *)*it;
        if (o && CKIsChildClassOf(o, CKCID_BEOBJECT)) {
            if (!o->IsInGroup(this)) {
                o->AddToGroup(this);
            }
            ++it;
        } else {
            it = m_ObjectArray.Remove(it);
        }
    }

    CKObject::PostLoad();
}

void CKGroup::PreDelete() {
    CKBeObject::PreDelete();
    for (XObjectPointerArray::Iterator it = m_ObjectArray.Begin(); it != m_ObjectArray.End(); ++it) {
        CKBeObject *o = (CKBeObject *)*it;
        if (o && !o->IsToBeDeleted()) {
            o->RemoveFromGroup(this);
        }
    }
}

void CKGroup::CheckPreDeletion() {
    CKObject::CheckPreDeletion();
    if (m_ObjectArray.Check())
        m_ClassIdUpdated = FALSE;
}

int CKGroup::GetMemoryOccupation() {
    int size = CKBeObject::GetMemoryOccupation() + (int) (sizeof(CKGroup) - sizeof(CKBeObject));
    size += m_ObjectArray.GetMemoryOccupation(FALSE);
    return size;
}

int CKGroup::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    if (!CKIsChildClassOf(o, CKCID_BEOBJECT))
        return CKBeObject::IsObjectUsed(o, cid);
    if (m_ObjectArray.FindObject(o))
        return TRUE;
    return CKBeObject::IsObjectUsed(o, cid);
}

CKERROR CKGroup::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;
    if (context.GetClassDependencies(m_ClassID) & 1) {
        if (context.IsInMode(CK_DEPENDENCIES_DELETE)) {
            XObjectPointerArray array;
            for (int i = 0; i < m_ObjectArray.Size(); ++i) {
                CKBeObject *o = (CKBeObject *)m_ObjectArray[i];
                if (o && !CKIsChildClassOf(o, CKCID_LEVEL) && !CKIsChildClassOf(o, CKCID_SCENE)) {
                    array.PushBack(o);
                }
            }
            array.Prepare(context);
        } else {
            m_ObjectArray.Prepare(context);
        }
    }
    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKGroup::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;
    m_ObjectArray.Remap(context);
    for (int i = 0; i < m_ObjectArray.Size(); ++i) {
        CKBeObject *o = (CKBeObject *)m_ObjectArray[i];
        if (o) {
            o->AddToGroup(this);
        }
    }
    return CK_OK;
}

CKERROR CKGroup::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKBeObject::Copy(o, context);
    if (err != CK_OK)
        return err;
    m_ObjectArray = ((CKGroup &)o).m_ObjectArray;
    return CK_OK;
}

CKSTRING CKGroup::GetClassName() {
    return "Group";
}

int CKGroup::GetDependenciesCount(int mode) {
    if (mode == CK_DEPENDENCIES_COPY || mode == CK_DEPENDENCIES_DELETE || mode == CK_DEPENDENCIES_SAVE)
        return 1;
    return 0;
}

CKSTRING CKGroup::GetDependencies(int i, int mode) {
    return (i == 0) ? "Objects" : nullptr;
}

void CKGroup::Register() {
    CKCLASSNOTIFYFROM(CKGroup, CKBeObject);
    CKPARAMETERFROMCLASS(CKGroup, CKPGUID_GROUP);
}

CKGroup *CKGroup::CreateInstance(CKContext *Context) {
    return new CKGroup(Context);
}

void CKGroup::ComputeClassID() {
    m_ClassIdUpdated = TRUE;
    if (m_ObjectArray.Size() == 0 || !m_ObjectArray[0]) {
        m_CommonClassId = CKCID_BEOBJECT;
    } else {
        m_CommonClassId = m_ObjectArray[0]->GetClassID();
    }
    for (int i = 1; i < m_ObjectArray.Size(); ++i) {
        CKObject *o = m_ObjectArray[i];
        if (o && !CKIsChildClassOf(o, m_CommonClassId)) {
            m_CommonClassId = CKGetCommonParent(m_CommonClassId, o->GetClassID());
        }
    }
}
