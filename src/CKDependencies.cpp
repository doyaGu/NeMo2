#include "CKDependencies.h"

#include "CKContext.h"
#include "CKObject.h"
#include "CKBehavior.h"
#include "CKLevel.h"
#include "CKScene.h"

CKDependenciesContext::CKDependenciesContext(const CKDependenciesContext &other)
    : m_DynamicObjects(other.m_DynamicObjects),
      m_CKContext(other.m_CKContext),
      m_Dependencies(other.m_Dependencies),
      m_MapID(other.m_MapID),
      m_Objects(other.m_Objects),
      m_DependenciesStack(other.m_DependenciesStack),
      m_Scripts(other.m_Scripts),
      m_Mode(other.m_Mode),
      m_CreationMode(other.m_CreationMode),
      m_CopyAppendString(other.m_CopyAppendString),
      m_ObjectsClassMask(other.m_ObjectsClassMask) {}

CKDependenciesContext &CKDependenciesContext::operator=(const CKDependenciesContext &other) {
    if (this != &other) {
        m_DynamicObjects = other.m_DynamicObjects;
        m_CKContext = other.m_CKContext;
        m_Dependencies = other.m_Dependencies;
        m_MapID = other.m_MapID;
        m_Objects = other.m_Objects;
        m_DependenciesStack = other.m_DependenciesStack;
        m_Scripts = other.m_Scripts;
        m_Mode = other.m_Mode;
        m_CreationMode = other.m_CreationMode;
        m_CopyAppendString = other.m_CopyAppendString;
        m_ObjectsClassMask = other.m_ObjectsClassMask;
    }
    return *this;
}

void CKDependenciesContext::AddObjects(CK_ID *ids, int count) {
    m_Objects.Reserve(m_Objects.Size() + count);
    if (count <= 0)
        return;

    for (int i = 0; i < count; ++i) {
        if (m_CKContext->GetObject(ids[i]))
            m_Objects.PushBack(ids[i]);
    }
}

int CKDependenciesContext::GetObjectsCount() {
    return m_Objects.Size();
}

CKObject *CKDependenciesContext::GetObjects(int i) {
    return m_CKContext->GetObject(m_Objects[i]);
}

CK_ID CKDependenciesContext::RemapID(CK_ID &id) {
    XHashItID it = m_MapID.Find(id);
    if (it != m_MapID.End()) {
        CK_ID remapped = *it;
        if (remapped)
            id = remapped;
    }
    return id;
}

CKObject *CKDependenciesContext::Remap(const CKObject *o) {
    if (!o) return nullptr;

    CK_ID id = o->m_ID;
    XHashItID it = m_MapID.Find(id);
    if (it != m_MapID.End()) {
        CK_ID remapped = *it;
        if (remapped)
            id = remapped;
    }
    return m_CKContext->GetObject(id);
}

XObjectArray CKDependenciesContext::FillDependencies() {
    XObjectArray objects;
    objects.Reserve(m_MapID.Size());
    for (XHashItID it = m_MapID.Begin(); it != m_MapID.End(); ++it) {
        objects.PushBack(it.GetKey());
    }
    return objects;
}

XObjectArray CKDependenciesContext::FillRemappedDependencies() {
    XObjectArray objects;
    objects.Reserve(m_MapID.Size());
    for (XHashItID it = m_MapID.Begin(); it != m_MapID.End(); ++it) {
        objects.PushBack(*it);
    }
    return objects;
}

CKDWORD CKDependenciesContext::GetClassDependencies(int c) {
    CKDependencies *deps = m_Dependencies;
    if (deps) {
        if (deps->m_Flags & CK_DEPENDENCIES_NONE) return 0;
        if (deps->m_Flags & CK_DEPENDENCIES_FULL) return 0xFFFF;
    } else {
        deps = CKGetDefaultClassDependencies((CK_DEPENDENCIES_OPMODE) m_Mode);
    }

    return (*deps)[c];
}

void CKDependenciesContext::Copy(CKSTRING appendstring) {
    m_CopyAppendString = appendstring;
    m_Mode = CK_DEPENDENCIES_COPY;

    m_Objects.Prepare(*this);
    m_ObjectsClassMask.Clear();
    m_Scripts.Clear();

    for (XHashItID it = m_MapID.Begin(); it != m_MapID.End(); ++it) {
        CK_ID originalId = it.GetKey();
        CKObject *original = m_CKContext->GetObject(originalId);
        if (!original) continue;

        m_ObjectsClassMask.Set(original->GetClassID());

        if (CKIsChildClassOf(original, CKCID_BEHAVIOR)) {
            CKBehavior *beh = (CKBehavior *) original;
            if (beh->GetType() & CKBEHAVIOR_SCRIPT) {
                m_Scripts.PushBack(beh->GetID());
            }
        }
    }

    m_CKContext->ExecuteManagersOnPreCopy(this);

    for (XHashItID it = m_MapID.Begin(); it != m_MapID.End(); ++it) {
        CK_ID originalId = it.GetKey();
        CKObject *original = m_CKContext->GetObject(originalId);
        if (!original) continue;

        CKObject *copy = m_CKContext->CreateObject(original->GetClassID(), nullptr, (CK_OBJECTCREATION_OPTIONS) m_CreationMode);
        if (copy) {
            *it = copy->GetID();
            copy->Copy(*original, *this);
        } else {
            *it = 0;
        }
    }

    for (XHashItID it = m_MapID.Begin(); it != m_MapID.End(); ++it) {
        CKObject *copy = m_CKContext->GetObject(*it);
        if (copy) {
            copy->RemapDependencies(*this);
        }
    }

    CKDWORD sceneObjectDependencies = GetClassDependencies(CKCID_SCENEOBJECT);
    CKLevel *currentLevel = m_CKContext->GetCurrentLevel();
    if (currentLevel) {
        CKScene *levelScene = currentLevel->GetLevelScene();
        const int sceneCount = m_CKContext->GetObjectsCountByClassID(CKCID_SCENE);
        CK_ID *sceneIds = m_CKContext->GetObjectsListByClassID(CKCID_SCENE);
        for (int i = 0; i < sceneCount; ++i) {
            CKScene *scene = (CKScene *) m_CKContext->GetObject(sceneIds[i]);
            if (!scene) continue;

            if (scene == levelScene || (sceneObjectDependencies & CK_DEPENDENCIES_COPY_SCENEOBJECT_SCENES)) {
                scene->BeginAddSequence(TRUE);
                for (XHashItID it = m_MapID.Begin(); it != m_MapID.End(); ++it) {
                    CKObject *original = m_CKContext->GetObject(it.GetKey());
                    CKObject *copy = m_CKContext->GetObject(*it);
                    if (!original || !copy) continue;

                    if (CKIsChildClassOf(copy, CKCID_SCENE)) {
                        // The copied scene itself needs to be added to the level
                        currentLevel->AddScene((CKScene *) copy);
                    } else if (CKIsChildClassOf(copy, CKCID_SCENEOBJECT)) {
                        CKSceneObject *sceneObjOrig = (CKSceneObject *) original;
                        CKSceneObject *sceneObjCopy = (CKSceneObject *) copy;
                        if (sceneObjOrig->IsInScene(scene)) {
                            scene->AddObject(sceneObjCopy);
                            CKDWORD origFlags = scene->GetObjectFlags(sceneObjOrig);
                            scene->SetObjectFlags(sceneObjCopy, (CK_SCENEOBJECT_FLAGS) (origFlags & ~CK_SCENEOBJECT_ACTIVE));

                            // Activate the object if required by creation options or if it was originally active
                            const bool shouldActivate = ((m_CreationMode & CK_OBJECTCREATION_ACTIVATE) &&
                                CKIsChildClassOf(sceneObjCopy, CKCID_BEOBJECT)) || scene->IsObjectActive(sceneObjOrig);
                            if (shouldActivate) {
                                scene->Activate(sceneObjCopy, TRUE);
                            }
                        }
                    }
                }
                scene->BeginAddSequence(FALSE);
            }
        }
    }

    m_Scripts.Remap(*this);
    m_CKContext->ExecuteManagersOnPostCopy(this);
    m_Objects.Remap(*this);
}

CKBOOL CKDependenciesContext::ContainClassID(CK_CLASSID Cid) {
    if (Cid < 0 || Cid >= m_ObjectsClassMask.Size()) return FALSE;
    return m_ObjectsClassMask.IsSet(Cid);
}

CKBOOL CKDependenciesContext::AddDependencies(CK_ID id) {
    return m_MapID.Insert(id, 0, FALSE);
}

void CKDependenciesContext::Clear() {
    m_DynamicObjects.Clear();
    m_MapID.Clear();
    m_Objects.Clear();
    m_Scripts.Clear();
    m_ObjectsClassMask.Clear();
}

CKERROR CKDependenciesContext::FinishPrepareDependencies(CKObject *iMySelf, CK_CLASSID Cid) {
    if (iMySelf->GetClassID() == Cid) {
        m_DynamicObjects.PopBack();
    }
    return CK_OK;
}
