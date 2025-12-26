#include "CKSceneObject.h"
#include "CKScene.h"
#include "CKAttributeManager.h"

CK_CLASSID CKSceneObject::m_ClassID = CKCID_SCENEOBJECT;

CKBOOL CKSceneObject::IsActiveInScene(CKScene *scene) {
    if (!scene)
        return FALSE;
    return scene->IsObjectActive(this);
}

CKBOOL CKSceneObject::IsActiveInCurrentScene() {
    CKScene *scene = m_Context->GetCurrentScene();
    if (!scene)
        return FALSE;
    return scene->IsObjectActive(this);
}

CKBOOL CKSceneObject::IsInScene(CKScene *scene) {
    if (!scene)
        return FALSE;

    if (scene->m_SceneGlobalIndex >= m_Scenes.Size())
        return FALSE;

    return m_Scenes.IsSet(scene->m_SceneGlobalIndex);
}

int CKSceneObject::GetSceneInCount() {
    return m_Scenes.BitSet();
}

CKScene *CKSceneObject::GetSceneIn(int index) {
    int pos = m_Scenes.GetSetBitPosition(index);
    if (pos < 0)
        return nullptr;

    int count = m_Context->GetObjectsCountByClassID(CKCID_SCENE);
    if (count <= 0)
        return nullptr;

    CK_ID *ids = m_Context->GetObjectsListByClassID(CKCID_SCENE);

    for (int i = 0; i < count; ++i)
    {
        CKScene *scene = (CKScene *)m_Context->GetObject(ids[i]);
        if (scene && scene->m_SceneGlobalIndex == pos)
            return scene;
    }

    return nullptr;
}

CKSceneObject::CKSceneObject(CKContext *Context, CKSTRING name) : CKObject(Context, name) {}

CKSceneObject::~CKSceneObject() {}

CK_CLASSID CKSceneObject::GetClassID() {
    return m_ClassID;
}

int CKSceneObject::GetMemoryOccupation() {
    // Object itself + XBitArray header + owned bit-buffer.
    return CKObject::GetMemoryOccupation() + static_cast<int>(sizeof(XBitArray)) + m_Scenes.GetMemoryOccupation();
}

CKSTRING CKSceneObject::GetClassName() {
    return "Scene Object";
}

int CKSceneObject::GetDependenciesCount(int mode) {
    return mode == 1;
}

CKSTRING CKSceneObject::GetDependencies(int i, int mode) {
    if (mode == 1 && i == 0)
        return "Scenes Belonging";
    return nullptr;
}

void CKSceneObject::Register() {
    CKPARAMETERFROMCLASS(CKSceneObject, CKPGUID_SCENEOBJECT);
    CKCLASSDEFAULTCOPYDEPENDENCIES(CKSceneObject, CK_DEPENDENCIES_COPY_OBJECT_NAME);
}

CKSceneObject *CKSceneObject::CreateInstance(CKContext *Context) {
    return new CKSceneObject(Context);
}

void CKSceneObject::AddToScene(CKScene *scene, CKBOOL dependencies) {
    if (scene)
        scene->AddObject(this);
}

void CKSceneObject::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    if (scene)
        scene->RemoveObject(this);
}

void CKSceneObject::AddSceneIn(CKScene *scene) {
    if (!scene)
        return;

    if (m_Scenes.TestSet(scene->m_SceneGlobalIndex))
        m_Context->GetAttributeManager()->RefreshList(this, scene);
}

void CKSceneObject::RemoveSceneIn(CKScene *scene) {
    if (scene)
        m_Scenes.TestUnset(scene->m_SceneGlobalIndex);
}

void CKSceneObject::RemoveFromAllScenes() {
    int count = GetSceneInCount();
    for (int i = count - 1; i >= 0; --i) {
        CKScene *scene = GetSceneIn(i);
        if (scene)
            scene->RemoveObject(this);
    }
}