#include "CKBaseManager.h"
#include "CKGlobals.h"
#include "CKContext.h"

CKBaseManager::CKBaseManager(CKContext *Context, CKGUID guid, CKSTRING Name) {
    m_ManagerGuid = guid;
    m_ManagerName = CKStrdup(Name);
    m_Context = Context;
    m_ProcessingTime = 0.0f;
}

CKBaseManager::~CKBaseManager() {
    if (m_Context && !m_Context->m_Init) {
        if (m_Context->m_InactiveManagers.IsHere(this))
            m_Context->m_ManagerTable.Remove(m_ManagerGuid);
        m_Context->BuildSortedLists();
    }
    delete[] m_ManagerName;
    m_ManagerName = nullptr;
}

CKERROR CKBaseManager::CKDestroyObject(CKObject *obj, CKDWORD Flags, CKDependencies *depoptions) {
    return m_Context->DestroyObject(obj, Flags, depoptions);
}

CKERROR CKBaseManager::CKDestroyObject(CK_ID id, CKDWORD Flags, CKDependencies *depoptions) {
    return m_Context->DestroyObject(id, Flags, depoptions);
}

CKERROR CKBaseManager::CKDestroyObjects(CK_ID *obj_ids, int Count, CKDWORD Flags, CKDependencies *depoptions) {
    return m_Context->DestroyObjects(obj_ids, Count, Flags, depoptions);
}

CKObject *CKBaseManager::CKGetObject(CK_ID id) {
    return m_Context->GetObject(id);
}

void CKBaseManager::StartProfile() {
    m_Timer.Reset();
}

void CKBaseManager::StopProfile() {
    m_ProcessingTime += m_Timer.Current();
}
