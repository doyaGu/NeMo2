#include "CKContext.h"

#include "CKPluginManager.h"
#include "CKBaseManager.h"
#include "CKObjectManager.h"
#include "CKParameterManager.h"
#include "CKAttributeManager.h"
#include "CKTimeManager.h"
#include "CKMessageManager.h"
#include "CKBehaviorManager.h"
#include "CKRenderManager.h"
#include "CKPathManager.h"
#include "CKInterfaceManager.h"
#include "CKGlobals.h"
#include "CKFile.h"
#include "CKBitmapReader.h"
#include "CKDebugContext.h"
#include "CKLevel.h"
#include "CKScene.h"
#include "CKRenderContext.h"
#include "CKBehavior.h"
#include "CKCharacter.h"
#include "CKSynchroObject.h"
#include "CKParameterOperation.h"

extern XClassInfoArray g_CKClassInfo;
extern CKPluginManager g_ThePluginManager;

CKObject *CKContext::CreateObject(CK_CLASSID cid, CKSTRING Name, CK_OBJECTCREATION_OPTIONS Options, CK_CREATIONMODE *Res) {
    CKObject *obj = nullptr;
    CK_CREATIONMODE creationMode = CKLOAD_OK;
    char *buffer = GetStringBuffer(512);
    if (!buffer)
        return nullptr;

    CKDWORD creationOptions = (Options & CK_OBJECTCREATION_FLAGSMASK);
    CKClassDesc &classDesc = g_CKClassInfo[cid];
    if (classDesc.DefaultOptions & CK_GENERALOPTIONS_NODUPLICATENAMECHECK)
        creationOptions = CK_OBJECTCREATION_NONAMECHECK;
    if (classDesc.DefaultOptions & CK_GENERALOPTIONS_AUTOMATICUSECURRENT)
        creationOptions = CK_OBJECTCREATION_USECURRENT;

    if (Name) {
        switch (creationOptions) {
        case CK_OBJECTCREATION_RENAME:
            obj = GetObjectByNameAndClass(Name, cid);
            if (obj) {
                GetSecureName(buffer, Name, cid);
                creationMode = CKLOAD_RENAME;
            }
            break;
        case CK_OBJECTCREATION_USECURRENT:
            obj = GetObjectByNameAndClass(Name, cid);
            if (obj) {
                creationMode = CKLOAD_USECURRENT;
            }
            break;
        case CK_OBJECTCREATION_ASK:
            creationMode = LoadVerifyObjectUnicity(Name, cid, buffer, &obj);
            break;
        case CK_OBJECTCREATION_NONAMECHECK:
        case CK_OBJECTCREATION_REPLACE:
        default:
            break;
        }
    }

    if (creationMode != CKLOAD_USECURRENT) {
        if (classDesc.CreationFct) {
            m_InDynamicCreationMode = (Options & CK_OBJECTCREATION_DYNAMIC) != 0;
            obj = classDesc.CreationFct(this);
            m_InDynamicCreationMode = FALSE;

            if (obj) {
                if (Options & CK_OBJECTCREATION_DYNAMIC)
                    obj->ModifyObjectFlags(CK_OBJECT_DYNAMIC, 0);
                m_ObjectManager->FinishRegisterObject(obj);
            }
        }
    }

    if (obj) {
        if (Name != nullptr) {
            if (creationMode == CKLOAD_RENAME)
                obj->SetName(buffer);
            else
                obj->SetName(Name);
        }
    }

    if (Res)
        *Res = creationMode;

    return obj;
}

CKObject *CKContext::CopyObject(CKObject *src, CKDependencies *Dependencies, CKSTRING AppendName, CK_OBJECTCREATION_OPTIONS Options) {
    if (!src) return nullptr;

    CKDependenciesContext depContext(this);
    depContext.SetCreationMode(Options);
    depContext.SetOperationMode(CK_DEPENDENCIES_COPY);
    if (Dependencies) depContext.StartDependencies(Dependencies);
    depContext.AddObjects(&src->m_ID, 1);
    depContext.Copy(AppendName);
    if (Dependencies) depContext.StopDependencies();
    return depContext.Remap(src);
}

const XObjectArray &CKContext::CopyObjects(const XObjectArray &SrcObjects, CKDependencies *Dependencies, CK_OBJECTCREATION_OPTIONS Options, CKSTRING AppendName) {
    CKDependenciesContext depContext(this);
    depContext.SetCreationMode(Options);
    depContext.SetOperationMode(CK_DEPENDENCIES_COPY);
    if (Dependencies) depContext.StartDependencies(Dependencies);
    depContext.AddObjects(SrcObjects.Begin(), SrcObjects.Size());
    depContext.Copy(AppendName);

    m_CopyObjects.Resize(0);
    const int count = depContext.GetObjectsCount();
    for (int i = 0; i < count; i++) {
        CKObject *obj = depContext.GetObjects(i);
        m_CopyObjects.PushBack(obj ? obj->GetID() : 0);
    }
    if (Dependencies) depContext.StopDependencies();
    return m_CopyObjects;
}

CKObject *CKContext::GetObject(CK_ID ObjID) {
    return m_ObjectManager->GetObject(ObjID);
}

int CKContext::GetObjectCount() {
    return m_ObjectManager->GetObjectsCount();
}

int CKContext::GetObjectSize(CKObject *obj) {
    return obj ? obj->GetMemoryOccupation() : 0;
}

CKERROR CKContext::DestroyObject(CKObject *obj, CKDWORD Flags, CKDependencies *depoptions) {
    if (!obj) return CKERR_INVALIDPARAMETER;
    return DestroyObjects(&obj->m_ID, 1, Flags, depoptions);
}

CKERROR CKContext::DestroyObject(CK_ID id, CKDWORD Flags, CKDependencies *depoptions) {
    if (id == 0) return CKERR_INVALIDPARAMETER;
    return DestroyObjects(&id, 1, Flags, depoptions);
}

CKERROR CKContext::DestroyObjects(CK_ID *obj_ids, int Count, CKDWORD Flags, CKDependencies *depoptions) {
    if (m_DeferDestroyObjects) {
        DeferredDestroyObjects(obj_ids, Count, depoptions, Flags);
    } else {
        if (PrepareDestroyObjects(obj_ids, Count, Flags, depoptions) == CK_OK) {
            FinishDestroyObjects(m_DestroyObjectFlag);
        }
    }
    return CK_OK;
}

void CKContext::DestroyAllDynamicObjects() {
    m_ObjectManager->DeleteAllDynamicObjects();
}

void CKContext::ChangeObjectDynamic(CKObject *iObject, CKBOOL iSetDynamic) {
    if (iSetDynamic)
        m_ObjectManager->SetDynamic(iObject);
    else
        m_ObjectManager->UnSetDynamic(iObject);
}

const XObjectPointerArray &CKContext::CKFillObjectsUnused() {
    CKDependencies deps;
    deps.m_Flags = CK_DEPENDENCIES_FULL;
    CKDependenciesContext depContext(this);
    depContext.SetOperationMode(CK_DEPENDENCIES_BUILD);
    depContext.StartDependencies(&deps);

    CKRenderContext *dev = GetPlayerRenderContext();
    if (dev) dev->PrepareDependencies(depContext);

    XObjectPointerArray objects;
    objects += GetObjectListByType(CKCID_DATAARRAY, FALSE);
    objects += GetObjectListByType(CKCID_GROUP, FALSE);
    objects += GetObjectListByType(CKCID_BEHAVIOR, FALSE);
    objects.Prepare(depContext);

    CKScene *currentScene = GetCurrentScene();

    m_ObjectsUnused.Resize(0);
    const int count = m_ObjectManager->GetObjectsCount();
    for (int i = 0; i < count; ++i) {
        CKObject *obj = m_ObjectManager->GetObject(i);
        if (!obj) continue;

        if (!depContext.IsDependenciesHere(obj->GetID())) {
            if (!CKIsChildClassOf(obj, CKCID_SCENEOBJECT) ||
                ((CKSceneObject *) obj)->IsInScene(currentScene)) {
                m_ObjectsUnused.PushBack(obj);
            }
        }
    }

    depContext.StopDependencies();
    return m_ObjectsUnused;
}

CKObject *CKContext::GetObjectByName(CKSTRING name, CKObject *previous) {
    return m_ObjectManager->GetObjectByName(name, previous);
}

CKObject *CKContext::GetObjectByNameAndClass(CKSTRING name, CK_CLASSID cid, CKObject *previous) {
    return m_ObjectManager->GetObjectByName(name, cid, previous);
}

CKObject *CKContext::GetObjectByNameAndParentClass(CKSTRING name, CK_CLASSID pcid, CKObject *previous) {
    return m_ObjectManager->GetObjectByNameAndParentClass(name, pcid, previous);
}

const XObjectPointerArray &CKContext::GetObjectListByType(CK_CLASSID cid, CKBOOL derived) {
    m_ObjectsUnused.Resize(0);
    m_ObjectManager->GetObjectListByType(cid, m_ObjectsUnused, derived);
    return m_ObjectsUnused;
}

int CKContext::GetObjectsCountByClassID(CK_CLASSID cid) {
    return m_ObjectManager->GetObjectsCountByClassID(cid);
}

CK_ID *CKContext::GetObjectsListByClassID(CK_CLASSID cid) {
    return m_ObjectManager->GetObjectsListByClassID(cid);
}

CKERROR CKContext::Play() {
    if (m_Playing)
        return CK_OK;

    CKLevel *currentLevel = GetCurrentLevel();
    if (!currentLevel) {
        return CKERR_NOCURRENTLEVEL;
    }

    if (m_Reseted) {
        if (!currentLevel->m_IsReseted)
            Reset();
        currentLevel->ActivateAllScript();
    }

    m_Playing = TRUE;
    ExecuteManagersOnCKPlay();

    m_Reseted = FALSE;
    return CK_OK;
}

CKERROR CKContext::Pause() {
    if (m_Playing) {
        ExecuteManagersOnCKPause();
        m_Playing = FALSE;
    }
    return CK_OK;
}

CKERROR CKContext::Reset() {
    ExecuteManagersOnCKReset();

    m_Playing = FALSE;
    m_Reseted = TRUE;

    CKLevel *currentLevel = GetCurrentLevel();
    if (!currentLevel) return CKERR_NOCURRENTLEVEL;
    m_BehaviorContext.CurrentLevel = currentLevel;

    WarnAllBehaviors(CKM_BEHAVIORRESET);

    // Characters
    int count = m_ObjectManager->GetObjectsCountByClassID(CKCID_CHARACTER);
    CK_ID *idList = m_ObjectManager->GetObjectsListByClassID(CKCID_CHARACTER);
    for (int i = 0; i < count; ++i) {
        CKCharacter *character = (CKCharacter *) GetObject(idList[i]);
        if (character) character->FlushSecondaryAnimations();
    }

    // Synchro Objects
    count = m_ObjectManager->GetObjectsCountByClassID(CKCID_SYNCHRO);
    idList = m_ObjectManager->GetObjectsListByClassID(CKCID_SYNCHRO);
    for (int i = 0; i < count; ++i) {
        CKSynchroObject *synchro = (CKSynchroObject *) GetObject(idList[i]);
        if (synchro) synchro->Reset();
    }

    // Critical Section Objects
    count = m_ObjectManager->GetObjectsCountByClassID(CKCID_CRITICALSECTION);
    idList = m_ObjectManager->GetObjectsListByClassID(CKCID_CRITICALSECTION);
    for (int i = 0; i < count; ++i) {
        CKCriticalSectionObject *cs = (CKCriticalSectionObject *) GetObject(idList[i]);
        if (cs) cs->Reset();
    }

    int stateCount = m_ObjectManager->GetObjectsCountByClassID(CKCID_STATE);
    CK_ID *stateList = m_ObjectManager->GetObjectsListByClassID(CKCID_STATE);
    for (int i = 0; i < stateCount; ++i) {
        CKStateObject *stateObject = (CKStateObject *) m_ObjectManager->m_Objects[stateList[i]];
        if (stateObject)
            stateObject->Reset();
    }

    currentLevel->SetNextActiveScene(nullptr, CK_SCENEOBJECTACTIVITY_SCENEDEFAULT, CK_SCENEOBJECTRESET_RESET);
    currentLevel->Reset();

    CKScene *currentScene = currentLevel->GetCurrentScene();
    if (currentScene) currentScene->Launch();

    ExecuteManagersOnCKPostReset();
    return CK_OK;
}

CKBOOL CKContext::IsPlaying() {
    return m_Playing;
}

CKBOOL CKContext::IsReseted() {
    return m_Reseted;
}

CKERROR CKContext::Process() {
    if (m_Playing) {
        VxTimeProfiler profiler;

        ExecuteManagersPreProcess();

        CKLevel *currentLevel = GetCurrentLevel();
        if (currentLevel) {
            currentLevel->PreProcess();
        }

        m_BehaviorManager->Execute(m_TimeManager->GetLastDeltaTime());

        ExecuteManagersPostProcess();

        m_Stats.ProcessTime = profiler.Current();
    }

    return CK_OK;
}

CKERROR CKContext::ClearAll() {
    m_RunTime = FALSE;
    m_InClearAll = TRUE;
    ExecuteManagersPreClearAll();
    m_ObjectManager->ClearAllObjects();
    m_CurrentLevel = 0;
    m_PVInformation = m_VirtoolsVersion;
    ExecuteManagersPostClearAll();

    for (XManagerArray::Iterator it = m_InactiveManagers.Begin(); it != m_InactiveManagers.End(); ++it) {
        ActivateManager(*it, TRUE);
    }

    return CK_OK;
}

CKBOOL CKContext::IsInClearAll() {
    return m_InClearAll;
}

CKLevel *CKContext::GetCurrentLevel() {
    return (CKLevel *) GetObject(m_CurrentLevel);
}

CKRenderContext *CKContext::GetPlayerRenderContext() {
    return m_BehaviorContext.CurrentRenderContext;
}

CKScene *CKContext::GetCurrentScene() {
    CKLevel *level = GetCurrentLevel();
    return level ? level->GetCurrentScene() : nullptr;
}

void CKContext::SetCurrentLevel(CKLevel *level) {
    m_CurrentLevel = (level != nullptr) ? level->GetID() : 0;
    m_BehaviorContext.CurrentLevel = level;
}

CKParameterIn *CKContext::CreateCKParameterIn(CKSTRING Name, CKParameterType type, CKBOOL Dynamic) {
    if (!m_ParameterManager->CheckParamTypeValidity(type))
        return nullptr;

    CKParameterIn *pIn = (CKParameterIn *) CreateObject(
            CKCID_PARAMETERIN, Name, Dynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK);

    if (pIn)
        pIn->SetType(type);

    return pIn;
}

CKParameterIn *CKContext::CreateCKParameterIn(CKSTRING Name, CKGUID guid, CKBOOL Dynamic) {
    CKParameterType type = m_ParameterManager->ParameterGuidToType(guid);
    return CreateCKParameterIn(Name, type, Dynamic);
}

CKParameterIn *CKContext::CreateCKParameterIn(CKSTRING Name, CKSTRING TypeName, CKBOOL Dynamic) {
    CKParameterType type = m_ParameterManager->ParameterNameToType(TypeName);
    return CreateCKParameterIn(Name, type, Dynamic);
}

CKParameterOut *CKContext::CreateCKParameterOut(CKSTRING Name, CKParameterType type, CKBOOL Dynamic) {
    if (!m_ParameterManager->CheckParamTypeValidity(type)) {
        return (CKParameterOut *) CreateObject(
            CKCID_PARAMETEROUT, Name, Dynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK);
    }

    CKParameterOut *pOut = (CKParameterOut *) CreateObject(
        CKCID_PARAMETEROUT, Name, Dynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK);

    if (pOut)
        pOut->SetType(type);

    return pOut;
}

CKParameterOut *CKContext::CreateCKParameterOut(CKSTRING Name, CKGUID guid, CKBOOL Dynamic) {
    CKParameterType type = m_ParameterManager->ParameterGuidToType(guid);
    return CreateCKParameterOut(Name, type, Dynamic);
}

CKParameterOut *CKContext::CreateCKParameterOut(CKSTRING Name, CKSTRING TypeName, CKBOOL Dynamic) {
    CKParameterType type = m_ParameterManager->ParameterNameToType(TypeName);
    return CreateCKParameterOut(Name, type, Dynamic);
}

CKParameterLocal *CKContext::CreateCKParameterLocal(CKSTRING Name, CKParameterType type, CKBOOL Dynamic) {
    CKParameterLocal *pLocal = nullptr;
    if (m_ParameterManager->CheckParamTypeValidity(type)) {
        pLocal = (CKParameterLocal *) CreateObject(
            CKCID_PARAMETERLOCAL, Name, Dynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK);
        pLocal->SetType(type);
    }
    return pLocal;
}

CKParameterLocal *CKContext::CreateCKParameterLocal(CKSTRING Name, CKGUID guid, CKBOOL Dynamic) {
    CKParameterType type = m_ParameterManager->ParameterGuidToType(guid);
    return CreateCKParameterLocal(Name, type, Dynamic);
}

CKParameterLocal *CKContext::CreateCKParameterLocal(CKSTRING Name, CKSTRING TypeName, CKBOOL Dynamic) {
    CKParameterType type = m_ParameterManager->ParameterNameToType(TypeName);
    return CreateCKParameterLocal(Name, type, Dynamic);
}

CKParameterOperation *CKContext::CreateCKParameterOperation(CKSTRING Name, CKGUID opguid, CKGUID ResGuid, CKGUID p1Guid, CKGUID p2Guid) {
    CKParameterOperation *pOperation = (CKParameterOperation *) CreateObject(CKCID_PARAMETEROPERATION, Name);
    pOperation->Reconstruct(Name, opguid, ResGuid, p1Guid, p2Guid);
    return pOperation;
}

CKFile *CKContext::CreateCKFile() {
    return new CKFile(this);
}

CKERROR CKContext::DeleteCKFile(CKFile *file) {
    if (!file)
        return CKERR_INVALIDPARAMETER;
    delete file;
    return CK_OK;
}

void CKContext::SetInterfaceMode(CKBOOL mode, CKUICALLBACKFCT CallBackFct, void *data) {
    m_InterfaceMode = mode;
    m_UICallBackFct = CallBackFct;
    m_InterfaceModeData = data;
}

CKBOOL CKContext::IsInInterfaceMode() {
    return m_InterfaceMode;
}

CKERROR CKContext::OutputToConsole(CKSTRING str, CKBOOL bBeep) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_OUTTOCONSOLE;
    cbs.DoBeep = bBeep;
    cbs.ConsoleString = str;
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CKERROR CKContext::OutputToConsoleEx(CKSTRING format, ...) {
    if (!m_UICallBackFct)
        return CK_OK;

    char *buf = GetStringBuffer(256);
    if (!buf)
        return CKERR_INVALIDPARAMETER;

    va_list args;
    va_start(args, format);
    vsnprintf(buf, 256, format, args);
    va_end(args);
    return OutputToConsole(buf, FALSE);
}

CKERROR CKContext::OutputToConsoleExBeep(CKSTRING format, ...) {
    if (!m_UICallBackFct)
        return CK_OK;

    char *buf = GetStringBuffer(256);
    if (!buf)
        return CKERR_INVALIDPARAMETER;

    va_list args;
    va_start(args, format);
    vsnprintf(buf, 256, format, args);
    va_end(args);
    return OutputToConsole(buf, TRUE);
}

CKERROR CKContext::OutputToInfo(CKSTRING format, ...) {
    if (!m_UICallBackFct)
        return CK_OK;

    char *buf = GetStringBuffer(256);
    if (!buf)
        return CKERR_INVALIDPARAMETER;

    va_list args;
    va_start(args, format);
    vsnprintf(buf, 256, format, args);
    va_end(args);

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_OUTTOINFOBAR;
    cbs.Param1 = 0;
    cbs.ConsoleString = buf;
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CKERROR CKContext::RefreshBuildingBlocks(const XArray<CKGUID> &iGuids) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_REFRESHBUILDINGBLOCKS;
    cbs.Param1 = (CKDWORD) iGuids.Begin();
    cbs.Param2 = iGuids.Size();
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CKERROR CKContext::ShowSetup(CK_ID id) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_SHOWSETUP;
    cbs.ObjectID = id;
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CK_ID CKContext::ChooseObject(void *dialogParentWnd) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_CHOOSEOBJECT;
    cbs.Param1 = (CKDWORD) dialogParentWnd;
    m_UICallBackFct(cbs, m_InterfaceModeData);
    return cbs.ObjectID;
}

CKERROR CKContext::Select(const XObjectArray &o, CKBOOL clearSelection) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_SELECT;
    cbs.Objects = (XObjectArray *) &o;
    cbs.ClearSelection = clearSelection;
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CKDWORD CKContext::SendInterfaceMessage(CKDWORD reason, CKDWORD param1, CKDWORD param2) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = reason;
    cbs.Param1 = param1;
    cbs.Param2 = param2;
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CKERROR CKContext::UICopyObjects(const XObjectArray &iObjects, CKBOOL iClearClipboard) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_COPYOBJECTS;
    cbs.Objects = (XObjectArray *) &iObjects;
    cbs.ClearSelection = iClearClipboard;
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CKERROR CKContext::UIPasteObjects(const XObjectArray &oObjects) {
    if (!m_UICallBackFct)
        return CK_OK;

    CKUICallbackStruct cbs;
    cbs.Reason = CKUIM_PASTEOBJECTS;
    cbs.Objects = (XObjectArray *) &oObjects;
    return m_UICallBackFct(cbs, m_InterfaceModeData);
}

CKRenderManager *CKContext::GetRenderManager() {
    if (!m_RenderManager)
        m_RenderManager = (CKRenderManager *) GetManagerByGuid(RENDER_MANAGER_GUID);
    return m_RenderManager;
}

CKBehaviorManager *CKContext::GetBehaviorManager() {
    return m_BehaviorManager;
}

CKParameterManager *CKContext::GetParameterManager() {
    return m_ParameterManager;
}

CKMessageManager *CKContext::GetMessageManager() {
    return m_MessageManager;
}

CKTimeManager *CKContext::GetTimeManager() {
    return m_TimeManager;
}

CKAttributeManager *CKContext::GetAttributeManager() {
    return m_AttributeManager;
}

CKPathManager *CKContext::GetPathManager() {
    return m_PathManager;
}

XManagerHashTableIt CKContext::GetManagers() {
    return m_ManagerTable.Begin();
}

CKBaseManager *CKContext::GetManagerByGuid(CKGUID guid) {
    XManagerHashTableIt it = m_ManagerTable.Find(guid);
    if (it == m_ManagerTable.End())
        return nullptr;
    return *it;
}

CKBaseManager *CKContext::GetManagerByName(CKSTRING ManagerName) {
    for (XManagerHashTableIt it = m_ManagerTable.Begin(); it != m_ManagerTable.End(); ++it) {
        CKBaseManager *manager = *it;
        if (!strcmp(manager->GetName(), ManagerName))
            return manager;
    }
    for (int i = 0; i < m_InactiveManagers.Size(); ++i) {
        CKBaseManager *manager = m_InactiveManagers[i];
        if (!strcmp(manager->GetName(), ManagerName))
            return manager;
    }
    return nullptr;
}

int CKContext::GetManagerCount() {
    return m_ManagerTable.Size();
}

CKBaseManager *CKContext::GetManager(int index) {
    int i = 0;
    for (XManagerHashTableIt it = m_ManagerTable.Begin(); it != m_ManagerTable.End(); ++it) {
        if (i++ == index)
            return *it;
        if (i > index)
            break;
    }
    return nullptr;
}

CKBOOL CKContext::IsManagerActive(CKBaseManager *manager) {
    return manager && manager == GetManagerByGuid(manager->GetGuid());
}

CKBOOL CKContext::HasManagerDuplicates(CKBaseManager *manager) {
    if (m_InactiveManagers.Size() == 0)
        return FALSE;
    for (auto it = m_InactiveManagers.Begin(); it != m_InactiveManagers.End(); ++it) {
        if (*it == manager)
            return TRUE;
    }
    return FALSE;
}

void CKContext::ActivateManager(CKBaseManager *manager, CKBOOL activate) {
    if (!manager)
        return;

    CKGUID managerGuid = manager->m_ManagerGuid;
    CKBaseManager *existingManager = GetManagerByGuid(managerGuid);

    if (activate) {
        if (manager == existingManager)
            return;

        if (existingManager) {
            existingManager->OnCKEnd();
            m_ManagerTable.Remove(managerGuid);
            m_InactiveManagers.PushBack(existingManager);
        }

        m_ManagerTable.Insert(managerGuid, manager);
        m_InactiveManagers.Remove(manager);
        manager->OnCKInit();
    } else {
        if (existingManager != manager)
            return;

        manager->OnCKEnd();
        m_ManagerTable.Remove(managerGuid);
        m_InactiveManagers.PushBack(manager);
    }

    BuildSortedLists();
}

int CKContext::GetInactiveManagerCount() {
    return m_InactiveManagers.Size();
}

CKBaseManager *CKContext::GetInactiveManager(int index) {
    if (index < 0 || index >= m_InactiveManagers.Size())
        return nullptr;
    return m_InactiveManagers[index];
}

CKERROR CKContext::RegisterNewManager(CKBaseManager *manager) {
    if (!manager)
        return CKERR_INVALIDPARAMETER;

    XManagerHashTablePair pair = m_ManagerTable.TestInsert(manager->GetGuid(), manager);
    if (!pair.m_New) {
        m_InactiveManagers.PushBack(manager);
        return CKERR_ALREADYPRESENT;
    }

    BuildSortedLists();
    if (m_InitManagerOnRegister)
        manager->OnCKInit();
    return CK_OK;
}

void CKContext::EnableProfiling(CKBOOL enable) {
    m_ProfilingEnabled = enable;
}

CKBOOL CKContext::IsProfilingEnable() {
    return m_ProfilingEnabled;
}

void CKContext::GetProfileStats(CKStats *stats) {
    if (stats)
        *stats = m_ProfileStats;
}

void CKContext::UserProfileStart(CKDWORD UserSlot) {
    if (UserSlot < 8) {
        m_UserProfileTimers[UserSlot].Reset();
    }
}

float CKContext::UserProfileEnd(CKDWORD UserSlot) {
    if (UserSlot < 8) {
        float time = m_UserProfileTime[UserSlot] + m_UserProfileTimers[UserSlot].Current();
        m_UserProfileTime[UserSlot] = time;
        m_Stats.UserProfiles[UserSlot] = time;
        return time;
    }
    return 0.0f;
}

float CKContext::GetLastUserProfileTime(CKDWORD UserSlot) {
    if (UserSlot < 8)
        return m_UserProfileTime[UserSlot];
    return 0.0f;
}

CKSTRING CKContext::GetStringBuffer(int size) {
    if (m_StringBuffer.Length() < size) {
        m_StringBuffer.Resize(size);
    }
    return m_StringBuffer.Str();
}

CKGUID CKContext::GetSecureGuid() {
    CKGUID guid;

    do {
        guid.d1 = (rand() & 0xFFFF) | ((rand() & 0xFFFF) << 16);
        guid.d2 = (rand() & 0xFFFF) | ((rand() & 0xFFFF) << 16);
    } while (CKGetObjectDeclarationFromGuid(guid) ||
        m_ParameterManager->OperationGuidToCode(guid) >= 0 ||
        m_ParameterManager->ParameterGuidToType(guid) >= 0 ||
        GetManagerByGuid(guid) != nullptr);

    return guid;
}

CKDWORD CKContext::GetStartOptions() {
    return m_StartOptions;
}

WIN_HANDLE CKContext::GetMainWindow() {
    return m_MainWindow;
}

int CKContext::GetSelectedRenderEngine() {
    return m_SelectedRenderEngine;
}

void CKContext::SetCompressionLevel(int level) {
    if (level > 0 && level < 10)
        m_CompressionLevel = level;
}

int CKContext::GetCompressionLevel() {
    return m_CompressionLevel;
}

void CKContext::SetFileWriteMode(CK_FILE_WRITEMODE mode) {
    m_FileWriteMode = mode;
}

CK_FILE_WRITEMODE CKContext::GetFileWriteMode() {
    return m_FileWriteMode;
}

CK_TEXTURE_SAVEOPTIONS CKContext::GetGlobalImagesSaveOptions() {
    return m_GlobalImagesSaveOptions;
}

void CKContext::SetGlobalImagesSaveOptions(CK_TEXTURE_SAVEOPTIONS Options) {
    if (Options != CKTEXTURE_USEGLOBAL)
        m_GlobalImagesSaveOptions = Options;
}

CKBitmapProperties *CKContext::GetGlobalImagesSaveFormat() {
    return m_GlobalImagesSaveFormat;
}

void CKContext::SetGlobalImagesSaveFormat(CKBitmapProperties *Format) {
    if (Format && Format->m_Size != 0) {
        delete[] reinterpret_cast<CKBYTE *>(m_GlobalImagesSaveFormat);
        m_GlobalImagesSaveFormat = CKCopyBitmapProperties(Format);
    }
}

CK_SOUND_SAVEOPTIONS CKContext::GetGlobalSoundsSaveOptions() {
    return m_GlobalSoundsSaveOptions;
}

void CKContext::SetGlobalSoundsSaveOptions(CK_SOUND_SAVEOPTIONS Options) {
    if (Options != CKSOUND_USEGLOBAL)
        m_GlobalSoundsSaveOptions = Options;
}

CKERROR CKContext::Load(CKSTRING FileName, CKObjectArray *liste, CK_LOAD_FLAGS LoadFlags, CKGUID *ReaderGuid) {
    if (!FileName)
        return CKERR_INVALIDPARAMETER;
    m_LastFileLoaded = FileName;
    return g_ThePluginManager.Load(this, FileName, liste, LoadFlags, nullptr, ReaderGuid);
}

CKERROR CKContext::Load(int BufferSize, void *MemoryBuffer, CKObjectArray *ckarray, CK_LOAD_FLAGS LoadFlags) {
    if (!MemoryBuffer || !ckarray)
        return CKERR_INVALIDPARAMETER;

    CKFile *file = CreateCKFile();
    if (!file)
        return CKERR_INVALIDFILE;
    CKERROR err = file->Load(MemoryBuffer, BufferSize, ckarray, LoadFlags);
    file->UpdateAndApplyAnimationsTo(nullptr);
    DeleteCKFile(file);
    return err;
}

CKSTRING CKContext::GetLastFileLoaded() {
    return (CKSTRING) m_LastFileLoaded.CStr();
}

CKSTRING CKContext::GetLastCmoLoaded() {
    return (CKSTRING) m_LastCmoLoaded.CStr();
}

void CKContext::SetLastCmoLoaded(CKSTRING str) {
    if (!str) return;
    size_t len = strlen(str);
    if (len > 4 && !strcmp(&str[len - 4], ".cmo"))
        m_LastCmoLoaded = str;
}

CKERROR CKContext::GetFileInfo(CKSTRING FileName, CKFileInfo *FileInfo) {
    if (!FileInfo)
        return CKERR_INVALIDPARAMETER;
    if (!FileName)
        return CKERR_INVALIDFILE;

    VxMemoryMappedFile file(FileName);
    if (!file.IsValid())
        return CKERR_INVALIDFILE;

    return GetFileInfo(file.GetFileSize(), file.GetBase(), FileInfo);
}

CKERROR CKContext::GetFileInfo(int BufferSize, void *MemoryBuffer, CKFileInfo *FileInfo) {
    if (!MemoryBuffer || !FileInfo)
        return CKERR_INVALIDPARAMETER;

    if (BufferSize < 64)
        return CKERR_INVALIDFILE;

    int header1[8];
    int header2[8];

    memcpy(header1, MemoryBuffer, sizeof(header1));
    memcpy(header2, (char *) MemoryBuffer + 32, sizeof(header2));

    if (header1[5]) {
        memset(header1, 0, sizeof(header1));
        memset(header2, 0, sizeof(header2));
    }

    if (header1[4] < 5)
        memset(header2, 0, sizeof(header2));

    FileInfo->ProductVersion = header2[5];
    FileInfo->ProductBuild = header2[6];
    FileInfo->FileWriteMode = header1[6];
    FileInfo->CKVersion = header1[3];
    FileInfo->FileVersion = header1[4];
    FileInfo->FileSize = BufferSize;
    FileInfo->ManagerCount = header2[2];
    FileInfo->ObjectCount = header2[3];
    FileInfo->MaxIDSaved = header2[4];
    FileInfo->Hdr1PackSize = header1[7];
    FileInfo->Hdr1UnPackSize = header2[1];
    FileInfo->DataUnPackSize = header2[1];
    FileInfo->DataPackSize = header2[0];
    FileInfo->Crc = header1[2];
    return CK_OK;
}

CKERROR CKContext::Save(CKSTRING FileName, CKObjectArray *liste, CKDWORD SaveFlags, CKDependencies *dependencies, CKGUID *ReaderGuid) {
    CKObjectArray *saveList = liste;
    CKObjectArray *tempObjectArray = nullptr;

    if (dependencies) {
        tempObjectArray = CreateCKObjectArray();
        if (!tempObjectArray) return CKERR_OUTOFMEMORY;

        CKDependenciesContext depContext(this);
        depContext.SetOperationMode(CK_DEPENDENCIES_SAVE);
        depContext.StartDependencies(dependencies);

        liste->Reset();
        while (!liste->EndOfList()) {
            CK_ID objId = liste->GetDataId();
            depContext.AddObjects(&objId, 1);
            liste->Next();
        }

        depContext.FillDependencies();

        const XHashID &depMap = depContext.GetDependenciesMap();
        for (XHashID::ConstIterator it = depMap.Begin(); it != depMap.End(); ++it) {
            tempObjectArray->InsertRear(it.GetKey());
        }

        depContext.StopDependencies();
        saveList = tempObjectArray;
    }

    CKERROR result = g_ThePluginManager.Save(this, FileName, saveList, SaveFlags, ReaderGuid);

    if (tempObjectArray)
        DeleteCKObjectArray(tempObjectArray);

    return result;
}

CKERROR CKContext::LoadAnimationOnCharacter(CKSTRING FileName, CKObjectArray *liste, CKCharacter *carac,
                                            CKGUID *ReaderGuid, CKBOOL AsDynamicObjects) {
    CKDWORD flags = CK_LOAD_ANIMATION;
    if (AsDynamicObjects)
        flags |= CK_LOAD_AS_DYNAMIC_OBJECT;
    return g_ThePluginManager.Load(this, FileName, liste, (CK_LOAD_FLAGS) flags, carac, ReaderGuid);
}

CKERROR CKContext::LoadAnimationOnCharacter(int BufferSize, void *MemoryBuffer, CKObjectArray *ckarray,
                                            CKCharacter *carac, CKBOOL AsDynamicObjects) {
    if (!MemoryBuffer || !ckarray)
        return CKERR_INVALIDPARAMETER;
    CKFile *file = CreateCKFile();
    if (!file)
        return CKERR_INVALIDFILE;

    CKDWORD flags = CK_LOAD_ANIMATION;
    if (AsDynamicObjects)
        flags |= CK_LOAD_AS_DYNAMIC_OBJECT;
    CKERROR err = file->Load(MemoryBuffer, BufferSize, ckarray, (CK_LOAD_FLAGS) flags);
    file->UpdateAndApplyAnimationsTo(carac);
    DeleteCKFile(file);
    return err;
}

void CKContext::SetAutomaticLoadMode(CK_LOADMODE GeneralMode, CK_LOADMODE _3DObjectsMode, CK_LOADMODE MeshMode,
                                     CK_LOADMODE MatTexturesMode) {
    m_GeneralLoadMode = GeneralMode;
    m_3DObjectsLoadMode = _3DObjectsMode;
    m_MeshLoadMode = MeshMode;
    m_MatTexturesLoadMode = MatTexturesMode;
}

void CKContext::SetUserLoadCallback(CK_USERLOADCALLBACK fct, void *Arg) {
    m_UserLoadCallBack = fct;
    m_UserLoadCallBackArgs = Arg;
}

CK_LOADMODE CKContext::LoadVerifyObjectUnicity(CKSTRING OldName, CK_CLASSID Cid, const CKSTRING NewName, CKObject **newobj) {
    if (!OldName) return CKLOAD_OK;

    CKClassDesc &classDesc = g_CKClassInfo[Cid];
    if (classDesc.DefaultOptions & CK_GENERALOPTIONS_NODUPLICATENAMECHECK)
        return CKLOAD_OK;

    CKObject *existingObj = GetObjectByNameAndClass(OldName, Cid);
    if (!existingObj) return CKLOAD_OK;

    if (newobj) *newobj = existingObj;

    if (classDesc.DefaultOptions & CK_GENERALOPTIONS_AUTOMATICUSECURRENT)
        return CKLOAD_USECURRENT;

    if (m_UserLoadCallBack)
        return m_UserLoadCallBack(Cid, OldName, NewName, newobj, m_UserLoadCallBackArgs);

    if (!(classDesc.DefaultOptions & CK_GENERALOPTIONS_CANUSECURRENTOBJECT)) {
        CK_LOADMODE autoMode = CKLOAD_INVALID;
        if (CKIsChildClassOf(Cid, CKCID_3DENTITY)) {
            autoMode = m_3DObjectsLoadMode;
        } else {
            autoMode = m_GeneralLoadMode;
        }

        if (autoMode != CKLOAD_INVALID) {
            if (autoMode == CKLOAD_RENAME) GetSecureName(NewName, OldName, Cid);
            return autoMode;
        }

        CK_OBJECTCREATION_OPTIONS globalOption = (CK_OBJECTCREATION_OPTIONS) m_GeneralRenameOption;
        if (globalOption != CK_OBJECTCREATION_NONAMECHECK) {
            if (globalOption == CKLOAD_RENAME) {
                GetSecureName(NewName, OldName, Cid);
                if (newobj) *newobj = nullptr;
                return CKLOAD_RENAME;
            }
            if (globalOption == CKLOAD_USECURRENT) {
                if (newobj) *newobj = existingObj;
                return CKLOAD_USECURRENT;
            }
            return (CK_LOADMODE) globalOption;
        }
    } else {
        // It can use current object
        CK_LOADMODE autoMode = CKLOAD_INVALID;
        if (CKIsChildClassOf(Cid, CKCID_MESH)) {
            autoMode = m_MeshLoadMode;
        } else if (CKIsChildClassOf(Cid, CKCID_MATERIAL) || CKIsChildClassOf(Cid, CKCID_TEXTURE)) {
            autoMode = m_MatTexturesLoadMode;
        }

        if (autoMode != CKLOAD_INVALID) {
            if (autoMode == CKLOAD_RENAME) GetSecureName(NewName, OldName, Cid);
            return autoMode;
        }

        CK_OBJECTCREATION_OPTIONS globalOption = (CK_OBJECTCREATION_OPTIONS) m_MatTexturesRenameOption;
        if (globalOption != CK_OBJECTCREATION_NONAMECHECK) {
            if (globalOption == CKLOAD_RENAME) {
                GetSecureName(NewName, OldName, Cid);
                if (newobj) *newobj = nullptr;
                return CKLOAD_RENAME;
            }
            if (globalOption == CKLOAD_USECURRENT) {
                if (newobj) *newobj = existingObj;
                return CKLOAD_USECURRENT;
            }
            return (CK_LOADMODE) globalOption;
        }
    }

    // Fallback to UI Dialog if available
    CKInterfaceManager *im = (CKInterfaceManager *) GetManagerByGuid(INTERFACE_MANAGER_GUID);
    int choice = im ? im->DoRenameDialog(OldName, Cid) : 0; // 0 is "Replace" in the dialog

    switch (choice) {
    case 0: // Replace
        return CKLOAD_OK;
    case 1: // Replace All
        if (classDesc.DefaultOptions & CK_GENERALOPTIONS_CANUSECURRENTOBJECT)
            m_MatTexturesRenameOption = CKLOAD_REPLACE;
        else m_GeneralRenameOption = CKLOAD_REPLACE;
        return CKLOAD_OK;
    case 2: // Rename
        GetSecureName(NewName, OldName, Cid);
        if (newobj) *newobj = nullptr;
        return CKLOAD_RENAME;
    case 3: // Rename All
        if (classDesc.DefaultOptions & CK_GENERALOPTIONS_CANUSECURRENTOBJECT)
            m_MatTexturesRenameOption = CKLOAD_RENAME;
        else m_GeneralRenameOption = CKLOAD_RENAME;
        GetSecureName(NewName, OldName, Cid);
        if (newobj) *newobj = nullptr;
        return CKLOAD_RENAME;
    case 4: // Use Existing
        if (newobj) *newobj = existingObj;
        return CKLOAD_USECURRENT;
    case 5: // Use Existing for All
        if (classDesc.DefaultOptions & CK_GENERALOPTIONS_CANUSECURRENTOBJECT)
            m_MatTexturesRenameOption = CKLOAD_USECURRENT;
        else m_GeneralRenameOption = CKLOAD_USECURRENT;
        if (newobj) *newobj = existingObj;
        return CKLOAD_USECURRENT;
    default:
        return CKLOAD_OK;
    }
}

CKBOOL CKContext::IsInLoad() {
    return m_InLoad;
}

CKBOOL CKContext::IsInSave() {
    return m_Saving;
}

CKBOOL CKContext::IsRunTime() {
    return m_RunTime;
}

void CKContext::ClearManagersData() {
    CKRenderManager *renderManager = GetRenderManager();
    if (renderManager) {
        int count = renderManager->GetRenderContextCount();
        while (count > 0) {
            CKRenderContext *context = renderManager->GetRenderContext(0);
            renderManager->DestroyRenderContext(context);
            count = renderManager->GetRenderContextCount();
        }
    }

    m_ObjectManager->DeleteAllObjects();
    ExecuteManagersOnCKEnd();

    for (XManagerHashTableIt it = m_ManagerTable.Begin(); it != m_ManagerTable.End(); ++it) {
        CKBaseManager *manager = *it;
        if (manager) delete manager;
    }
    m_ManagerTable.Clear();

    for (XManagerArray::Iterator it = m_InactiveManagers.Begin(); it != m_InactiveManagers.End(); ++it) {
        CKBaseManager *manager = *it;
        if (manager) delete manager;
    }
    m_InactiveManagers.Clear();
}

void CKContext::ExecuteManagersOnCKInit() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnCKInit];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnCKInit();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnCKEnd() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnCKEnd];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnCKEnd();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPreProcess() {
    // Reset processing time for all managers before the frame starts
    for (XManagerHashTableIt it = m_ManagerTable.Begin(); it != m_ManagerTable.End(); ++it) {
        CKBaseManager *manager = *it;
        manager->m_ProcessingTime = 0.0f;
    }

    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PreProcess];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        VxTimeProfiler profiler;
        m_CurrentManager->PreProcess();
        CKBaseManager *manager = *it;
        manager->m_ProcessingTime += profiler.Current();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPostProcess() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PostProcess];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        VxTimeProfiler profiler;
        m_CurrentManager->PostProcess();
        CKBaseManager *manager = *it;
        manager->m_ProcessingTime += profiler.Current();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPreClearAll() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PreClearAll];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PreClearAll();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPostClearAll() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PostClearAll];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PostClearAll();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnSequenceDeleted(CK_ID *objids, int count) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnSequenceDeleted];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->SequenceDeleted(objids, count);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnSequenceToBeDeleted(CK_ID *objids, int count) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnSequenceToBeDeleted];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->SequenceToBeDeleted(objids, count);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnCKReset() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnCKReset];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnCKReset();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnCKPostReset() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnCKPostReset];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnCKPostReset();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnCKPause() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnCKPause];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnCKPause();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnCKPlay() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnCKPlay];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnCKPlay();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPreLaunchScene(CKScene *OldScene, CKScene *NewScene) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PreLaunchScene];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PreLaunchScene(OldScene, NewScene);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPostLaunchScene(CKScene *OldScene, CKScene *NewScene) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PostLaunchScene];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PostLaunchScene(OldScene, NewScene);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPreLoad() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PreLoad];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PreLoad();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPostLoad() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PostLoad];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PostLoad();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPreSave() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PreSave];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PreSave();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersPostSave() {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_PostSave];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->PostSave();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnSequenceAddedToScene(CKScene *scn, CK_ID *objids, int coun) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnSequenceAddedToScene];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->SequenceAddedToScene(scn, objids, coun);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnSequenceRemovedFromScene(CKScene *scn, CK_ID *objids, int coun) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnSequenceRemovedFromScene];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++
         it) {
        m_CurrentManager = *it;
        m_CurrentManager->SequenceRemovedFromScene(scn, objids, coun);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnPreCopy(CKDependenciesContext *context) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnPreCopy];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnPreCopy(*context);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnPostCopy(CKDependenciesContext *context) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnPostCopy];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        m_CurrentManager->OnPostCopy(*context);
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnPreRender(CKRenderContext *dev) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnPreRender];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        VxTimeProfiler profiler;
        m_CurrentManager->OnPreRender(dev);
        CKBaseManager *manager = *it;
        manager->m_ProcessingTime += profiler.Current();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnPostRender(CKRenderContext *dev) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnPostRender];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        VxTimeProfiler profiler;
        m_CurrentManager->OnPostRender(dev);
        CKBaseManager *manager = *it;
        manager->m_ProcessingTime += profiler.Current();
    }
    m_CurrentManager = nullptr;
}

void CKContext::ExecuteManagersOnPostSpriteRender(CKRenderContext *dev) {
    XManagerArray &managers = m_ManagerList[CKMANAGER_INDEX_OnPostSpriteRender];
    for (XManagerArray::Iterator it = managers.Begin(); it != managers.End(); ++it) {
        m_CurrentManager = *it;
        VxTimeProfiler profiler;
        m_CurrentManager->OnPostSpriteRender(dev);
        CKBaseManager *manager = *it;
        manager->m_ProcessingTime += profiler.Current();
    }
    m_CurrentManager = nullptr;
}

void CKContext::AddProfileTime(CK_PROFILE_CATEGORY cat, float time) {
    switch (cat) {
    case CK_PROFILE_RENDERTIME:
        m_Stats.RenderTime += time;
        break;
    case CK_PROFILE_IKTIME:
        m_Stats.IKManagement += time;
        break;
    case CK_PROFILE_ANIMATIONTIME:
        m_Stats.AnimationManagement += time;
        break;
    }
}

CKERROR CKContext::ProcessDebugStart(float delta_time) {
    ExecuteManagersPreProcess();
    CKLevel *level = GetCurrentLevel();
    if (level)
        level->PreProcess();
    return m_BehaviorManager->ExecuteDebugStart(delta_time);
}

CKBOOL CKContext::ProcessDebugStep() {
    if (m_Playing)
        return m_DebugContext->DebugStep();
    else
        return FALSE;
}

CKERROR CKContext::ProcessDebugEnd() {
    if (!m_Playing)
        return CKERR_PAUSED;
    if (!m_DebugContext->InDebug)
        return CKERR_PAUSED;
    m_DebugContext->Clear();
    ExecuteManagersPostProcess();
    return CK_OK;
}

CKDebugContext *CKContext::GetDebugContext() {
    return m_DebugContext;
}

void CKContext::SetVirtoolsVersion(CK_VIRTOOLS_VERSION ver, CKDWORD Build) {
    m_VirtoolsVersion = ver;
    m_VirtoolsBuild = Build;
    m_PVInformation = ver;
}

int CKContext::GetPVInformation() {
    return m_PVInformation;
}

CKBOOL CKContext::IsInDynamicCreationMode() {
    return m_InDynamicCreationMode;
}

int CKContext::WarnAllBehaviors(CKDWORD Message) {
    const int behaviorCount = m_ObjectManager->GetObjectsCountByClassID(CKCID_BEHAVIOR);
    CK_ID *behaviorList = m_ObjectManager->GetObjectsListByClassID(CKCID_BEHAVIOR);

    int result = 0;
    for (int i = 0; i < behaviorCount; ++i) {
        CKBehavior *beh = (CKBehavior *) m_ObjectManager->GetObject(behaviorList[i]);
        if (!beh)
            continue;

        CKObject *owner = beh->GetOwner();
        if (!owner)
            continue;

        result = beh->CallCallbackFunction(Message);
    }
    return result;
}

void CKContext::GetSecureName(CKSTRING secureName, const char *name, CK_CLASSID cid) {
    int highestNumber = -1;
    XString baseName(name);

    CK_ID *objList = GetObjectsListByClassID(cid);
    const int objCount = GetObjectsCountByClassID(cid);

    for (int i = 0; i < objCount; ++i) {
        CKObject *obj = GetObject(objList[i]);
        if (!obj)
            continue;

        const char *objName = obj->GetName();
        if (!objName)
            continue;

        if (strncmp(objName, baseName.CStr(), baseName.Length()) == 0) {
            if (objName[baseName.Length()] != '\0') {
                int currentNum = atoi(&objName[baseName.Length()]);
                if (currentNum > highestNumber) {
                    highestNumber = currentNum;
                }
            }
        }
    }

    sprintf(secureName, "%s%03d", name, highestNumber + 1);
}

void CKContext::GetObjectSecureName(XString &secureName, CKSTRING name, CK_CLASSID cid) {
    XString baseName(name);
    int highestNumber = -1;

    XWORD len = baseName.Length();
    if (len > 3) {
        const char *str = baseName.CStr();
        if (isdigit(str[len - 3]) && isdigit(str[len - 2]) && isdigit(str[len - 1])) {
            highestNumber = atoi(str + (len - 3));
            baseName.Crop(0, len - 3);
        }
    }

    CK_ID *objList = GetObjectsListByClassID(cid);
    const int objCount = GetObjectsCountByClassID(cid);

    for (int i = 0; i < objCount; ++i) {
        CKObject *obj = GetObject(objList[i]);
        if (!obj || !obj->GetName())
            continue;

        const char *objName = obj->GetName();
        XWORD baseLen = baseName.Length();

        if (strncmp(objName, baseName.CStr(), baseLen) == 0) {
            const char *suffix = objName + baseLen;

            if (*suffix != '\0') {
                int currentNum = atoi(suffix);
                if (currentNum > highestNumber) {
                    highestNumber = currentNum;
                }
            } else {
                if (highestNumber < 0) {
                    highestNumber = 0;
                }
            }
        }
    }

    // Step 4: Constructing the final secure name.
    secureName = baseName;
    if (highestNumber != -1) {
        char numBuffer[4];
        sprintf(numBuffer, "%03d", highestNumber + 1);
        secureName << numBuffer;
    }
}

int CKContext::PrepareDestroyObjects(CK_ID *obj_ids, int Count, CKDWORD Flags, CKDependencies *Dependencies) {
    if (Count <= 0)
        return CKERR_INVALIDPARAMETER;

    const bool hasPendingDependencies = m_DependenciesContext.m_MapID.Size() != 0;
    if (!hasPendingDependencies) {
        m_DestroyObjectFlag = Flags;
    } else {
        m_DestroyObjectFlag &= Flags;
    }

    m_DependenciesContext.SetOperationMode(CK_DEPENDENCIES_DELETE);
    if (Dependencies) m_DependenciesContext.StartDependencies(Dependencies);

    for (int i = 0; i < Count; ++i) {
        CKObject *obj = GetObject(obj_ids[i]);
        if (obj) {
            obj->PrepareDependencies(m_DependenciesContext);
        }
    }

    if (Dependencies) m_DependenciesContext.StopDependencies();

    return hasPendingDependencies ? CKERR_INVALIDPARAMETER : CK_OK;
}

int CKContext::FinishDestroyObjects(CKDWORD Flags) {
    XHashID &depMap = m_DependenciesContext.m_MapID;
    if (depMap.Size() == 0) return CK_OK;

    XObjectArray objectsToDelete;
    objectsToDelete.Resize(depMap.Size());
    int i = 0;
    for (XHashID::Iterator it = depMap.Begin(); it != depMap.End(); ++it) {
        CK_ID objId = it.GetKey();
        objectsToDelete[i++] = objId;
    }

    m_DependenciesContext.Clear();
    m_ObjectManager->DeleteObjects(objectsToDelete.Begin(), objectsToDelete.Size(), 0, Flags);

    return CK_OK;
}

void CKContext::BuildSortedLists() {
    for (int i = 0; i < 32; ++i) {
        m_ManagerList[i].Clear();
    }

    for (auto it = m_ManagerTable.Begin(); it != m_ManagerTable.End(); ++it) {
        CKBaseManager *manager = *it;
        for (int i = 0; i < 32; ++i) {
            if (manager->GetValidFunctionsMask() & (1 << i)) {
                m_ManagerList[i].PushBack(manager);
            }
        }
    }
}

void CKContext::DeferredDestroyObjects(CK_ID *obj_ids, int Count, CKDependencies *Dependencies, CKDWORD Flags) {
    CKDeferredDeletion *deferred = m_ObjectManager->MatchDeletion(Dependencies, Flags);
    if (!deferred) {
        deferred = new CKDeferredDeletion();
        deferred->m_DependenciesPtr = nullptr;
        deferred->m_Flags = Flags;
        m_ObjectManager->RegisterDeletion(deferred);
    }

    CKDependencies &dependencies = deferred->m_Dependencies;
    for (int i = 0; i < Count; ++i) {
        dependencies.PushBack(obj_ids[i]);
    }

    if (Dependencies) {
        CKDependencies *dep = new CKDependencies();
        *dep = *Dependencies;
        deferred->m_DependenciesPtr = dep;
    }
}

void *CKContext::AllocateMemoryPool(int count, int &index) {
    int newIndex = m_MemoryPoolMask.GetUnsetBitPosition(0);
    while (m_MemoryPools.Size() <= newIndex) {
        m_MemoryPools.PushBack(new VxMemoryPool);
    }

    VxMemoryPool *pool = m_MemoryPools[newIndex];
    if (pool) {
        pool->Allocate(count);
        m_MemoryPoolMask.Set(newIndex);
        index = newIndex;
        return pool->Buffer();
    }

    return nullptr;
}

void CKContext::ReleaseMemoryPool(int index) {
    if (index >= 0)
        m_MemoryPoolMask.Unset(index);
}

CKContext::CKContext(WIN_HANDLE iWin, int iRenderEngine, CKDWORD Flags) : m_DependenciesContext(this) {
    field_498 = 0;
    field_494 = nullptr;
    field_4A0 = 0;
    field_49C = nullptr;

    m_MainWindow = iWin;
    m_InitManagerOnRegister = FALSE;
    m_RunTime = FALSE;
    m_StartOptions = Flags;
    m_InterfaceMode = 0;
    m_VirtoolsBuild = 0x2010001;
    m_VirtoolsVersion = 0;
    m_UICallBackFct = nullptr;
    m_InterfaceModeData = nullptr;
    m_DeferDestroyObjects = 0;
    m_Playing = FALSE;
    m_Reseted = TRUE;
    m_InLoad = FALSE;
    m_Saving = FALSE;
    m_InClearAll = FALSE;
    m_Init = FALSE;
    m_InDynamicCreationMode = FALSE;
    m_GlobalImagesSaveOptions = CKTEXTURE_RAWDATA;
    m_GlobalSoundsSaveOptions = CKSOUND_EXTERNAL;
    m_FileWriteMode = CKFILE_UNCOMPRESSED;
    m_GlobalImagesSaveFormat = new CKBitmapProperties;
    memset(m_GlobalImagesSaveFormat, 0, sizeof(CKBitmapProperties));
    m_GlobalImagesSaveFormat->m_Ext = CKFileExtension("bmp");
    m_GeneralLoadMode = CKLOAD_INVALID;
    m_3DObjectsLoadMode = CKLOAD_INVALID;
    m_MeshLoadMode = CKLOAD_INVALID;
    m_MatTexturesLoadMode = CKLOAD_INVALID;
    m_UserLoadCallBack = nullptr;
    m_UserLoadCallBackArgs = nullptr;

    field_3C8 = new char[260];
    field_3CC = new char[260];
    m_GeneralRenameOption = 0;
    m_MatTexturesRenameOption = 0;

    m_CurrentLevel = 0;
    m_CompressionLevel = 5;
    m_CurrentManager = nullptr;

    m_ObjectManager = new CKObjectManager(this);
    m_ParameterManager = new CKParameterManager(this);
    m_AttributeManager = new CKAttributeManager(this);
    m_TimeManager = new CKTimeManager(this);
    m_MessageManager = new CKMessageManager(this);
    m_BehaviorManager = new CKBehaviorManager(this);
    m_PathManager = new CKPathManager(this);
    m_DebugContext = new CKDebugContext(this);

    m_BehaviorContext.AttributeManager = m_AttributeManager;
    m_BehaviorContext.MessageManager = m_MessageManager;
    m_BehaviorContext.ParameterManager = m_ParameterManager;
    m_BehaviorContext.TimeManager = m_TimeManager;
    m_BehaviorContext.Context = this;

    m_ProfilingEnabled = FALSE;
    m_SelectedRenderEngine = iRenderEngine;
    m_RenderManager = nullptr;

    m_MemoryPools.Resize(8);
    for (XArray<VxMemoryPool *>::Iterator it = m_MemoryPools.Begin();
         it != m_MemoryPools.End(); ++it) {
        *it = new VxMemoryPool;
    }

    memset(&m_Stats, 0, sizeof(CKStats));
    memset(&m_ProfileStats, 0, sizeof(CKStats));
}

CKContext::~CKContext() {
    m_Init = TRUE;
    delete[] reinterpret_cast<CKBYTE *>(m_GlobalImagesSaveFormat);

    delete[] field_3C8;
    field_3C8 = nullptr;
    delete[] field_3CC;
    field_3CC = nullptr;

    if (m_DebugContext)
        delete m_DebugContext;
    m_DebugContext = nullptr;

    for (XArray<VxMemoryPool *>::Iterator it = m_MemoryPools.Begin();
         it != m_MemoryPools.End(); ++it) {
        delete *it;
    }

    ClearManagersData();
}
