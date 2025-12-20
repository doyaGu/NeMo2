#include "CKBehaviorManager.h"

#include "CKBehavior.h"
#include "CKDebugContext.h"
#include "CKStateChunk.h"
#include "CKCharacter.h"

CKERROR CKBehaviorManager::Execute(float delta) {
    VxTimeProfiler totalProfiler;
    VxTimeProfiler behaviorProfiler;

    // Setup execution context
    m_Context->m_DeferDestroyObjects = TRUE;
    m_Context->m_BehaviorContext.CurrentScene = m_Context->GetCurrentScene();
    m_Context->m_BehaviorContext.DeltaTime = delta;

    // Update object activation states
    ManageObjectsActivity();

    // Process behavior activation first
    for (auto it = m_BeObjects.Begin(); it != m_BeObjects.End(); ++it) {
        CKBeObject *beo = (CKBeObject *) *it;
        if (!beo) continue;

        // Re-query the count each time because Activate() may add/remove scripts mid-loop.
        for (int i = 0; i < beo->GetScriptCount(); ++i) {
            CKBehavior *beh = beo->GetScript(i);
            const CKDWORD flags = beh->GetFlags();
            if (flags & (CKBEHAVIOR_ACTIVATENEXTFRAME | CKBEHAVIOR_DEACTIVATENEXTFRAME)) {
                CKBOOL active = (flags & CKBEHAVIOR_ACTIVATENEXTFRAME) != 0;
                CKBOOL reset = (flags & CKBEHAVIOR_RESETNEXTFRAME) != 0;
                beh->Activate(active, reset);
                beh->ModifyFlags(0, CKBEHAVIOR_ACTIVATENEXTFRAME | CKBEHAVIOR_DEACTIVATENEXTFRAME | CKBEHAVIOR_RESETNEXTFRAME);
            }
        }
    }

    // Main execution loop
    for (auto it = m_BeObjects.Begin(); it != m_BeObjects.End(); ++it) {
        CKBeObject *obj = (CKBeObject *)*it;
        if (!obj) continue;

        if (m_Context->m_ProfilingEnabled) {
            behaviorProfiler.Reset();
        }

        // Special handling for characters
        if (obj->GetClassID() == CKCID_CHARACTER) {
            CKCharacter *character = (CKCharacter *) obj;
            if (character->IsAutomaticProcess()) {
                character->ProcessAnimation(delta);
            }
        }

        // Execute object behaviors
        float time = 0.0f;
        if (m_Context->m_ProfilingEnabled)
            time = behaviorProfiler.Current();

        obj->ExecuteBehaviors(delta);

        obj->m_LastExecutionTime += time;
        m_Context->m_Stats.BehaviorCodeExecution += time;
    }

    m_Context->m_Stats.TotalBehaviorExecution = totalProfiler.Current();
    m_Context->m_DeferDestroyObjects = FALSE;
    return CK_OK;
}

CKERROR CKBehaviorManager::ExecuteDebugStart(float delta) {
    m_Context->m_Stats.TotalBehaviorExecution = 0.0f;
    m_Context->m_BehaviorContext.CurrentScene = m_Context->GetCurrentScene();
    m_Context->m_BehaviorContext.DeltaTime = delta;
    ManageObjectsActivity();
    m_Context->m_DebugContext->Init(m_BeObjects, delta);
    return CK_OK;
}

void CKBehaviorManager::ManageObjectsActivity() {
    if (m_BeObjectNextFrame.Size() == 0)
        return;

    CKBOOL changed = FALSE;
    for (auto it = m_BeObjectNextFrame.Begin(); it != m_BeObjectNextFrame.End(); ++it) {
        CKBeObject *beo = (CKBeObject *) m_Context->GetObject(it.GetKey());

        if (!beo) {
            continue;
        }

        if (*it < 0) {
            beo->ResetExecutionTime();
            m_BeObjects.Remove(beo);
        } else if (m_BeObjects.AddIfNotHere(beo)) {
            changed = TRUE;
        }
    }

    m_BeObjectNextFrame.Clear();
    if (changed) {
        SortObjects();
    }
}

int CKBehaviorManager::GetObjectsCount() {
    return m_BeObjects.Size();
}

CKBeObject *CKBehaviorManager::GetObject(int pos) {
    if (pos < 0 || pos >= m_BeObjects.Size())
        return nullptr;
    return (CKBeObject *) m_BeObjects[pos];
}

CKERROR CKBehaviorManager::AddObject(CKBeObject *beo) {
    if (!beo)
        return CKERR_INVALIDPARAMETER;

    if (m_BeObjects.AddIfNotHere(beo)) {
        SortObjects();
    }
    return CK_OK;
}

void CKBehaviorManager::SortObjects() {
    m_BeObjects.Sort(CKBeObject::BeObjectPrioritySort);
}

int CKBehaviorManager::RemoveAllObjects() {
    m_BeObjects.Clear();
    m_BeObjectNextFrame.Clear();

    for (auto it = m_Behaviors.Begin(); it != m_Behaviors.End(); ++it) {
        CKBehavior *beh = (CKBehavior *) *it;
        if (beh)
            beh->m_Flags &= ~CKBEHAVIOR_EXECUTEDLASTFRAME;
    }
    m_Behaviors.Clear();
    return 0;
}

int CKBehaviorManager::GetBehaviorMaxIteration() {
    return m_BehaviorMaxIteration;
}

void CKBehaviorManager::SetBehaviorMaxIteration(int n) {
    m_BehaviorMaxIteration = n;
}

int CKBehaviorManager::AddObjectNextFrame(CKBeObject *beo) {
    if (!beo)
        return CKERR_INVALIDOBJECT;

    if (!m_BeObjectNextFrame.Insert(beo->GetID(), 1, FALSE)) {
        int *it = m_BeObjectNextFrame.FindPtr(beo->GetID());
        if (it)
            *it += 1;
    }
    return CK_OK;
}

int CKBehaviorManager::RemoveObjectNextFrame(CKBeObject *beo) {
    if (!beo)
        return CKERR_INVALIDOBJECT;

    if (!m_BeObjectNextFrame.Insert(beo->GetID(), -1, FALSE)) {
        int *it = m_BeObjectNextFrame.FindPtr(beo->GetID());
        if (it)
            *it -= 1;
    }
    return CK_OK;
}

CKStateChunk *CKBehaviorManager::SaveData(CKFile *SavedFile) {
    if (m_BehaviorMaxIteration == 8000)
        return nullptr;
    CKStateChunk *chunk = CreateCKStateChunk(0, SavedFile);
    chunk->StartWrite();
    chunk->WriteIdentifier(0x58D621AE);
    chunk->WriteInt(m_BehaviorMaxIteration);
    chunk->CloseChunk();
    return chunk;
}

CKERROR CKBehaviorManager::LoadData(CKStateChunk *chunk, CKFile *LoadedFile) {
    if (!chunk) {
        m_BehaviorMaxIteration = 8000;
        return CKERR_INVALIDPARAMETER;
    }

    chunk->StartRead();
    chunk->SeekIdentifier(0x58D621AE);
    m_BehaviorMaxIteration = chunk->ReadInt();
    return CK_OK;
}

CKERROR CKBehaviorManager::OnCKPlay() {
    m_Context->WarnAllBehaviors(CKM_BEHAVIORRESUME);
    return CK_OK;
}

CKERROR CKBehaviorManager::OnCKPause() {
    m_Context->WarnAllBehaviors(CKM_BEHAVIORPAUSE);
    return CK_OK;
}

CKERROR CKBehaviorManager::PreProcess() {
    for (auto it = m_Behaviors.Begin(); it != m_Behaviors.End(); ++it) {
        CKBehavior *beh = (CKBehavior *) *it;
        if (beh)
            beh->m_Flags &= ~CKBEHAVIOR_EXECUTEDLASTFRAME;
    }
    m_Behaviors.Resize(0);
    return CK_OK;
}

CKERROR CKBehaviorManager::PreClearAll() {
    RemoveAllObjects();
    return CK_OK;
}

CKERROR CKBehaviorManager::SequenceDeleted(CK_ID *objids, int count) {
    for (auto it = m_BeObjectNextFrame.Begin(); it != m_BeObjectNextFrame.End();) {
        CKBeObject *beo = (CKBeObject *) m_Context->GetObject(it.GetKey());
        if (beo) {
            ++it;
        } else {
            it = m_BeObjectNextFrame.Remove(it);
        }
    }
    return CK_OK;
}

CKERROR CKBehaviorManager::SequenceToBeDeleted(CK_ID *objids, int count) {
    m_BeObjects.Check();
    m_Behaviors.Check();
    return CK_OK;
}

CKBehaviorManager::CKBehaviorManager(CKContext *Context) : CKBaseManager(Context, BEHAVIOR_MANAGER_GUID, "Behavior Manager") {
    m_CurrentBehavior = nullptr;
    m_BehaviorMaxIteration = 8000;
    m_Context->RegisterNewManager(this);
}

CKBehaviorManager::~CKBehaviorManager() {}
