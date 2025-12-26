#include "CKSynchroObject.h"

#include "CKBeObject.h"
#include "CKStateChunk.h"

CK_CLASSID CKSynchroObject::m_ClassID = CKCID_SYNCHRO;

void CKSynchroObject::Reset() {
    m_Passed.Clear();
    m_Arrived.Clear();
}

void CKSynchroObject::SetRendezVousNumberOfWaiters(int waiters) {
    m_MaxWaiters = waiters;
}

int CKSynchroObject::GetRendezVousNumberOfWaiters() {
    return m_MaxWaiters;
}

CKBOOL CKSynchroObject::CanIPassRendezVous(CKBeObject *asker) {
    if (!asker)
        return FALSE;

    int count = m_Arrived.GetCount();
    if (count == m_MaxWaiters) {
        if (!m_Arrived.PtrSeek(asker))
            return FALSE;
    } else {
        if (m_Arrived.PtrSeek(asker))
            return FALSE;
        m_Arrived.InsertRear(asker);
        if (m_Arrived.GetCount() != m_MaxWaiters) {
            return FALSE;
        }
    }
    m_Passed.AddIfNotHere(asker);
    if (m_Passed.GetCount() == m_MaxWaiters) {
        Reset();
    }
    return TRUE;
}

int CKSynchroObject::GetRendezVousNumberOfArrivedObjects() {
    return m_Arrived.GetCount();
}

CKBeObject *CKSynchroObject::GetRendezVousArrivedObject(int pos) {
    return (CKBeObject *) m_Context->GetObject(m_Arrived.Seek(pos));
}

CKSynchroObject::CKSynchroObject(CKContext *Context, CKSTRING name) : CKObject(Context, name) {
    m_MaxWaiters = 0;
}

CKSynchroObject::~CKSynchroObject() {
    Reset();
}

CK_CLASSID CKSynchroObject::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKSynchroObject::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_SYNCHRO, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(CK_STATESAVE_SYNCHRODATA);
    chunk->WriteInt(m_MaxWaiters);

    chunk->WriteObjectArray(&m_Arrived, m_Context);
    chunk->WriteObjectArray(&m_Passed, m_Context);

    chunk->CloseChunk();
    return chunk;
}

CKERROR CKSynchroObject::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Load base CKObject data
    CKObject::Load(chunk, file);

    if (chunk->SeekIdentifier(CK_STATESAVE_SYNCHRODATA)) {
        m_MaxWaiters = chunk->ReadInt();

        chunk->ReadObjectArray(&m_Arrived);
        m_Arrived.Check(m_Context);

        chunk->ReadObjectArray(&m_Passed);
        m_Passed.Check(m_Context);
    }

    return CK_OK;
}

void CKSynchroObject::CheckPostDeletion() {
    m_Arrived.Check(m_Context);
    m_Passed.Check(m_Context);
}

int CKSynchroObject::GetMemoryOccupation() {
    // Base object + in-object members + owned linked-list nodes.
    const int nodeBytes = static_cast<int>(sizeof(CKObjectArray::Node));
    const int nodes = m_Arrived.GetCount() + m_Passed.GetCount();

    return CKObject::GetMemoryOccupation() +
        static_cast<int>(sizeof(CKSynchroObject) - sizeof(CKObject)) +
        nodeBytes * nodes;
}

CKBOOL CKSynchroObject::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (CKIsChildClassOf(obj, CKCID_BEOBJECT)) {
        if (m_Arrived.PtrFind(obj) || m_Passed.PtrFind(obj))
            return TRUE;
    }
    return CKObject::IsObjectUsed(obj, cid);
}

CKSTRING CKSynchroObject::GetClassName() {
    return "Synchronization Object";
}

int CKSynchroObject::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKSynchroObject::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKSynchroObject::Register() {
    CKCLASSNOTIFYFROM(CKSynchroObject, CKBeObject);
    CKCLASSDEFAULTOPTIONS(CKSynchroObject, CK_DEPENDENCIES_SAVE);
    CKPARAMETERFROMCLASS(CKSynchroObject, CKPGUID_SYNCHRO);
}

CKSynchroObject *CKSynchroObject::CreateInstance(CKContext *Context) {
    return new CKSynchroObject(Context);
}

CK_CLASSID CKStateObject::m_ClassID = CKCID_STATE;

void CKStateObject::Reset() {
    LeaveState();
}

CKBOOL CKStateObject::IsStateActive() {
    return m_Event;
}

void CKStateObject::EnterState() {
    m_Event = TRUE;
}

void CKStateObject::LeaveState() {
    m_Event = FALSE;
}

CKStateObject::CKStateObject(CKContext *Context, CKSTRING name) : CKObject(Context, name) {
    m_Event = FALSE;
}

CKStateObject::~CKStateObject() {}

CK_CLASSID CKStateObject::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKStateObject::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_STATE, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(0x10);
    chunk->WriteInt(m_Event);

    chunk->CloseChunk();
    return chunk;
}

CKERROR CKStateObject::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;
    CKObject::Load(chunk, file);
    if (chunk->SeekIdentifier(0x10)) {
        m_Event = chunk->ReadInt();
    }
    return CK_OK;
}

int CKStateObject::GetMemoryOccupation() {
    return CKObject::GetMemoryOccupation() +
        static_cast<int>(sizeof(CKStateObject) - sizeof(CKObject));
}

CKSTRING CKStateObject::GetClassName() {
    return "State";
}

int CKStateObject::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKStateObject::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKStateObject::Register() {
    CKCLASSDEFAULTOPTIONS(CKStateObject, CK_DEPENDENCIES_SAVE);
    CKPARAMETERFROMCLASS(CKStateObject, CKPGUID_STATE);
}

CKStateObject *CKStateObject::CreateInstance(CKContext *Context) {
    return new CKStateObject(Context);
}

CK_CLASSID CKCriticalSectionObject::m_ClassID = CKCID_CRITICALSECTION;

void CKCriticalSectionObject::Reset() {
    m_ObjectInSection = 0;
}

CKBOOL CKCriticalSectionObject::EnterCriticalSection(CKBeObject *asker) {
    if (!asker || m_ObjectInSection != 0)
        return FALSE;
    m_ObjectInSection = asker->GetID();
    return TRUE;
}

CKBOOL CKCriticalSectionObject::LeaveCriticalSection(CKBeObject *asker) {
    if (!asker || m_ObjectInSection != asker->GetID())
        return FALSE;
    m_ObjectInSection = 0;
    return TRUE;
}

CKCriticalSectionObject::CKCriticalSectionObject(CKContext *Context, CKSTRING name) : CKObject(Context, name) {
    m_ObjectInSection = 0;
}

CKCriticalSectionObject::~CKCriticalSectionObject() {}

CK_CLASSID CKCriticalSectionObject::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKCriticalSectionObject::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_CRITICALSECTION, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(0x10);
    chunk->WriteObject(m_Context->GetObject(m_ObjectInSection));

    chunk->CloseChunk();
    return chunk;
}

CKERROR CKCriticalSectionObject::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;
    CKObject::Load(chunk, file);
    if (chunk->SeekIdentifier(0x10)) {
        m_ObjectInSection = chunk->ReadObjectID();
    }
    return CK_OK;
}

void CKCriticalSectionObject::CheckPostDeletion() {
    CKObject *obj = m_Context->GetObject(m_ObjectInSection);
    if (!obj)
        m_ObjectInSection = 0;
}

int CKCriticalSectionObject::GetMemoryOccupation() {
    return CKObject::GetMemoryOccupation() +
        static_cast<int>(sizeof(CKCriticalSectionObject) - sizeof(CKObject));
}

CKBOOL CKCriticalSectionObject::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (obj && m_ObjectInSection == obj->GetID())
        return TRUE;
    return CKObject::IsObjectUsed(obj, cid);
}

CKSTRING CKCriticalSectionObject::GetClassName() {
    return "Critical Section";
}

int CKCriticalSectionObject::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKCriticalSectionObject::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKCriticalSectionObject::Register() {
    CKCLASSNOTIFYFROM(CKCriticalSectionObject, CKBeObject);
    CKCLASSDEFAULTOPTIONS(CKCriticalSectionObject, CK_DEPENDENCIES_SAVE);
    CKPARAMETERFROMCLASS(CKCriticalSectionObject, CKPGUID_CRITICALSECTION);
}

CKCriticalSectionObject *CKCriticalSectionObject::CreateInstance(CKContext *Context) {
    return new CKCriticalSectionObject(Context);
}
