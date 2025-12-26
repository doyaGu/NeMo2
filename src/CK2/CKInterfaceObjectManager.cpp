#include "CKInterfaceObjectManager.h"

#include "CKStateChunk.h"

CK_CLASSID CKInterfaceObjectManager::m_ClassID = CKCID_INTERFACEOBJECTMANAGER;

void CKInterfaceObjectManager::SetGuid(CKGUID guid) {
    m_Guid = guid;
}

CKGUID CKInterfaceObjectManager::GetGuid() {
    return m_Guid;
}

void CKInterfaceObjectManager::AddStateChunk(CKStateChunk *chunk) {
    CKStateChunk **chunks = new CKStateChunk *[m_Count + 1];
    if (m_Chunks)
        memcpy(chunks, m_Chunks, sizeof(CKStateChunk *) * m_Count);
    if (m_Chunks)
        delete[] m_Chunks;

    m_Chunks = chunks;
    m_Chunks[m_Count] = chunk;
    ++m_Count;
}

void CKInterfaceObjectManager::RemoveStateChunk(CKStateChunk *chunk) {
    if (m_Count <= 0)
        return;

    int i;
    for (i = 0; i < m_Count; ++i) {
        if (m_Chunks[i] == chunk)
            break;
    }

    if (i != m_Count) {
        memmove(&m_Chunks[i], &m_Chunks[i + 1], sizeof(CKStateChunk *) * (m_Count - (i + 1)));
        if (chunk)
            delete chunk;

        --m_Count;
    }
}

int CKInterfaceObjectManager::GetChunkCount() {
    return m_Count;
}

CKStateChunk *CKInterfaceObjectManager::GetChunk(int pos) {
    if (pos < 0 || pos >= m_Count || !m_Chunks)
        return nullptr;

    return m_Chunks[pos];
}

CKInterfaceObjectManager::CKInterfaceObjectManager(CKContext *Context, CKSTRING name) : CKObject(Context, name) {
    m_Count = 0;
    m_Chunks = nullptr;
}

CKInterfaceObjectManager::~CKInterfaceObjectManager() {
    for (int i = 0; i < m_Count; ++i) {
        CKStateChunk *chunk = m_Chunks[i];
        if (chunk)
            delete chunk;
    }

    delete[] m_Chunks;
    m_Chunks = nullptr;
}

CK_CLASSID CKInterfaceObjectManager::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKInterfaceObjectManager::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_INTERFACEOBJECTMANAGER, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(0x1234567);
    chunk->WriteInt(m_Count);
    for (int i = 0; i < m_Count; ++i) {
        chunk->WriteSubChunk(m_Chunks[i]);
    }

    chunk->WriteIdentifier(0x87654321);
    chunk->WriteGuid(m_Guid);

    if (GetClassID() == CKCID_INTERFACEOBJECTMANAGER)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();

    return chunk;
}

CKERROR CKInterfaceObjectManager::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk || !m_Context->IsInInterfaceMode())
        return CKERR_INVALIDPARAMETER;

    if (chunk->SeekIdentifier(0x1234567)) {
        m_Count = chunk->ReadInt();
        m_Chunks = new CKStateChunk *[m_Count];
        for (int i = 0; i < m_Count; ++i) {
            m_Chunks[i] = chunk->ReadSubChunk();
        }
    }

    if (chunk->SeekIdentifier(0x87654321)) {
        m_Guid = chunk->ReadGuid();
    } else {
        m_Guid = CKGUID();
    }

    return CK_OK;
}

CKSTRING CKInterfaceObjectManager::GetClassName() {
    return "VirtoolsInterfaceObject";
}

int CKInterfaceObjectManager::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING CKInterfaceObjectManager::GetDependencies(int i, int mode) {
    return nullptr;
}

void CKInterfaceObjectManager::Register() {
    CKCLASSDEFAULTOPTIONS(CKInterfaceObjectManager, CK_DEPENDENCIES_COPY);
}

CKInterfaceObjectManager *CKInterfaceObjectManager::CreateInstance(CKContext *Context) {
    return new CKInterfaceObjectManager(Context);
}
