#include "CKSceneObjectDesc.h"

#include "CKStateChunk.h"

CKERROR CKSceneObjectDesc::ReadState(CKStateChunk *chunk) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    chunk->StartRead();

    // Read object information
    if (chunk->SeekIdentifier(CK_STATESAVE_SCENEOBJECTDESCALL)) {
        m_Object = chunk->ReadObjectID();

        // Clean existing state
        delete m_InitialValue;

        // Read initial value chunk
        m_InitialValue = chunk->ReadSubChunk();

        // Read and discard deprecated chunk
        CKStateChunk* deprecated = chunk->ReadSubChunk();
        delete deprecated;

        // Read flags
        if (chunk->SeekIdentifier(2)) {
            m_Flags = (unsigned short) chunk->ReadInt();
            chunk->ReadInt();
        } else {
            m_Flags = chunk->ReadInt();
        }

        // Convert legacy flags
        m_Flags &= CK_SCENEOBJECT_ACTIVE;

        if (m_Flags & 2) {
            m_Flags |= CK_SCENEOBJECT_START_RESET;
        }

        if (m_Flags & 1) {
            m_Flags |= CK_SCENEOBJECT_START_ACTIVATE;
        } else {
            m_Flags |= CK_SCENEOBJECT_START_DEACTIVATE;
        }
    }

    return CK_OK;
}

void CKSceneObjectDesc::Clear() {
    if (m_InitialValue) {
        delete m_InitialValue;
    }
    m_InitialValue = nullptr;
    m_Object = 0;
    m_Flags = CK_SCENEOBJECT_START_ACTIVATE | CK_SCENEOBJECT_START_RESET;
}

void CKSceneObjectDesc::Init(CKObject *obj) {
    m_Object = obj ? obj->GetID() : 0;
    m_InitialValue = nullptr;
    m_Flags = CK_SCENEOBJECT_START_ACTIVATE | CK_SCENEOBJECT_START_RESET;
}
