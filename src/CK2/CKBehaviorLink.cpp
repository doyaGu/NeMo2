#include "CKBehaviorLink.h"

#include "CKBehavior.h"
#include "CKBehaviorIO.h"
#include "CKStateChunk.h"

CK_CLASSID CKBehaviorLink::m_ClassID = CKCID_BEHAVIORLINK;

CKERROR CKBehaviorLink::SetInBehaviorIO(CKBehaviorIO *ckbioin) {
    if (!ckbioin)
        return CKERR_INVALIDPARAMETER;

    if (m_InIO) {
        m_InIO->m_Links.RemoveObject(this);
    }
    m_InIO = ckbioin;
    m_InIO->m_Links.AddIfNotHere(this);
    return CK_OK;
}

CKERROR CKBehaviorLink::SetOutBehaviorIO(CKBehaviorIO *ckbioout) {
    if (!ckbioout)
        return CKERR_INVALIDPARAMETER;

    m_OutIO = ckbioout;
    return CK_OK;
}

CKBehaviorLink::CKBehaviorLink(CKContext *Context, CKSTRING name) : CKObject(Context, name) {
    m_OldFlags = 0;
    m_InIO = nullptr;
    m_OutIO = nullptr;
    m_ActivationDelay = 1;
    m_InitialActivationDelay = 1;
}

CKBehaviorLink::~CKBehaviorLink() {}

CK_CLASSID CKBehaviorLink::GetClassID() {
    return m_ClassID;
}

CKStateChunk *CKBehaviorLink::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);
    if (file || (flags & CK_STATESAVE_BEHAV_LINKONLY)) {
        CKStateChunk *chunk = CreateCKStateChunk(CKCID_BEHAVIORLINK, file);
        chunk->StartWrite();
        chunk->AddChunkAndDelete(baseChunk);
        chunk->WriteIdentifier(CK_STATESAVE_BEHAV_LINK_NEWDATA);
        chunk->WriteDword((m_ActivationDelay & 0xFFFF) | (m_InitialActivationDelay << 16));
        chunk->WriteObject(m_InIO);
        chunk->WriteObject(m_OutIO);

        if (GetClassID() == CKCID_BEHAVIORLINK) {
            chunk->CloseChunk();
        } else {
            chunk->UpdateDataSize();
        }

        return chunk;
    }
    return baseChunk;
}

CKERROR CKBehaviorLink::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKObject::Load(chunk, file);

    if (m_InIO) {
        m_InIO->m_Links.RemoveObject(this);
    }

    if (chunk->SeekIdentifier(CK_STATESAVE_BEHAV_LINK_NEWDATA)) {
        // New format loading
        CKDWORD delays = chunk->ReadDword();
        m_ActivationDelay = (short) (delays & 0xFFFF);
        m_InitialActivationDelay = (short) ((delays >> 16) & 0xFFFF);
        m_InIO = (CKBehaviorIO *)chunk->ReadObject(m_Context);
        m_OutIO = (CKBehaviorIO *)chunk->ReadObject(m_Context);
    } else {
        // Legacy format support
        if (chunk->SeekIdentifier(CK_STATESAVE_BEHAV_LINK_CURDELAY)) {
            m_ActivationDelay = (short) chunk->ReadInt();
        }
        if (chunk->SeekIdentifier(CK_STATESAVE_BEHAV_LINK_IOS)) {
            m_InIO = (CKBehaviorIO *)chunk->ReadObject(m_Context);
            m_OutIO = (CKBehaviorIO *)chunk->ReadObject(m_Context);
        }
        if (chunk->SeekIdentifier(CK_STATESAVE_BEHAV_LINK_DELAY)) {
            m_InitialActivationDelay = (short) chunk->ReadInt();
        }
    }

    return CK_OK;
}

void CKBehaviorLink::PostLoad() {
    if (m_InIO)
        m_InIO->m_Links.AddIfNotHere(this);

    CKObject::PostLoad();
}

void CKBehaviorLink::PreDelete() {
    CKObject::PreDelete();

    if (m_InIO) {
        if (!m_InIO->IsToBeDeleted()) {
            m_InIO->m_Links.RemoveObject(this);
        }
        CKBehavior *owner = m_InIO->GetOwner();
        if (owner && !owner->IsToBeDeleted()) {
            if (owner->m_GraphData) {
                owner->m_GraphData->m_SubBehaviorLinks.Remove(this);
            }
        }
    }
}

int CKBehaviorLink::GetMemoryOccupation() {
    return CKObject::GetMemoryOccupation() +
        static_cast<int>(sizeof(CKBehaviorLink) - sizeof(CKObject));
}

CKBOOL CKBehaviorLink::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (obj == m_InIO || obj == m_OutIO)
        return TRUE;
    return CKObject::IsObjectUsed(obj, cid);
}

CKERROR CKBehaviorLink::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    m_InIO = (CKBehaviorIO *)context.Remap(m_InIO);
    m_OutIO = (CKBehaviorIO *)context.Remap(m_OutIO);
    if (m_InIO)
        m_InIO->m_Links.AddIfNotHere(this);

    return CK_OK;
}

CKERROR CKBehaviorLink::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    CKBehaviorLink *link = (CKBehaviorLink *)&o;
    m_OldFlags = link->m_OldFlags & ~CKBL_OLD_IN_DELAYED_LIST;
    m_ActivationDelay = link->m_ActivationDelay;
    m_InitialActivationDelay = link->m_InitialActivationDelay;
    m_InIO = link->m_InIO;
    m_OutIO = link->m_OutIO;
    return CK_OK;
}

CKSTRING CKBehaviorLink::GetClassName() {
    return "Behavior Link";
}

int CKBehaviorLink::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKBehaviorLink::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKBehaviorLink::Register() {
    CKCLASSDEFAULTOPTIONS(CKBehaviorLink, CK_DEPENDENCIES_COPY);
}

CKBehaviorLink *CKBehaviorLink::CreateInstance(CKContext *Context) {
    return new CKBehaviorLink(Context);
}
