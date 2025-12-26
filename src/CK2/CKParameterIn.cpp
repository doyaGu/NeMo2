#include "CKParameterIn.h"

#include "CKParameterManager.h"
#include "CKParameterOperation.h"
#include "CKFile.h"
#include "CKBehavior.h"

CK_CLASSID CKParameterIn::m_ClassID = CKCID_PARAMETERIN;

CKERROR CKParameterIn::SetDirectSource(CKParameter *param) {
    if (param && param->m_ParamType) {
        CKParameterManager *pm = m_Context->GetParameterManager();

        int paramTypeIndex = param->GetType();
        int thisTypeIndex = (m_ParamType) ? m_ParamType->Index : -1;

        if (!pm->IsDerivedFrom(paramTypeIndex, thisTypeIndex) &&
            !pm->IsDerivedFrom(thisTypeIndex, paramTypeIndex)) {
            return CKERR_INCOMPATIBLEPARAMETERS;
        }

        if (m_ParamType->Cid) {
            CKDWORD paramCid = param->m_ParamType->Cid;
            if (!CKIsChildClassOf(m_ParamType->Cid, paramCid) &&
                !CKIsChildClassOf(paramCid, m_ParamType->Cid)) {
                return CKERR_INCOMPATIBLEPARAMETERS;
            }
        }
    }

    m_ObjectFlags &= ~CK_PARAMETERIN_SHARED;
    m_OutSource = param;

    return CK_OK;
}

CKERROR CKParameterIn::ShareSourceWith(CKParameterIn *param) {
    if (param && param->m_ParamType) {
        CKParameterManager *paramManager = m_Context->GetParameterManager();

        int sourceTypeIndex = param->m_ParamType->Index;
        int thisTypeIndex = m_ParamType->Index;

        if (!paramManager->IsDerivedFrom(sourceTypeIndex, thisTypeIndex) &&
            !paramManager->IsDerivedFrom(thisTypeIndex, sourceTypeIndex)) {
            return CKERR_INCOMPATIBLEPARAMETERS;
        }

        if (m_ParamType->Cid) {
            CKDWORD sourceCid = param->m_ParamType->Cid;
            if (!CKIsChildClassOf(m_ParamType->Cid, sourceCid) &&
                !CKIsChildClassOf(sourceCid, m_ParamType->Cid)) {
                return CKERR_INCOMPATIBLEPARAMETERS;
            }
        }
    }

    m_ObjectFlags |= CK_PARAMETERIN_SHARED;

    m_InShared = param;

    return CK_OK;
}

void CKParameterIn::SetType(CKParameterType type, CKBOOL UpdateSource, CKSTRING NewName) {
    CKParameterIn *current = this;
    while (true) {
        CKParameterManager *paramManager = current->m_Context->GetParameterManager();
        current->m_ParamType = paramManager->GetParameterTypeDescription(type);

        // Update the parameter name if a new name is provided
        if (NewName) {
            current->SetName(NewName, FALSE);
        }

        // If UpdateSource is false, return immediately
        if (!UpdateSource) {
            break;
        }

        CKBehavior *ownerBehavior = (CKBehavior *)current->GetOwner();
        if (!CKIsChildClassOf(ownerBehavior, CKCID_BEHAVIOR)) {
            break;
        }

        if (!(current->m_ObjectFlags & CK_PARAMETERIN_SHARED)) {
            CKParameter *outSource = current->m_OutSource;
            if (outSource && CKIsChildClassOf(outSource, CKCID_PARAMETERLOCAL)) {
                CKBehavior *parent = ownerBehavior->GetParent();
                if (outSource->GetOwner() == parent) {
                    outSource->SetType(type);
                    if (NewName) {
                        outSource->SetName(NewName, 0);
                    }
                }
            }
            return;
        } else {
            current = current->m_InShared;
            if (current) {
                CKBehavior *sharedOwner = (CKBehavior *)current->GetOwner();
                if (sharedOwner == ownerBehavior->GetParent()) {
                    continue;
                }
            }
            return;
        }
    }
}

void CKParameterIn::SetGUID(CKGUID guid, CKBOOL UpdateSource, CKSTRING NewName) {
    CKParameterManager *pm = m_Context->GetParameterManager();
    int paramType = pm->ParameterGuidToType(guid);
    SetType(paramType, UpdateSource, NewName);
}

CKParameterIn::CKParameterIn(CKContext *Context, CKSTRING name, int type) : CKObject(Context, name) {
    CKParameterManager *pm = Context->GetParameterManager();
    CKParameterTypeDesc *typeDesc = pm->GetParameterTypeDescription(type);
    m_Owner = nullptr;
    m_OutSource = nullptr;
    m_ParamType = typeDesc;
}

CKParameterIn::~CKParameterIn() {
}

CK_CLASSID CKParameterIn::GetClassID() {
    return m_ClassID;
}

void CKParameterIn::PreSave(CKFile *file, CKDWORD flags) {
    CKObject::PreSave(file, flags);
    file->SaveObjectAsReference(GetRealSource());
}

CKStateChunk *CKParameterIn::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_PARAMETERIN, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if (m_ObjectFlags & CK_PARAMETERIN_SHARED) {
        chunk->WriteIdentifier(CK_STATESAVE_PARAMETERIN_DATASHARED);
    } else {
        chunk->WriteIdentifier(CK_STATESAVE_PARAMETERIN_DATASOURCE);
    }

    chunk->CheckSize(16);

    CKGUID guid = m_ParamType ? m_ParamType->Guid : CKGUID();
    chunk->WriteGuid(guid);

    chunk->WriteObject(m_OutSource);

    if (m_ObjectFlags & CK_PARAMETERIN_DISABLED) {
        chunk->WriteIdentifier(CK_STATESAVE_PARAMETERIN_DISABLED);
    }

    chunk->CloseChunk();
    return chunk;
}

static void ConvertLegacyGuid(CKGUID &guid) {
    // Handle GUID conversions using official definitions
    if (guid == CKPGUID_OLDMESSAGE) {
        guid = CKPGUID_MESSAGE;
    } else if (guid == CKPGUID_OLDATTRIBUTE) {
        guid = CKPGUID_ATTRIBUTE;
    } else if (guid == CKPGUID_OLDTIME) {
        guid = CKPGUID_TIME;
    }
}

CKERROR CKParameterIn::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKObject::Load(chunk, file);

    m_ObjectFlags &= ~CK_OBJECT_PARAMMASK;

    const int dataVersion = chunk->GetDataVersion();
    if (dataVersion >= 1) {
        // Handle modern data versions
        if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_DATASHARED)) {
            CKGUID guid = chunk->ReadGuid();
            ConvertLegacyGuid(guid);

            CKParameterManager *pm = m_Context->GetParameterManager();
            m_ParamType = pm->GetParameterTypeDescription(guid);

            // Handle legacy object IDs for versions < 5
            if (chunk->GetDataVersion() < 5) {
                chunk->ReadObjectID();
            }

            m_InShared = (CKParameterIn *)chunk->ReadObject(m_Context);
            m_ObjectFlags |= CK_PARAMETERIN_SHARED;
        } else if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_DATASOURCE)) {
            CKGUID guid = chunk->ReadGuid();
            ConvertLegacyGuid(guid);

            CKParameterManager *pm = m_Context->GetParameterManager();
            m_ParamType = pm->GetParameterTypeDescription(guid);

            // Handle legacy object IDs for versions < 5
            if (chunk->GetDataVersion() < 5) {
                chunk->ReadObjectID();
            }

            m_InShared = (CKParameterIn *)chunk->ReadObject(m_Context);
        } else if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_DEFAULTDATA)) {
            CKGUID guid = chunk->ReadGuid();
            ConvertLegacyGuid(guid);

            CKParameterManager *pm = m_Context->GetParameterManager();
            m_ParamType = pm->GetParameterTypeDescription(guid);

            m_Owner = chunk->ReadObject(m_Context);
            m_OutSource = (CKParameter *)chunk->ReadObject(m_Context);
            CKParameterIn *param = (CKParameterIn *)chunk->ReadObject(m_Context);
            if (!m_OutSource) {
                m_InShared = param;
            } else {
                m_ObjectFlags |= CK_PARAMETERIN_SHARED;
            }
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_DISABLED)) {
            m_ObjectFlags |= CK_PARAMETERIN_DISABLED;
        }
    } else {
        // Handle legacy version (pre data version 1)

        // First check for DEFAULTDATA (0x400) in legacy path
        if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_DEFAULTDATA)) {
            CKGUID guid = chunk->ReadGuid();
            ConvertLegacyGuid(guid);

            CKParameterManager *pm = m_Context->GetParameterManager();
            m_ParamType = pm->GetParameterTypeDescription(guid);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_OWNER)) {
            m_Owner = chunk->ReadObject(m_Context);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_INSHARED)) {
            CKParameter *param = (CKParameter *)chunk->ReadObject(m_Context);
            if (param) {
                m_ObjectFlags |= CK_PARAMETERIN_SHARED;
                m_OutSource = param;
            }
        }

        if (!(m_ObjectFlags & CK_PARAMETERIN_SHARED) &&
            chunk->SeekIdentifier(CK_STATESAVE_PARAMETERIN_OUTSOURCE)) {
            CKParameter *param = (CKParameter *)chunk->ReadObject(m_Context);
            if (param) {
                m_OutSource = param;
            }
        }
    }

    return CK_OK;
}

void CKParameterIn::PreDelete() {
    if (m_Owner && !m_Owner->IsToBeDeleted()) {
        if (CKIsChildClassOf(m_Owner, CKCID_PARAMETEROPERATION)) {
            CKParameterOperation *pop = (CKParameterOperation *)m_Owner;
            if (pop->m_In1 == this) {
                pop->m_In1 = nullptr;
            }
            if (pop->m_In2 == this)
                pop->m_In2 = nullptr;
        } else {
            CKBehavior *beh = (CKBehavior *)m_Owner;
            while (beh) {
                if (beh->m_InputTargetParam == m_ID) {
                    beh->m_InputTargetParam = 0;
                }

                for (auto *it = beh->m_InParameter.Begin(); it != beh->m_InParameter.End(); ++it) {
                    if (*it == this) {
                        beh->m_InParameter.Remove(it);
                        break;
                    }
                }

                beh = beh->GetParent();
            }
        }
    }
}

void CKParameterIn::CheckPreDeletion() {
    if (m_OutSource && m_OutSource->IsToBeDeleted()) {
        m_OutSource = nullptr;
    }
}

int CKParameterIn::GetMemoryOccupation() {
    return CKObject::GetMemoryOccupation() +
        static_cast<int>(sizeof(CKParameterIn) - sizeof(CKObject));
}

int CKParameterIn::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    if (m_OutSource == o) {
        return TRUE;
    }
    return CKObject::IsObjectUsed(o, cid);
}

CKERROR CKParameterIn::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::PrepareDependencies(context);
    if (err != CK_OK) {
        return err;
    }
    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKParameterIn::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    m_Owner = context.Remap(m_Owner);

    m_OutSource = (CKParameter *)context.Remap(m_OutSource);

    return CK_OK;
}

CKERROR CKParameterIn::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK) {
        return err;
    }

    CKParameterIn *pin = (CKParameterIn *)&o;

    m_ParamType = pin->m_ParamType;
    m_Owner = pin->m_Owner;
    m_OutSource = pin->m_OutSource;

    m_ObjectFlags &= ~CK_OBJECT_PARAMMASK;
    m_ObjectFlags |= pin->m_ObjectFlags & CK_OBJECT_PARAMMASK;

    return CK_OK;
}

CKSTRING CKParameterIn::GetClassName() {
    return "Parameter In";
}

int CKParameterIn::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKParameterIn::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKParameterIn::Register() {
    CKCLASSNOTIFYFROM(CKParameterIn, CKParameterIn);
    CKCLASSNOTIFYFROM(CKParameterIn, CKParameter);
    CKCLASSDEFAULTOPTIONS(CKParameterIn, CK_DEPENDENCIES_COPY);
}

CKParameterIn *CKParameterIn::CreateInstance(CKContext *Context) {
    return new CKParameterIn(Context);
}
