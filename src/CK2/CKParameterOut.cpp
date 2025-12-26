#include "CKParameterOut.h"

#include "CKBehavior.h"
#include "CKFile.h"
#include "CKParameterOperation.h"

CK_CLASSID CKParameterOut::m_ClassID = CKCID_PARAMETEROUT;

CKERROR CKParameterOut::GetValue(void *buf, CKBOOL update) {
    if (update && (m_ObjectFlags & CK_PARAMETEROUT_PARAMOP) != 0) {
        Update();
    }
    return CKParameter::GetValue(buf, FALSE);
}

CKERROR CKParameterOut::SetValue(const void *buf, int size) {
    CKERROR err = CKParameter::SetValue(buf, size);
    if (err == CK_OK) {
        DataChanged();
    }
    return err;
}

CKERROR CKParameterOut::CopyValue(CKParameter *param, CKBOOL UpdateParam) {
    if (!param)
        return CKERR_INVALIDPARAMETER;

    CKERROR err = CKParameter::CopyValue(param, UpdateParam);
    if (err == CK_OK) {
        DataChanged();
    }
    return err;
}

void *CKParameterOut::GetReadDataPtr(CKBOOL update) {
    if (update && (m_ObjectFlags & CK_PARAMETEROUT_PARAMOP) != 0) {
        Update();
    }
    return CKParameter::GetReadDataPtr(TRUE);
}

int CKParameterOut::GetStringValue(CKSTRING Value, CKBOOL update) {
    if (update && (m_ObjectFlags & CK_PARAMETEROUT_PARAMOP) != 0) {
        Update();
    }
    return CKParameter::GetStringValue(Value, update);
}

void CKParameterOut::DataChanged() {
    for (auto it = m_Destinations.Begin(); it != m_Destinations.End(); ++it) {
        CKParameter *param = (CKParameter *) *it;
        if (param) {
            param->CopyValue(this, FALSE);
        }
    }
}

CKERROR CKParameterOut::AddDestination(CKParameter *param, CKBOOL CheckType) {
    if (!param)
        return CKERR_INVALIDPARAMETER;

    if (param->GetObjectFlags() & CK_PARAMETERIN_THIS || CheckType && !IsCompatibleWith(param))
        return CKERR_INCOMPATIBLEPARAMETERS;

    if (CKIsChildClassOf(param, CKCID_PARAMETEROUT)) {
        CKParameterOut *out = (CKParameterOut *) param;
        for (auto it = out->m_Destinations.Begin(); it != out->m_Destinations.End(); ++it) {
            if (*it == this)
                return CKERR_INVALIDPARAMETER;
        }
    }

    if (!m_Destinations.AddIfNotHere(param))
        return CKERR_ALREADYPRESENT;

    return CK_OK;
}

void CKParameterOut::RemoveDestination(CKParameter *param) {
    m_Destinations.Remove(param);
}

int CKParameterOut::GetDestinationCount() {
    return m_Destinations.Size();
}

CKParameter *CKParameterOut::GetDestination(int pos) {
    if (pos >= 0 && pos < m_Destinations.Size()) {
        return (CKParameter *) m_Destinations[pos];
    }
    return nullptr;
}

void CKParameterOut::RemoveAllDestinations() {
    m_Destinations.Clear();
}

CKParameterOut::CKParameterOut(CKContext *Context, CKSTRING name): CKParameter(Context, name) {
}

CKParameterOut::~CKParameterOut() {
}

CK_CLASSID CKParameterOut::GetClassID() {
    return m_ClassID;
}

void CKParameterOut::PreSave(CKFile *file, CKDWORD flags) {
    CKParameter::PreSave(file, flags);
    for (auto it = m_Destinations.Begin(); it != m_Destinations.End(); ++it) {
        file->SaveObjectAsReference(*it);
    }
}

CKStateChunk *CKParameterOut::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKParameter::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_PARAMETEROUT, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    const int destCount = m_Destinations.Size();
    if (destCount > 0) {
        chunk->WriteIdentifier(CK_STATESAVE_PARAMETEROUT_DESTINATIONS);
        chunk->WriteInt(destCount);

        for (int i = 0; i < destCount; ++i) {
            chunk->WriteObject(m_Destinations[i]);
        }
    }

    chunk->CloseChunk();
    return chunk;
}

CKERROR CKParameterOut::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKParameter::Load(chunk, file);

    m_ObjectFlags &= ~(CK_PARAMETEROUT_SETTINGS |
                       CK_PARAMETERIN_DISABLED |
                       CK_PARAMETERIN_THIS |
                       CK_PARAMETERIN_SHARED |
                       CK_PARAMETEROUT_DELETEAFTERUSE);

    if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETEROUT_DESTINATIONS)) {
        m_Destinations.Clear();
        const int destCount = chunk->ReadInt();
        for (int i = 0; i < destCount; ++i) {
            CKParameter *param = (CKParameter *) chunk->ReadObject(m_Context);
            AddDestination(param, FALSE);
        }
    }

    return CK_OK;
}

void CKParameterOut::PreDelete() {
    CKObject::PreDelete();
    CKObject *owner = GetOwner();
    if (!owner)
        return;

    CKBehavior *beh = nullptr;
    if (CKIsChildClassOf(owner, CKCID_BEHAVIOR)) {
        beh = (CKBehavior *) owner;
    } else if (CKIsChildClassOf(owner, CKCID_PARAMETEROPERATION)) {
        beh = ((CKParameterOperation *) owner)->GetOwner();
    }

    if (!beh)
        return;

    CKBehavior *ownerScript = beh->GetOwnerScript();
    if (ownerScript && !ownerScript->IsToBeDeleted()) {
        CKObject *parent = owner;
        while (parent) {
            if (CKIsChildClassOf(parent, CKCID_BEHAVIOR)) {
                beh = (CKBehavior *) parent;
                if (!beh->IsToBeDeleted()) {
                    XSObjectPointerArray &outParams = beh->m_OutParameter;
                    for (auto it = outParams.Begin(); it != outParams.End(); ++it) {
                        if (*it == this) {
                            outParams.Remove(*it);
                            break;
                        }
                    }
                }

                parent = beh->GetParent();
            } else if (CKIsChildClassOf(parent, CKCID_PARAMETEROPERATION)) {
                CKParameterOperation *op = (CKParameterOperation *) parent;

                if (op->GetOutParameter() == this) {
                    op->m_Out = nullptr;
                }
                parent = op->GetOwner();
            } else {
                break; // Unexpected parent type
            }
        }
    }
}

void CKParameterOut::CheckPreDeletion() {
    CKObject::CheckPreDeletion();
    m_Destinations.Check();
}

int CKParameterOut::GetMemoryOccupation() {
    // Base parameter + in-object members + owned destination-pointer buffer.
    return CKParameter::GetMemoryOccupation() +
        static_cast<int>(sizeof(CKParameterOut) - sizeof(CKParameter)) +
        m_Destinations.GetMemoryOccupation(FALSE);
}

int CKParameterOut::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    if (CKIsChildClassOf(cid, CKCID_PARAMETER)) {
        XSObjectPointerArray &destinations = m_Destinations;
        for (auto it = destinations.Begin(); it != destinations.End(); ++it) {
            if (*it == o) {
                return 1;
            }
        }
    }

    return CKParameter::IsObjectUsed(o, cid);
}

CKERROR CKParameterOut::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKParameter::PrepareDependencies(context);
    if (err != CK_OK) {
        return err;
    }
    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKParameterOut::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKParameter::RemapDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    m_Destinations.Remap(context);
    return CK_OK;
}

CKERROR CKParameterOut::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKParameter::Copy(o, context);
    if (err != CK_OK) {
        return err;
    }

    CKParameterOut *pOut = (CKParameterOut *) &o;
    for (auto it = pOut->m_Destinations.Begin(); it != pOut->m_Destinations.End(); ++it) {
        CKParameter *param = (CKParameter *) *it;
        AddDestination(param, TRUE);
    }
    return CK_OK;
}

CKSTRING CKParameterOut::GetClassName() {
    return "Parameter Out";
}

int CKParameterOut::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKParameterOut::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKParameterOut::Register() {
    CKCLASSDEFAULTOPTIONS(CKParameterOut, CK_DEPENDENCIES_COPY);
}

CKParameterOut *CKParameterOut::CreateInstance(CKContext *Context) {
    return new CKParameterOut(Context);
}

void CKParameterOut::Update() {
    if (!IsToBeDeleted()) {
        CKParameterOperation *op = (CKParameterOperation *) GetOwner();
        if (op) {
            op->DoOperation();
        }
    }
}
