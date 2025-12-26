#include "CKParameterLocal.h"

#include "CKParameterManager.h"
#include "CKBehavior.h"
#include "CKBeObject.h"
#include "CKStateChunk.h"

CK_CLASSID CKParameterLocal::m_ClassID = CKCID_PARAMETERLOCAL;

void CKParameterLocal::SetAsMyselfParameter(CKBOOL act) {
    if (act) {
        CKBehavior *behaviorOwner = (CKBehavior *)m_Owner;
        CKBeObject *targetObject = nullptr;

        // Get the actual object owner (either behavior's owner or direct owner)
        if (behaviorOwner && behaviorOwner->GetClassID() == CKCID_BEHAVIOR) {
            targetObject = behaviorOwner->GetOwner();
        } else {
            targetObject = (CKBeObject *)m_Owner;
        }

        // Determine parameter type from object's class
        CKParameterManager *pm = m_Context->GetParameterManager();
        int paramType = -1;

        if (targetObject) {
            paramType = pm->ClassIDToType(targetObject->GetClassID());
        }

        if (paramType > 0) {
            SetType(paramType);
        } else {
            SetGUID(CKPGUID_BEOBJECT);
        }

        *(CK_ID *)m_Buffer = targetObject ? targetObject->GetID() : 0;

        m_ObjectFlags |= CK_PARAMETERIN_THIS;
    } else {
        m_ObjectFlags &= ~CK_PARAMETERIN_THIS;
    }
}

void CKParameterLocal::SetOwner(CKObject *o) {
    CKParameter::SetOwner(o);
    if (m_ObjectFlags & CK_PARAMETERIN_THIS) {
        SetAsMyselfParameter(TRUE);
    }
}

CKERROR CKParameterLocal::SetValue(const void *buf, int size) {
    if (m_ObjectFlags & CK_PARAMETERIN_THIS) {
        return CKERR_INVALIDPARAMETER;
    }
    return CKParameter::SetValue(buf, size);
}

CKERROR CKParameterLocal::CopyValue(CKParameter *param, CKBOOL UpdateParam) {
    if (m_ObjectFlags & CK_PARAMETERIN_THIS) {
        return CKERR_INVALIDPARAMETER;
    }
    return CKParameter::CopyValue(param, UpdateParam);
}

void *CKParameterLocal::GetWriteDataPtr() {
    if (m_ObjectFlags & CK_PARAMETERIN_THIS) {
        return nullptr;
    }
    return CKParameter::GetWriteDataPtr();
}

CKERROR CKParameterLocal::SetStringValue(CKSTRING Value) {
    if (m_ObjectFlags & CK_PARAMETERIN_THIS) {
        return CKERR_INVALIDPARAMETER;
    }
    return CKParameter::SetStringValue(Value);
}

CKParameterLocal::CKParameterLocal(CKContext *Context, CKSTRING name, int type) : CKParameter(Context, name, type) {}

CKParameterLocal::~CKParameterLocal() {}

CK_CLASSID CKParameterLocal::GetClassID() {
    return m_ClassID;
}

void CKParameterLocal::PreDelete() {
    CKObject::PreDelete();
    CKObject *owner = GetOwner();
    if (owner && CKIsChildClassOf(owner, CKCID_BEHAVIOR) && !owner->IsToBeDeleted()) {
        CKBehavior *beh = (CKBehavior *)owner;
        XSObjectPointerArray &localParams = beh->m_LocalParameter;
        for (auto it = localParams.Begin(); it != localParams.End(); ++it) {
            if (*it == this) {
                localParams.Remove(it);
                return;
            }
        }
    }
}

CKStateChunk *CKParameterLocal::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = (m_ObjectFlags & CK_PARAMETERIN_THIS)
                                  ? CKObject::Save(file, flags)
                                  : CKParameter::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_PARAMETERLOCAL, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if (m_ObjectFlags & CK_PARAMETERIN_THIS) {
        chunk->WriteIdentifier(CK_STATESAVE_PARAMETEROUT_MYSELF);
    }

    if (m_ObjectFlags & CK_PARAMETEROUT_SETTINGS) {
        chunk->WriteIdentifier(CK_STATESAVE_PARAMETEROUT_ISSETTING);
    }

    chunk->CloseChunk();
    return chunk;
}

CKERROR CKParameterLocal::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKParameter::Load(chunk, file);

    if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETEROUT_MYSELF)) {
        SetAsMyselfParameter(TRUE);
    }

    if (chunk->SeekIdentifier(CK_STATESAVE_PARAMETEROUT_ISSETTING)) {
        m_ObjectFlags |= CK_PARAMETEROUT_SETTINGS;
    }

    return CK_OK;
}

int CKParameterLocal::GetMemoryOccupation() {
    return CKParameter::GetMemoryOccupation();
}

CKERROR CKParameterLocal::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKParameter::PrepareDependencies(context);
    if (err != CK_OK) {
        return err;
    }
    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKParameterLocal::RemapDependencies(CKDependenciesContext &context) {
    return CKParameter::RemapDependencies(context);
}

CKERROR CKParameterLocal::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKParameter::Copy(o, context);
    if (err != CK_OK) {
        return err;
    }

    CKParameterLocal *param = (CKParameterLocal *)&o;
    if (param->m_ObjectFlags & CK_PARAMETERIN_THIS) {
        m_ObjectFlags |= CK_PARAMETERIN_THIS;
    }
    return CK_OK;
}

CKSTRING CKParameterLocal::GetClassName() {
    return "Parameter Local";
}

int CKParameterLocal::GetDependenciesCount(int mode) {
    return CKParameter::GetDependenciesCount(mode);
}

CKSTRING CKParameterLocal::GetDependencies(int i, int mode) {
    return CKParameter::GetDependencies(i, mode);
}

void CKParameterLocal::Register() {
    CKCLASSDEFAULTOPTIONS(CKParameterLocal, CK_DEPENDENCIES_COPY);
}

CKParameterLocal *CKParameterLocal::CreateInstance(CKContext *Context) {
    return new CKParameterLocal(Context);
}
