#include "CKParameterOperation.h"

#include "CKBehavior.h"
#include "CKFile.h"
#include "CKParameterIn.h"
#include "CKParameterOut.h"
#include "CKParameterManager.h"

CK_ID CKParameterOperation::m_ClassID = CKCID_PARAMETEROPERATION;

CKSTRING CKParameterOperation::m_In1Name = "Pin 0";
CKSTRING CKParameterOperation::m_In2Name = "Pin 1";
CKSTRING CKParameterOperation::m_OutName = "Pout 0";

CKERROR CKParameterOperation::DoOperation() {
    if (!m_OperationFunction) {
        if (m_In1 && m_Out) {
            CKParameter *src = m_In1->GetRealSource();
            m_Out->CopyValue(src, TRUE);
        }
        return CKERR_NOTINITIALIZED;
    }

    VxTimeProfiler profiler;

    if (m_HasOperationFunction) {
        m_OperationFunction(m_Context, m_Out, m_In2, m_In1);
    } else {
        m_OperationFunction(m_Context, m_Out, m_In1, m_In2);
    }

    if (m_Out)
        m_Out->DataChanged();

    if (m_Context && m_Context->m_ProfilingEnabled) {
        m_Context->m_Stats.ParametricOperations += profiler.Current();
    }

    return CK_OK;
}

CKGUID CKParameterOperation::GetOperationGuid() {
    return m_OperationGuid;
}

void CKParameterOperation::Reconstruct(CKSTRING Name, CKGUID opguid, CKGUID ResGuid, CKGUID p1Guid, CKGUID p2Guid) {
    m_OperationGuid = opguid;
    m_OperationFunction = nullptr;
    m_HasOperationFunction = FALSE;

    if (m_In1) {
        m_In1->SetGUID(p1Guid, FALSE, FALSE);
    } else {
        m_In1 = m_Context->CreateCKParameterIn(nullptr, p1Guid, IsDynamic());
    }

    if (m_In2) {
        m_In2->SetGUID(p2Guid, FALSE, FALSE);
    } else {
        m_In2 = m_Context->CreateCKParameterIn(nullptr, p2Guid, IsDynamic());
    }

    if (m_Out) {
        m_Out->SetGUID(ResGuid);
    } else {
        m_Out = m_Context->CreateCKParameterOut(nullptr, ResGuid, IsDynamic());
    }

    Update();
}

void CKParameterOperation::Update() {
    CKParameterManager *pm = m_Context->GetParameterManager();
    if (!pm) return;

    SetName(pm->OperationGuidToName(m_OperationGuid), TRUE);

    m_HasOperationFunction = FALSE;

    if (m_In1) {
        m_In1->SetOwner(this);
        if (!m_In1->m_Name || (m_In1->m_Name != m_In1Name && strcmp(m_In1->m_Name, m_In1Name) != 0)) {
            m_In1->SetName(m_In1Name, TRUE);
        }
    }

    if (m_In2) {
        m_In2->SetOwner(this);
        if (!m_In2->m_Name || (m_In2->m_Name != m_In2Name && strcmp(m_In2->m_Name, m_In2Name) != 0)) {
            m_In2->SetName(m_In2Name, TRUE);
        }
    }

    if (m_Out) {
        m_Out->SetOwner(this);
        m_Out->m_ObjectFlags |= CK_PARAMETEROUT_PARAMOP;
        if (!m_Out->m_Name || (m_Out->m_Name != m_OutName && strcmp(m_Out->m_Name, m_OutName) != 0)) {
            m_Out->SetName(m_OutName, TRUE);
        }
    }

    CKGUID in1Guid = m_In1 ? m_In1->GetGUID() : CKPGUID_NONE;
    CKGUID in2Guid = m_In2 ? m_In2->GetGUID() : CKPGUID_NONE;
    CKGUID outGuid = m_Out ? m_Out->GetGUID() : CKPGUID_NONE;
    m_OperationFunction = pm->GetOperationFunction(m_OperationGuid, outGuid, in1Guid, in2Guid);
    if (!m_OperationFunction) {
        m_OperationFunction = pm->GetOperationFunction(m_OperationGuid, outGuid, in2Guid, in1Guid);
        if (m_OperationFunction) {
            m_HasOperationFunction = TRUE;
        }
    }
}

CK_PARAMETEROPERATION CKParameterOperation::GetOperationFunction() {
    return m_OperationFunction;
}

CKParameterOperation::CKParameterOperation(CKContext *Context, CKSTRING name) : CKObject(Context, name) {
    m_In1 = nullptr;
    m_In2 = nullptr;
    m_Out = nullptr;
    m_Owner = nullptr;
    m_OperationGuid = CKGUID();
    m_OperationFunction = nullptr;
    m_HasOperationFunction = FALSE;
}

CKParameterOperation::CKParameterOperation(CKContext *Context, CKSTRING name, CKGUID OpGuid, CKGUID ResGuid, CKGUID P1Guid, CKGUID P2Guid) {}

CKParameterOperation::~CKParameterOperation() {}

CK_CLASSID CKParameterOperation::GetClassID() {
    return m_ClassID;
}

void CKParameterOperation::PreSave(CKFile *file, CKDWORD flags) {
    CKObject::PreSave(file, flags);
    file->SaveObject(m_Out, flags);
    file->SaveObject(m_In1, flags);
    file->SaveObject(m_In2, flags);
}

CKStateChunk *CKParameterOperation::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_PARAMETEROPERATION, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if (file) {
        // Save operation data for file context
        chunk->WriteIdentifier(CK_STATESAVE_OPERATIONNEWDATA);
        chunk->WriteGuid(m_OperationGuid);

        // Write parameter object IDs
        chunk->StartObjectIDSequence(3);
        chunk->WriteObjectSequence(m_In1);
        chunk->WriteObjectSequence(m_In2);
        chunk->WriteObjectSequence(m_Out);
    } else {
        // Save based on individual flags
        if (flags & CK_STATESAVE_OPERATIONOP) {
            chunk->WriteIdentifier(CK_STATESAVE_OPERATIONOP);
            chunk->WriteGuid(m_OperationGuid);
        }

        if (flags & CK_STATESAVE_OPERATIONDEFAULTDATA) {
            chunk->WriteIdentifier(CK_STATESAVE_OPERATIONDEFAULTDATA);
            chunk->WriteObject(m_Owner);
        }

        if (flags & CK_STATESAVE_OPERATIONINPUTS) {
            chunk->WriteIdentifier(CK_STATESAVE_OPERATIONINPUTS);

            // Save input parameters
            CKStateChunk *in1Chunk = m_In1 ? m_In1->Save(nullptr, 0xFFFF) : nullptr;
            CKStateChunk *in2Chunk = m_In2 ? m_In2->Save(nullptr, 0xFFFF) : nullptr;

            chunk->WriteObject(m_In1);
            chunk->WriteSubChunk(in1Chunk);
            chunk->WriteObject(m_In2);
            chunk->WriteSubChunk(in2Chunk);

            if (in1Chunk)
                delete in1Chunk;
            if (in2Chunk)
                delete in2Chunk;
        }

        if (flags & CK_STATESAVE_OPERATIONOUTPUT) {
            chunk->WriteIdentifier(CK_STATESAVE_OPERATIONOUTPUT);

            // Save output parameter
            CKStateChunk *outChunk = m_Out ? m_Out->Save(nullptr, 0xFFFF) : nullptr;

            chunk->WriteObject(m_Out);
            chunk->WriteSubChunk(outChunk);

            if (outChunk)
                delete outChunk;
        }
    }

    chunk->CloseChunk();
    return chunk;
}

CKERROR CKParameterOperation::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    CKObject::Load(chunk, file);

    if (file) {
        if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONNEWDATA)) { // 0x400
            m_OperationGuid = chunk->ReadGuid();

            // Read parameter sequence
            chunk->StartReadSequence();
            if (chunk->GetDataVersion() < 5) {
                // Skip legacy data for versions <5
                chunk->ReadObject(m_Context);
            }

            m_In1 = (CKParameterIn *)chunk->ReadObject(m_Context);
            m_In2 = (CKParameterIn *)chunk->ReadObject(m_Context);
            m_Out = (CKParameterOut *)chunk->ReadObject(m_Context);
        } else {
            if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONOP)) { // 0x100
                m_OperationGuid = chunk->ReadGuid();
            }

            if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONDEFAULTDATA)) { // 0x200
                m_Owner = (CKBehavior *)chunk->ReadObject(m_Context);
            }

            if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONOUTPUT)) { // 0x80
                m_Out = (CKParameterOut *)chunk->ReadObject(m_Context);
            }

            if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONINPUTS)) { // 0x40
                m_In1 = (CKParameterIn *)chunk->ReadObject(m_Context);
                m_In2 = (CKParameterIn *)chunk->ReadObject(m_Context);
            }
        }
    } else {
        if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONOP)) { // 0x100
            m_OperationGuid = chunk->ReadGuid();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONDEFAULTDATA)) { // 0x200
            m_Owner = (CKBehavior *)chunk->ReadObject(m_Context);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONOUTPUT)) { // 0x80
            m_Out = (CKParameterOut *)chunk->ReadObject(m_Context);
            CKStateChunk *outChunk = chunk->ReadSubChunk();
            if (m_Out && outChunk) {
                m_Out->Load(outChunk, nullptr);
            }
            if (outChunk) {
                delete outChunk;
            }
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_OPERATIONINPUTS)) { // 0x40
            // Load first input
            m_In1 = (CKParameterIn *)chunk->ReadObject(m_Context);
            CKStateChunk *in1Chunk = chunk->ReadSubChunk();
            if (m_In1 && in1Chunk) {
                m_In1->Load(in1Chunk, nullptr);
            }
            if (in1Chunk) {
                delete in1Chunk;
            }

            // Load second input
            m_In2 = (CKParameterIn *)chunk->ReadObject(m_Context);
            CKStateChunk *in2Chunk = chunk->ReadSubChunk();
            if (m_In2 && in2Chunk) {
                m_In2->Load(in2Chunk, nullptr);
            }
            if (in2Chunk) {
                delete in2Chunk;
            }
        }

        Update();
    }

    return CK_OK;
}

void CKParameterOperation::PostLoad() {
    CKObject::PostLoad();
}

void CKParameterOperation::PreDelete() {
    CKObject::PreDelete();
    if (m_Owner && !m_Owner->IsToBeDeleted()) {
        if (m_Owner->m_GraphData) {
            m_Owner->m_GraphData->m_Operations.Remove(this);
        }
    }
}

int CKParameterOperation::GetMemoryOccupation() {
    return CKObject::GetMemoryOccupation() +
        static_cast<int>(sizeof(CKParameterOperation) - sizeof(CKObject));
}

int CKParameterOperation::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    if (m_In1 == o || m_In2 == o || m_Out == o) {
        return TRUE;
    }
    return CKObject::IsObjectUsed(o, cid);
}

CKERROR CKParameterOperation::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    if (m_In1) {
        m_In1->PrepareDependencies(context);
    }
    if (m_In2) {
        m_In2->PrepareDependencies(context);
    }
    if (m_Out) {
        m_Out->PrepareDependencies(context);
    }

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKParameterOperation::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    m_In1 = (CKParameterIn *)context.Remap(m_In1);
    m_In2 = (CKParameterIn *)context.Remap(m_In2);
    m_Out = (CKParameterOut *)context.Remap(m_Out);
    m_Owner = (CKBehavior *)context.Remap(m_Owner);
    Update();
    return CK_OK;
}

CKERROR CKParameterOperation::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    CKParameterOperation *op = (CKParameterOperation *)&o;
    m_In1 = op->m_In1;
    m_In2 = op->m_In2;
    m_Out = op->m_Out;
    m_Owner = op->m_Owner;
    m_OperationGuid = op->m_OperationGuid;
    m_OperationFunction = op->m_OperationFunction;
    m_HasOperationFunction = op->m_HasOperationFunction;
    return CK_OK;
}

CKSTRING CKParameterOperation::GetClassName() {
    return "Parameter Operation";
}

int CKParameterOperation::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKParameterOperation::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKParameterOperation::Register() {
    CKCLASSDEFAULTOPTIONS(CKParameterOperation, CK_DEPENDENCIES_COPY);
}

CKParameterOperation *CKParameterOperation::CreateInstance(CKContext *Context) {
    return new CKParameterOperation(Context);
}
