#include "CKBehavior.h"

#include "CKBehaviorIO.h"
#include "CKBehaviorLink.h"
#include "CKBehaviorPrototype.h"
#include "CKBeObject.h"
#include "CKScene.h"
#include "CKFile.h"
#include "CKParameter.h"
#include "CKParameterIn.h"
#include "CKParameterOut.h"
#include "CKParameterLocal.h"
#include "CKParameterOperation.h"
#include "CKParameterManager.h"
#include "CKStateChunk.h"
#include "CKDependencies.h"
#include "CKDebugContext.h"

#if defined(_MSC_VER)
#include <excpt.h>
#endif

static CKBehaviorIO **g_IosToDeactivate = nullptr;
static CKBehaviorIO **g_IosToActivate = nullptr;
static int g_IosBufferCapacity = 0;

static void EnsureIoBuffers(int neededCount) {
    if (neededCount <= g_IosBufferCapacity)
        return;

    delete[] g_IosToDeactivate;
    delete[] g_IosToActivate;

    g_IosToDeactivate = new CKBehaviorIO *[neededCount];
    g_IosToActivate = new CKBehaviorIO *[neededCount];
    g_IosBufferCapacity = neededCount;
}

CK_CLASSID CKBehavior::m_ClassID = CKCID_BEHAVIOR;

BehaviorGraphData::~BehaviorGraphData()
{
    delete[] m_BehaviorIterators;
    m_BehaviorIterators = nullptr;
    m_BehaviorIteratorCount = 0;
    m_BehaviorIteratorIndex = 0;
}

CK_BEHAVIOR_TYPE CKBehavior::GetType() {
    if ((m_Flags & CKBEHAVIOR_BUILDINGBLOCK) != 0)
        return CKBEHAVIORTYPE_BASE;
    if ((m_Flags & CKBEHAVIOR_SCRIPT) != 0)
        return CKBEHAVIORTYPE_SCRIPT;
    else
        return CKBEHAVIORTYPE_BEHAVIOR;
}

void CKBehavior::SetType(CK_BEHAVIOR_TYPE type) {
    // Only sets the SCRIPT flag when type is SCRIPT, does not clear otherwise
    if (type == CKBEHAVIORTYPE_SCRIPT) {
        m_Flags |= CKBEHAVIOR_SCRIPT;
    }
}

void CKBehavior::SetFlags(CK_BEHAVIOR_FLAGS flags) {
    m_Flags = flags;
}

CK_BEHAVIOR_FLAGS CKBehavior::GetFlags() {
    return (CK_BEHAVIOR_FLAGS) m_Flags;
}

CK_BEHAVIOR_FLAGS CKBehavior::ModifyFlags(CKDWORD Add, CKDWORD Remove) {
    m_Flags = (m_Flags | Add) & ~Remove;
    return (CK_BEHAVIOR_FLAGS) m_Flags;
}

void CKBehavior::UseGraph() {
    if (!m_GraphData) {
        m_GraphData = new BehaviorGraphData;
    }
    delete m_BlockData;
    m_BlockData = nullptr;
    m_Flags &= ~(CKBEHAVIOR_BUILDINGBLOCK | CKBEHAVIOR_USEFUNCTION);
}

void CKBehavior::UseFunction() {
    if (!m_BlockData) {
        m_BlockData = new BehaviorBlockData;
    }
    if (m_GraphData) {
        delete m_GraphData;
        m_GraphData = nullptr;
    }
    m_Flags |= CKBEHAVIOR_BUILDINGBLOCK | CKBEHAVIOR_USEFUNCTION;
}

CKBOOL CKBehavior::IsUsingFunction() {
    return (m_Flags & CKBEHAVIOR_USEFUNCTION) != 0;
}

CKBOOL CKBehavior::IsTargetable() {
    return (m_Flags & CKBEHAVIOR_TARGETABLE) != 0;
}

CKBeObject *CKBehavior::GetTarget() {
    if (!m_InputTargetParam)
        return (CKBeObject *) m_Context->GetObject(m_Owner);

    CKParameterIn *targetParam = GetTargetParameter();
    if (!targetParam)
        return nullptr;

    CK_ID objId = 0;
    CKParameter *param = targetParam->GetRealSource();
    if (param)
        param->GetValue(&objId);
    CKBeObject *obj = (CKBeObject *) m_Context->GetObject(objId);
    CKParameterManager *pm = m_Context->GetParameterManager();
    CK_CLASSID cid = pm->TypeToClassID(targetParam->GetType());
    if (!CKIsChildClassOf(obj, cid))
        return nullptr;
    return obj;
}

CKERROR CKBehavior::UseTarget(CKBOOL Use) {
    if (!Use) {
        // Disable target usage
        if (m_InputTargetParam) {
            // Destroy existing target parameter
            m_Context->DestroyObject(m_InputTargetParam, CK_DESTROY_NONOTIFY);

            m_InputTargetParam = 0;

            // Restore parent's compatible class if needed
            CKBehavior *parent = GetParent();
            if (parent) {
                if (CKIsChildClassOf(m_CompatibleClassID, parent->GetCompatibleClassID())) {
                    parent->SetCompatibleClassID(m_CompatibleClassID);
                }
            }
        }
        return CK_OK;
    }

    if (!IsTargetable()) {
        return CKERR_INVALIDPARAMETER;
    }

    // Enable target usage
    if (!m_InputTargetParam) {
        // Create new target parameter
        CKParameterManager *pm = m_Context->GetParameterManager();
        CKGUID typeGuid = pm->ClassIDToGuid(m_CompatibleClassID);

        CKParameterIn *targetParam = m_Context->CreateCKParameterIn("Target", typeGuid);
        if (targetParam) {
            m_InputTargetParam = targetParam->GetID();
            targetParam->SetOwner(this);
        }

        // Update sub-behaviors
        CKBehavior *parent = GetParent();
        if (parent) {
            // Enabling target usage removes this behavior from the set of sub-behaviors that
            // constrain the owner type. Recompute from scratch using only sub-behaviors that
            // don't use a target parameter.
            CK_CLASSID newCid = CKCID_BEOBJECT;
            const int subCount = parent->GetSubBehaviorCount();
            for (int i = 0; i < subCount; ++i) {
                CKBehavior *sub = parent->GetSubBehavior(i);
                if (sub && sub->m_InputTargetParam == 0) {
                    const CK_CLASSID subCID = sub->GetCompatibleClassID();
                    if (CKIsChildClassOf(subCID, newCid)) {
                        newCid = subCID;
                    }
                }
            }
            parent->SetCompatibleClassID(newCid);
        }
    }

    return CK_OK;
}

CKBOOL CKBehavior::IsUsingTarget() {
    return m_InputTargetParam != 0;
}

CKParameterIn *CKBehavior::GetTargetParameter() {
    return (CKParameterIn *) m_Context->GetObject(m_InputTargetParam);
}

void CKBehavior::SetAsTargetable(CKBOOL target) {
    if (target) {
        m_Flags |= CKBEHAVIOR_TARGETABLE;
    } else {
        UseTarget(FALSE);
        m_Flags &= ~CKBEHAVIOR_TARGETABLE;
    }
}

CKParameterIn *CKBehavior::ReplaceTargetParameter(CKParameterIn *targetParam) {
    if (!m_InputTargetParam || !CKIsChildClassOf(targetParam, CKCID_PARAMETERIN))
        return nullptr;

    CK_ID oldParamID = m_InputTargetParam;
    m_InputTargetParam = targetParam->GetID();
    targetParam->SetOwner(this);
    return (CKParameterIn *) m_Context->GetObject(oldParamID);
}

CKParameterIn *CKBehavior::RemoveTargetParameter() {
    CK_ID oldParamID = m_InputTargetParam;
    m_InputTargetParam = 0;
    return (CKParameterIn *) m_Context->GetObject(oldParamID);
}

CK_CLASSID CKBehavior::GetCompatibleClassID() {
    return m_CompatibleClassID;
}

void CKBehavior::SetCompatibleClassID(CK_CLASSID cid) {
    m_CompatibleClassID = cid;
}

void CKBehavior::SetFunction(CKBEHAVIORFCT fct) {
    if (m_BlockData)
        m_BlockData->m_Function = fct;
}

CKBEHAVIORFCT CKBehavior::GetFunction() {
    if (!m_BlockData)
        return nullptr;
    return m_BlockData->m_Function;
}

void CKBehavior::SetCallbackFunction(CKBEHAVIORCALLBACKFCT fct) {
    if (m_BlockData)
        m_BlockData->m_Callback = fct;
}

int CKBehavior::CallCallbackFunction(CKDWORD Message) {
    // Index maps CKM_BEHAVIOR* message to the corresponding CKCB_BEHAVIOR* bitmask.
    static const CKDWORD MessageFlags[CKM_MAX_BEHAVIOR_CALLBACKS] = {
        0x00000000,                 // 0 - unused
        CKCB_BEHAVIORPRESAVE,
        CKCB_BEHAVIORDELETE,
        CKCB_BEHAVIORATTACH,
        CKCB_BEHAVIORDETACH,
        CKCB_BEHAVIORPAUSE,
        CKCB_BEHAVIORRESUME,
        CKM_BEHAVIORCREATE,
        0x00000000,                 // 8 - unused
        CKCB_BEHAVIORRESET,
        CKCB_BEHAVIORPOSTSAVE,
        CKCB_BEHAVIORLOAD,
        CKCB_BEHAVIOREDITED,
        CKCB_BEHAVIORSETTINGSEDITED,
        CKCB_BEHAVIORREADSTATE,
        CKCB_BEHAVIORNEWSCENE,
        CKCB_BEHAVIORACTIVATESCRIPT,
        CKCB_BEHAVIORDEACTIVATESCRIPT,
        CKCB_BEHAVIORRESETINBREAKPOINT,
    };

    if (Message >= CKM_MAX_BEHAVIOR_CALLBACKS) {
        return CK_OK;
    }

    // Get behavior block data and validate callback conditions
    BehaviorBlockData *blockData = m_BlockData;
    if (!blockData || !blockData->m_Callback)
        return CK_OK;

    // Check if this message type is enabled in the callback mask
    if (!(MessageFlags[Message] & blockData->m_CallbackMask))
        return CK_OK;

    // Prepare context for callback
    CKContext *context = m_Context;
    context->m_BehaviorContext.CallbackArg = blockData->m_CallbackArg;
    context->m_BehaviorContext.CallbackMessage = Message;
    context->m_BehaviorContext.Behavior = this;

    // Execute the callback with prepared context
    return blockData->m_Callback(context->m_BehaviorContext);
}

int CKBehavior::CallSubBehaviorsCallbackFunction(CKDWORD Message, CKGUID *behguid) {
    int result = -1;
    if (m_GraphData) {
        int r = 0;
        for (auto *it = m_GraphData->m_SubBehaviors.Begin(); it != m_GraphData->m_SubBehaviors.End(); ++it) {
            CKBehavior *beh = (CKBehavior *) *it;
            if (beh) {
                r = beh->CallSubBehaviorsCallbackFunction(Message, behguid);
                if (r >= 0)
                    result = r;
            }
        }
    } else if (m_BlockData) {
        if (!behguid || m_BlockData->m_Guid == *behguid) {
            result = CallCallbackFunction(Message);
        }
    }

    return result;
}

CKBOOL CKBehavior::IsActive() {
    return (m_Flags & CKBEHAVIOR_ACTIVE) != 0;
}

int CKBehavior::Execute(float delta) {
    int retVal = 0;

    VxTimeProfiler profiler;

    CKBehaviorManager *behaviorManager = m_Context->GetBehaviorManager();
    behaviorManager->m_CurrentBehavior = this;
    ++m_Context->m_Stats.BehaviorsExecuted;

    if ((m_Flags & CKBEHAVIOR_EXECUTEDLASTFRAME) == 0) {
        m_Flags |= CKBEHAVIOR_EXECUTEDLASTFRAME;
        behaviorManager->m_Behaviors.PushBack(this);
    }

    if ((m_Flags & CKBEHAVIOR_USEFUNCTION) != 0) {
        retVal = ExecuteFunction();
    } else {
        CheckIOsActivation();

        int iterationCount = 0;
        int maxIterations = behaviorManager->GetBehaviorMaxIteration();

        while (m_GraphData->m_BehaviorIteratorIndex > 0) {
            if (iterationCount > maxIterations) {
                WarnInfiniteLoop();
                retVal = CKBR_INFINITELOOP;
                break;
            }
            ++iterationCount;

            --m_GraphData->m_BehaviorIteratorIndex;
            CKBehavior *subBehavior = m_GraphData->m_BehaviorIterators[m_GraphData->m_BehaviorIteratorIndex];

            subBehavior->m_Flags &= ~CKBEHAVIOR_RESERVED0;

            subBehavior->Execute(delta);

            behaviorManager->m_CurrentBehavior = this;
            FindNextBehaviorsToExecute(subBehavior);
        }
        CheckBehaviorActivity();
    }

    behaviorManager->m_CurrentBehavior = nullptr;

    if (m_Context->m_ProfilingEnabled) {
        m_LastExecutionTime += profiler.Current();
    }

    return retVal;
}

CKBOOL CKBehavior::IsParentScriptActiveInScene(CKScene *scn) {
    CKBeObject *owner = GetOwner();
    if (!owner)
        return FALSE;

    if (CKIsChildClassOf(owner, CKCID_LEVEL))
        return TRUE;

    if (!owner->IsInScene(scn))
        return FALSE;

    CKBehavior *beh = this;
    while (beh->GetParent()) {
        beh = beh->GetParent();
    }
    return (scn->GetObjectFlags(beh) & CK_SCENEOBJECT_ACTIVE) != 0;
}

int CKBehavior::GetShortestDelay(CKBehavior *beh) {
    XObjectPointerArray behParsed;
    return InternalGetShortestDelay(beh, behParsed);
}

CKBeObject *CKBehavior::GetOwner() {
    return (CKBeObject *) m_Context->GetObject(m_Owner);
}

CKBehavior *CKBehavior::GetParent() {
    return (CKBehavior *) m_Context->GetObject(m_BehParent);
}

CKBehavior *CKBehavior::GetOwnerScript() {
    CKBehavior *beh = this;
    while (beh->GetParent()) {
        beh = beh->GetParent();
    }
    return (beh->m_Flags & CKBEHAVIOR_SCRIPT) ? beh : nullptr;
}

CKERROR CKBehavior::InitFromPrototype(CKBehaviorPrototype *proto) {
    if (!proto)
        return CKERR_INVALIDPARAMETER;

    if (!m_BlockData)
        return CKERR_INVALIDOBJECT;

    SetName(proto->m_Name.Str(), TRUE);

    InitFctPtrFromPrototype(proto);

    m_BlockData->m_Guid = proto->GetGuid();

    SetType(CKBEHAVIORTYPE_BASE);

    m_CompatibleClassID = proto->GetApplyToClassID();

    // --- Retrieve a state object using the GUID.
    CKObjectDeclaration *decl = CKGetObjectDeclarationFromGuid(m_BlockData->m_Guid);
    m_BlockData->m_Version = decl ? decl->GetVersion() : 0x10000;

    // --- Extract the counts and lists of IOs and parameters from the prototype.
    int outIOCount = proto->m_OutIOCount;
    int inIOCount = proto->m_InIOCount;
    int inParamCount = proto->m_InParameterCount;
    int outParamCount = proto->m_OutParameterCount;
    int localParamCount = proto->m_LocalParameterCount;

    CKBEHAVIORIO_DESC **outIOList = proto->m_OutIOList;
    CKBEHAVIORIO_DESC **inIOList = proto->m_InIOList;
    CKPARAMETER_DESC **inParamList = proto->m_InParameterList;
    CKPARAMETER_DESC **outParamList = proto->m_OutParameterList;
    CKPARAMETER_DESC **localParamList = proto->m_LocalParameterList;

    // --- Create Input IOs.
    for (int i = 0; i < inIOCount; i++) {
        CKBehaviorIO *inputIO = CreateInput(nullptr);
        inputIO->SetName(inIOList[i]->Name, TRUE);
        inputIO->m_ObjectFlags = inIOList[i]->Flags | (inputIO->m_ObjectFlags & ~CK_OBJECT_IOTYPEMASK);
    }

    // --- Create Output IOs.
    for (int i = 0; i < outIOCount; i++) {
        CKBehaviorIO *outputIO = CreateOutput(nullptr);
        outputIO->SetName(outIOList[i]->Name, TRUE);
        outputIO->m_ObjectFlags = outIOList[i]->Flags | (outputIO->m_ObjectFlags & ~CK_OBJECT_IOTYPEMASK);
    }

    // --- Create Input Parameters.
    for (int i = 0; i < inParamCount; i++) {
        CKParameterIn *inputParam = CreateInputParameter(nullptr, inParamList[i]->Guid);
        if (inputParam)
            inputParam->SetName(inParamList[i]->Name, TRUE);
    }

    // --- Create Output Parameters.
    for (int i = 0; i < outParamCount; i++) {
        CKParameterOut *outputParam = CreateOutputParameter(nullptr, outParamList[i]->Guid);
        if (outputParam)
            outputParam->SetName(outParamList[i]->Name, TRUE);

        // Set the default value if one is provided.
        CKPARAMETER_DESC *desc = outParamList[i];
        if (desc->DefaultValueString) {
            outputParam->SetStringValue(desc->DefaultValueString);
        } else if (desc->DefaultValue && desc->DefaultValueSize) {
            outputParam->SetValue(desc->DefaultValue, desc->DefaultValueSize);
        }
    }

    // --- Create Local Parameters.
    for (int i = 0; i < localParamCount; i++) {
        CKParameterLocal *localParam = CreateLocalParameter(nullptr, localParamList[i]->Guid);
        if (localParam) {
            localParam->SetName(localParamList[i]->Name, true);
            CKPARAMETER_DESC *desc = localParamList[i];
            if (desc->DefaultValueString) {
                localParam->SetStringValue(desc->DefaultValueString);
            } else if (desc->DefaultValue && desc->DefaultValueSize) {
                localParam->SetValue(desc->DefaultValue, desc->DefaultValueSize);
            }
        }
    }
    return CK_OK;
}

CKERROR CKBehavior::InitFromGuid(CKGUID Guid) {
    CKBehaviorPrototype *proto = CKGetPrototypeFromGuid(Guid);
    if (!proto)
        return CK_OK;

    int err = InitFromPrototype(proto);
    m_Flags |= CKBEHAVIOR_BUILDINGBLOCK;
    return err;
}

CKERROR CKBehavior::InitFctPtrFromGuid(CKGUID Guid) {
    CKBehaviorPrototype *proto = GetPrototype();
    if (!proto)
        return CKERR_INVALIDPARAMETER;
    return InitFctPtrFromPrototype(proto);
}

CKERROR CKBehavior::InitFctPtrFromPrototype(CKBehaviorPrototype *proto) {
    if (!m_BlockData)
        return CKERR_INVALIDOBJECT;

    m_BlockData->m_Callback = proto->m_CallbackFctPtr;
    m_BlockData->m_CallbackMask = proto->m_CallBackMask;
    m_BlockData->m_CallbackArg = proto->m_CallBackParam;

    m_Flags |= proto->GetBehaviorFlags() | CKBEHAVIOR_BUILDINGBLOCK;

    if (proto->GetBehaviorFlags() & CKBEHAVIOR_LOCKED) {
        if (m_Flags & CKBEHAVIOR_LOCKED) {
            for (CKBehavior *parent = GetParent(); parent; parent = parent->GetParent()) {
                parent->m_Flags |= CKBEHAVIOR_LOCKED;
            }
        }
    } else {
        m_Flags &= ~CKBEHAVIOR_LOCKED;
    }

    if (proto->GetFunction()) {
        SetFunction(proto->GetFunction());
        UseFunction();
    }
    return CK_OK;
}

CKGUID CKBehavior::GetPrototypeGuid() {
    if (!m_BlockData)
        return CKGUID();
    return m_BlockData->m_Guid;
}

CKBehaviorPrototype *CKBehavior::GetPrototype() {
    if (!m_BlockData)
        return nullptr;
    return CKGetPrototypeFromGuid(m_BlockData->m_Guid);
}

CKSTRING CKBehavior::GetPrototypeName() {
    CKBehaviorPrototype *proto = GetPrototype();
    if (!proto)
        return nullptr;
    return proto->GetName();
}

CKDWORD CKBehavior::GetVersion() {
    if (!m_BlockData)
        return 0;
    return m_BlockData->m_Version;
}

void CKBehavior::SetVersion(CKDWORD version) {
    if (m_BlockData)
        m_BlockData->m_Version = version;
}

void CKBehavior::ActivateOutput(int pos, CKBOOL active) {
    if (pos < 0)
        return;

    CKBehaviorIO *io = GetOutput(pos);
    if (!io)
        return;

    io->Activate(active);
}

CKBOOL CKBehavior::IsOutputActive(int pos) {
    if (pos < 0)
        return FALSE;

    CKBehaviorIO *io = GetOutput(pos);
    return io ? io->IsActive() : FALSE;
}

CKBehaviorIO *CKBehavior::RemoveOutput(int pos) {
    CKBehaviorIO *io = GetOutput(pos);
    if (!io)
        return nullptr;
    m_OutputArray.RemoveAt(pos);
    return io;
}

CKERROR CKBehavior::DeleteOutput(int pos) {
    CKBehaviorIO *io = RemoveOutput(pos);
    if (!io)
        return CKERR_INVALIDPARAMETER;
    m_Context->DestroyObject(io);
    return CK_OK;
}

CKBehaviorIO *CKBehavior::GetOutput(int pos) {
    if (pos < 0 || pos >= m_OutputArray.Size())
        return nullptr;
    return (CKBehaviorIO *) m_OutputArray[pos];
}

int CKBehavior::GetOutputCount() {
    return m_OutputArray.Size();
}

int CKBehavior::GetOutputPosition(CKBehaviorIO *io) {
    if (io) {
        for (int i = 0; i < m_OutputArray.Size(); ++i) {
            if (m_OutputArray[i] == io) {
                return i;
            }
        }
    }
    return -1;
}

int CKBehavior::AddOutput(CKSTRING name) {
    CKBehaviorIO *io = CreateOutput(name);
    return io ? m_OutputArray.Size() - 1 : -1;
}

CKBehaviorIO *CKBehavior::ReplaceOutput(int pos, CKBehaviorIO *io) {
    if (pos < 0 || pos >= m_OutputArray.Size())
        return nullptr;

    CKBehaviorIO *oldIO = (CKBehaviorIO *) m_OutputArray[pos];
    m_OutputArray[pos] = io;
    return oldIO;
}

CKBehaviorIO *CKBehavior::CreateOutput(CKSTRING name) {
    const CK_OBJECTCREATION_OPTIONS options = ((m_ObjectFlags & CK_OBJECT_DYNAMIC) == CK_OBJECT_DYNAMIC)
                                                  ? CK_OBJECTCREATION_DYNAMIC
                                                  : CK_OBJECTCREATION_NONAMECHECK;
    CKBehaviorIO *io = (CKBehaviorIO *) m_Context->CreateObject(CKCID_BEHAVIORIO, name, options, 0);
    if (io) {
        m_OutputArray.PushBack(io);
        io->SetOwner(this);
        io->SetType(CK_BEHAVIORIO_OUT);
    }
    return io;
}

void CKBehavior::ActivateInput(int pos, CKBOOL active) {
    if (pos < 0)
        return;

    CKBehaviorIO *io = GetInput(pos);
    if (!io)
        return;

    io->Activate(active);
}

CKBOOL CKBehavior::IsInputActive(int pos) {
    if (pos < 0)
        return FALSE;

    CKBehaviorIO *io = GetInput(pos);
    return io ? io->IsActive() : FALSE;
}

CKBehaviorIO *CKBehavior::RemoveInput(int pos) {
    CKBehaviorIO *io = GetInput(pos);
    if (!io)
        return nullptr;
    m_InputArray.RemoveAt(pos);
    return io;
}

CKERROR CKBehavior::DeleteInput(int pos) {
    CKBehaviorIO *io = RemoveInput(pos);
    if (!io)
        return CKERR_INVALIDPARAMETER;
    m_Context->DestroyObject(io);
    return CK_OK;
}

CKBehaviorIO *CKBehavior::GetInput(int pos) {
    if (pos < 0 || pos >= m_InputArray.Size())
        return nullptr;
    return (CKBehaviorIO *) m_InputArray[pos];
}

int CKBehavior::GetInputCount() {
    return m_InputArray.Size();
}

int CKBehavior::GetInputPosition(CKBehaviorIO *io) {
    if (!io) return -1;

    for (int i = 0; i < m_InputArray.Size(); ++i) {
        if (m_InputArray[i] == io) {
            return i;
        }
    }
    return -1;
}

int CKBehavior::AddInput(CKSTRING name) {
    CKBehaviorIO *io = CreateInput(name);
    return io ? m_InputArray.Size() - 1 : -1;
}

CKBehaviorIO *CKBehavior::ReplaceInput(int pos, CKBehaviorIO *io) {
    if (pos < 0 || pos >= m_InputArray.Size())
        return nullptr;

    CKBehaviorIO *oldIO = (CKBehaviorIO *) m_InputArray[pos];
    m_InputArray[pos] = io;
    return oldIO;
}

CKBehaviorIO *CKBehavior::CreateInput(CKSTRING name) {
    const CK_OBJECTCREATION_OPTIONS options = ((m_ObjectFlags & CK_OBJECT_DYNAMIC) == CK_OBJECT_DYNAMIC)
                                                  ? CK_OBJECTCREATION_DYNAMIC
                                                  : CK_OBJECTCREATION_NONAMECHECK;
    CKBehaviorIO *io = (CKBehaviorIO *) m_Context->CreateObject(CKCID_BEHAVIORIO, name, options, 0);
    if (io) {
        m_InputArray.PushBack(io);
        io->SetOwner(this);
        io->SetType(CK_BEHAVIORIO_IN);
    }
    return io;
}

CKERROR CKBehavior::ExportInputParameter(CKParameterIn *p) {
    if (!p)
        return CKERR_INVALIDPARAMETER;
    m_InParameter.PushBack(p);
    return CK_OK;
}

CKParameterIn *CKBehavior::CreateInputParameter(CKSTRING name, CKParameterType type) {
    CKParameterIn *pin = m_Context->CreateCKParameterIn(name, type);
    if (pin)
        AddInputParameter(pin);
    return pin;
}

CKParameterIn *CKBehavior::CreateInputParameter(CKSTRING name, CKGUID guid) {
    CKParameterManager *pm = m_Context->GetParameterManager();
    CKParameterType type = pm->ParameterGuidToType(guid);
    return CreateInputParameter(name, type);
}

CKParameterIn *CKBehavior::InsertInputParameter(int pos, CKSTRING name, CKParameterType type) {
    CKParameterIn *pin = m_Context->CreateCKParameterIn(name, type);
    if (pin) {
        m_InParameter.Insert(pos, pin);
        pin->SetOwner(this);
    }
    return pin;
}

void CKBehavior::AddInputParameter(CKParameterIn *in) {
    if (in) {
        m_InParameter.PushBack(in);
        in->SetOwner(this);
    }
}

int CKBehavior::GetInputParameterPosition(CKParameterIn *in) {
    if (in) {
        for (int i = 0; i < m_InParameter.Size(); ++i) {
            if (m_InParameter[i] == in) {
                return i;
            }
        }
    }
    return -1;
}

CKParameterIn *CKBehavior::GetInputParameter(int pos) {
    if (pos < 0 || pos >= m_InParameter.Size())
        return nullptr;
    return (CKParameterIn *) m_InParameter[pos];
}

CKParameterIn *CKBehavior::RemoveInputParameter(int pos) {
    CKParameterIn *param = GetInputParameter(pos);
    if (param) {
        m_InParameter.RemoveAt(pos);
    }
    return param;
}

CKParameterIn *CKBehavior::ReplaceInputParameter(int pos, CKParameterIn *param) {
    if (!param)
        return nullptr;

    CKParameterIn *oldParam = GetInputParameter(pos);
    if (!oldParam)
        return nullptr;

    m_InParameter[pos] = param;
    return oldParam;
}

int CKBehavior::GetInputParameterCount() {
    return m_InParameter.Size();
}

CKERROR CKBehavior::GetInputParameterValue(int pos, void *buf) {
    CKParameterIn *inParam = GetInputParameter(pos);
    if (!inParam)
        return CKERR_INVALIDPARAMETER;
    CKParameter *param = inParam->GetRealSource();
    if (!param)
        return CKERR_NOTINITIALIZED;
    return param->GetValue(buf);
}

void *CKBehavior::GetInputParameterReadDataPtr(int pos) {
    CKParameterIn *inParam = GetInputParameter(pos);
    if (!inParam)
        return nullptr;
    return inParam->GetReadDataPtr();
}

CKObject *CKBehavior::GetInputParameterObject(int pos) {
    CKParameterIn *inParam = GetInputParameter(pos);
    if (!inParam)
        return nullptr;

    CK_ID objId = 0;
    CKParameter *param = inParam->GetRealSource();
    if (param)
        param->GetValue(&objId);
    return m_Context->GetObject(objId);
}

CKBOOL CKBehavior::IsInputParameterEnabled(int pos) {
    CKParameterIn *inParam = GetInputParameter(pos);
    if (!inParam)
        return FALSE;
    return inParam->IsEnabled();
}

void CKBehavior::EnableInputParameter(int pos, CKBOOL enable) {
    CKParameterIn *inParam = GetInputParameter(pos);
    if (!inParam)
        return;
    inParam->Enable(enable);
}

CKERROR CKBehavior::ExportOutputParameter(CKParameterOut *p) {
    if (!p)
        return CKERR_INVALIDPARAMETER;
    m_OutParameter.PushBack(p);
    return CK_OK;
}

CKParameterOut *CKBehavior::CreateOutputParameter(CKSTRING name, CKParameterType type) {
    CKParameterOut *pout = m_Context->CreateCKParameterOut(name, type);
    if (pout)
        AddOutputParameter(pout);
    return pout;
}

CKParameterOut *CKBehavior::CreateOutputParameter(CKSTRING name, CKGUID guid) {
    CKParameterManager *pm = m_Context->GetParameterManager();
    CKParameterType type = pm->ParameterGuidToType(guid);
    return CreateOutputParameter(name, type);
}

CKParameterOut *CKBehavior::InsertOutputParameter(int pos, CKSTRING name, CKParameterType type) {
    CKParameterOut *pout = m_Context->CreateCKParameterOut(name, type);
    if (pout) {
        m_OutParameter.Insert(pos, pout);
        pout->SetOwner(this);
    }
    return pout;
}

CKParameterOut *CKBehavior::GetOutputParameter(int pos) {
    if (pos < 0 || pos >= m_OutParameter.Size())
        return nullptr;
    return (CKParameterOut *) m_OutParameter[pos];
}

int CKBehavior::GetOutputParameterPosition(CKParameterOut *p) {
    if (p) {
        for (int i = 0; i < m_OutParameter.Size(); ++i) {
            if (m_OutParameter[i] == p) {
                return i;
            }
        }
    }
    return -1;
}

CKParameterOut *CKBehavior::ReplaceOutputParameter(int pos, CKParameterOut *p) {
    if (!p)
        return nullptr;

    CKParameterOut *oldParam = GetOutputParameter(pos);
    if (!oldParam)
        return nullptr;

    m_OutParameter[pos] = p;
    return oldParam;
}

CKParameterOut *CKBehavior::RemoveOutputParameter(int pos) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (param) {
        m_OutParameter.RemoveAt(pos);
    }
    return param;
}

void CKBehavior::AddOutputParameter(CKParameterOut *out) {
    if (out) {
        m_OutParameter.PushBack(out);
        out->SetOwner(this);
    }
}

int CKBehavior::GetOutputParameterCount() {
    return m_OutParameter.Size();
}

CKERROR CKBehavior::GetOutputParameterValue(int pos, void *buf) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (!param)
        return CKERR_INVALIDPARAMETER;
    return param->GetValue(buf);
}

CKERROR CKBehavior::SetOutputParameterValue(int pos, const void *buf, int size) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (!param)
        return CKERR_INVALIDPARAMETER;
    return param->SetValue(buf, size);
}

void *CKBehavior::GetOutputParameterWriteDataPtr(int pos) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (!param)
        return nullptr;
    return param->GetWriteDataPtr();
}

CKERROR CKBehavior::SetOutputParameterObject(int pos, CKObject *obj) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (!param)
        return CKERR_INVALIDPARAMETER;

    CK_ID objID = 0;
    if (obj)
        objID = obj->GetID();
    param->SetValue(&objID);
    return CK_OK;
}

CKObject *CKBehavior::GetOutputParameterObject(int pos) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (!param)
        return nullptr;

    CK_ID objID = 0;
    param->GetValue(&objID);
    return m_Context->GetObject(objID);
}

CKBOOL CKBehavior::IsOutputParameterEnabled(int pos) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (!param)
        return FALSE;
    return param->IsEnabled();
}

void CKBehavior::EnableOutputParameter(int pos, CKBOOL enable) {
    CKParameterOut *param = GetOutputParameter(pos);
    if (param)
        param->Enable(enable);
}

void CKBehavior::SetInputParameterDefaultValue(CKParameterIn *pin, CKParameter *plink) {
    CKBehavior *owner = (CKBehavior *) pin->GetOwner();
    if (!owner)
        return;

    if (owner->GetClassID() != CKCID_BEHAVIOR)
        return;

    CKGUID guid = owner->GetPrototypeGuid();
    CKBehaviorPrototype *proto = CKGetPrototypeFromGuid(guid);
    if (!proto)
        return;

    int index = -1;
    for (int i = 0; i < owner->m_InParameter.Size(); ++i) {
        if (owner->m_InParameter[i] == pin) {
            index = i;
            break;
        }
    }

    if (index >= 0 && index < proto->m_InParameterCount) {
        CKPARAMETER_DESC *desc = proto->m_InParameterList[index];
        if (desc->DefaultValueString) {
            plink->SetStringValue(desc->DefaultValueString);
        } else if (desc->DefaultValue && desc->DefaultValueSize) {
            plink->SetValue(desc->DefaultValue, desc->DefaultValueSize);
        }
    }
}

CKParameterLocal *CKBehavior::CreateLocalParameter(CKSTRING name, CKParameterType type) {
    CKParameterLocal *param = m_Context->CreateCKParameterLocal(name, type);
    if (param)
        AddLocalParameter(param);
    return param;
}

CKParameterLocal *CKBehavior::CreateLocalParameter(CKSTRING name, CKGUID guid) {
    CKParameterManager *pm = m_Context->GetParameterManager();
    CKParameterType type = pm->ParameterGuidToType(guid);
    return CreateLocalParameter(name, type);
}

CKParameterLocal *CKBehavior::GetLocalParameter(int pos) {
    if (pos < 0 || pos >= m_LocalParameter.Size())
        return nullptr;
    return (CKParameterLocal *) m_LocalParameter[pos];
}

CKParameterLocal *CKBehavior::RemoveLocalParameter(int pos) {
    CKParameterLocal *param = GetLocalParameter(pos);
    if (param) {
        m_LocalParameter.RemoveAt(pos);
    }
    return param;
}

void CKBehavior::AddLocalParameter(CKParameterLocal *loc) {
    if (loc) {
        m_LocalParameter.PushBack(loc);
        loc->SetOwner(this);
    }
}

int CKBehavior::GetLocalParameterPosition(CKParameterLocal *loc) {
    if (loc) {
        for (int i = 0; i < m_LocalParameter.Size(); ++i) {
            if (m_LocalParameter[i] == loc) {
                return i;
            }
        }
    }
    return -1;
}

int CKBehavior::GetLocalParameterCount() {
    return m_LocalParameter.Size();
}

CKERROR CKBehavior::GetLocalParameterValue(int pos, void *buf) {
    CKParameterLocal *param = GetLocalParameter(pos);
    if (!param)
        return CKERR_INVALIDPARAMETER;
    return param->GetValue(buf);
}

CKERROR CKBehavior::SetLocalParameterValue(int pos, const void *buf, int size) {
    CKParameterLocal *param = GetLocalParameter(pos);
    if (!param)
        return CKERR_INVALIDPARAMETER;
    return param->SetValue(buf, size);
}

void *CKBehavior::GetLocalParameterWriteDataPtr(int pos) {
    CKParameterLocal *param = GetLocalParameter(pos);
    if (!param)
        return nullptr;
    return param->GetWriteDataPtr();
}

void *CKBehavior::GetLocalParameterReadDataPtr(int pos) {
    CKParameterLocal *param = GetLocalParameter(pos);
    if (!param)
        return nullptr;
    return param->GetReadDataPtr();
}

CKObject *CKBehavior::GetLocalParameterObject(int pos) {
    CKParameterLocal *param = GetLocalParameter(pos);
    if (!param)
        return nullptr;

    CK_ID objID = 0;
    param->GetValue(&objID);
    return m_Context->GetObject(objID);
}

CKERROR CKBehavior::SetLocalParameterObject(int pos, CKObject *obj) {
    CKParameterLocal *param = GetLocalParameter(pos);
    if (!param)
        return CKERR_INVALIDPARAMETER;

    CK_ID objID = 0;
    if (obj)
        objID = obj->GetID();
    param->SetValue(&objID);
    return CK_OK;
}

CKBOOL CKBehavior::IsLocalParameterSetting(int pos) {
    CKBehaviorPrototype *proto = GetPrototype();
    if (!proto)
        return FALSE;

    if (pos >= proto->m_LocalParameterCount) {
        CKParameterLocal *param = GetLocalParameter(pos);
        if (!param)
            return FALSE;
        return (param->GetObjectFlags() & CK_PARAMETEROUT_SETTINGS) != 0;
    }

    if (pos < 0)
        return FALSE;

    return proto->m_LocalParameterList[pos]->Type == 3;
}

void CKBehavior::Activate(CKBOOL Active, CKBOOL breset) {
    if (breset || m_Flags == CKBEHAVIOR_NONE)
        Reset();
    if (Active)
        m_Flags |= CKBEHAVIOR_ACTIVE;
    else
        m_Flags &= ~CKBEHAVIOR_ACTIVE;
}

CKERROR CKBehavior::AddSubBehavior(CKBehavior *cbk) {
    if (!m_GraphData || !cbk || cbk == this)
        return CKERR_INVALIDPARAMETER;

    if (!cbk->IsUsingTarget()) {
        CKBeObject *owner = this->GetOwner();
        if (owner) {
            if (!CKIsChildClassOf(owner, cbk->GetCompatibleClassID()))
                return CKERR_INCOMPATIBLECLASSID;
        }
    }

    // Save the current owner and parent of the sub–behavior.
    CKBeObject *origOwner = cbk->GetOwner();
    CKBehavior *origParent = cbk->GetParent();

    // Set this behavior as the new parent.
    cbk->SetParent(this);

    // Set the owner of the sub–behavior to be the same as this behavior's owner.
    CKBeObject *newOwner = this->GetOwner();
    CKERROR err = cbk->SetOwner(newOwner, TRUE);
    if (err != CK_OK) {
        // Restore original parent and owner if setting the owner failed.
        cbk->SetParent(origParent);
        cbk->SetOwner(origOwner, true);
        return err;
    }

    cbk->m_Flags &= ~CKBEHAVIOR_TOPMOST;

    m_GraphData->m_SubBehaviors.AddIfNotHere(cbk);
    m_GraphData->m_SubBehaviors.Sort(BehaviorPrioritySort);

    if (!cbk->m_InputTargetParam && CKIsChildClassOf(cbk->GetCompatibleClassID(), m_CompatibleClassID)) {
        SetCompatibleClassID(cbk->GetCompatibleClassID());
    }
    return CK_OK;
}

CKBehavior *CKBehavior::RemoveSubBehavior(CKBehavior *cbk) {
    if (!m_GraphData)
        return nullptr;
    int pos = m_GraphData->m_SubBehaviors.GetPosition(cbk);
    if (pos == -1)
        return nullptr;
    return RemoveSubBehavior(pos);
}

CKBehavior *CKBehavior::RemoveSubBehavior(int pos) {
    if (!m_GraphData)
        return nullptr;

    XObjectPointerArray &subBehaviors = m_GraphData->m_SubBehaviors;

    if (pos < 0 || pos >= subBehaviors.Size())
        return nullptr;

    CKBehavior *removed = (CKBehavior *) subBehaviors[pos];
    subBehaviors.RemoveAt(pos);

    // CK2.dll: If array becomes empty, just set m_CompatibleClassID = CKCID_BEOBJECT
    if (subBehaviors.Size() == 0) {
        m_CompatibleClassID = CKCID_BEOBJECT;
    } else {
        // Only set TOPMOST and clear parent when array still has elements
        removed->m_Flags |= CKBEHAVIOR_TOPMOST;
        removed->SetParent(nullptr);

        // Update compatible class ID if the removed behavior had same CompatibleClassID
        if (removed->GetCompatibleClassID() == m_CompatibleClassID) {
            CK_CLASSID newCid = CKCID_BEOBJECT;
            const int count = GetSubBehaviorCount();
            for (int i = 0; i < count; ++i) {
                CKBehavior *beh = GetSubBehavior(i);
                if (beh && !beh->m_InputTargetParam) {
                    CK_CLASSID cid = beh->GetCompatibleClassID();
                    if (CKIsChildClassOf(cid, newCid)) {
                        newCid = cid;
                    }
                }
            }
            m_CompatibleClassID = newCid;
        }
    }

    return removed;
}

CKBehavior *CKBehavior::GetSubBehavior(int pos) {
    if (!m_GraphData || pos < 0 || pos >= m_GraphData->m_SubBehaviors.Size())
        return nullptr;
    return (CKBehavior *) m_GraphData->m_SubBehaviors[pos];
}

int CKBehavior::GetSubBehaviorCount() {
    if (!m_GraphData)
        return 0;
    return m_GraphData->m_SubBehaviors.Size();
}

CKERROR CKBehavior::AddSubBehaviorLink(CKBehaviorLink *cbkl) {
    if (!m_GraphData || !cbkl)
        return CKERR_INVALIDPARAMETER;
    cbkl->SetOldFlags(cbkl->GetOldFlags() & ~CKBL_OLD_IN_DELAYED_LIST);
    m_GraphData->m_SubBehaviorLinks.AddIfNotHere(cbkl);
    return CK_OK;
}

CKBehaviorLink *CKBehavior::RemoveSubBehaviorLink(CKBehaviorLink *cbkl) {
    if (!m_GraphData)
        return nullptr;
    if (!m_GraphData->m_SubBehaviorLinks.Remove(cbkl))
        return nullptr;
    m_GraphData->m_Links.Remove(cbkl);
    return cbkl;
}

CKBehaviorLink *CKBehavior::RemoveSubBehaviorLink(int pos) {
    if (!m_GraphData)
        return nullptr;

    XObjectPointerArray &subBehaviorLinks = m_GraphData->m_SubBehaviorLinks;
    if (pos < 0 || pos >= subBehaviorLinks.Size())
        return nullptr;

    CKBehaviorLink *plink = (CKBehaviorLink *) subBehaviorLinks[pos];
    subBehaviorLinks.RemoveAt(pos);
    m_GraphData->m_Links.Remove(plink);
    return plink;
}

CKBehaviorLink *CKBehavior::GetSubBehaviorLink(int pos) {
    if (!m_GraphData || pos < 0 || pos >= m_GraphData->m_SubBehaviorLinks.Size())
        return nullptr;
    return (CKBehaviorLink *) m_GraphData->m_SubBehaviorLinks[pos];
}

int CKBehavior::GetSubBehaviorLinkCount() {
    if (!m_GraphData)
        return 0;
    return m_GraphData->m_SubBehaviorLinks.Size();
}

CKERROR CKBehavior::AddParameterOperation(CKParameterOperation *op) {
    if (!m_GraphData || !op)
        return CKERR_INVALIDPARAMETER;
    m_GraphData->m_Operations.AddIfNotHere(op);
    op->SetOwner(this);
    return CK_OK;
}

CKParameterOperation *CKBehavior::GetParameterOperation(int pos) {
    if (!m_GraphData || pos < 0 || pos >= m_GraphData->m_Operations.Size())
        return nullptr;
    return (CKParameterOperation *) m_GraphData->m_Operations[pos];
}

CKParameterOperation *CKBehavior::RemoveParameterOperation(int pos) {
    if (!m_GraphData)
        return nullptr;

    XObjectPointerArray &operations = m_GraphData->m_Operations;
    CKObject **op = operations.RemoveAt(pos);
    if (!op)
        return nullptr;

    return (CKParameterOperation *)*op;
}

CKParameterOperation *CKBehavior::RemoveParameterOperation(CKParameterOperation *op) {
    if (!m_GraphData || !op)
        return nullptr;

    if (!m_GraphData->m_Operations.Remove(op))
        return nullptr;

    op->SetOwner(nullptr);
    return op;
}

int CKBehavior::GetParameterOperationCount() {
    if (!m_GraphData)
        return 0;
    return m_GraphData->m_Operations.Size();
}

int CKBehavior::GetPriority() {
    return m_Priority;
}

void CKBehavior::SetPriority(int priority) {
    m_Priority = priority;

    // Update parent behavior if exists
    if (CKBehavior *parent = GetParent()) {
        parent->SortSubs();
    } else if (CKBeObject *owner = GetOwner()) {
        // Update owner's script order if part of scene
        if (m_Flags & CKBEHAVIOR_SCRIPT) {
            owner->SortScripts();
        }
    }
}

float CKBehavior::GetLastExecutionTime() {
    return m_LastExecutionTime;
}

CKERROR CKBehavior::SetOwner(CKBeObject *owner, CKBOOL callback) {
    CKBeObject *currentOwner = GetOwner();
    if (currentOwner == owner) {
        SetSubBehaviorOwner(currentOwner, callback);
        return CK_OK;
    }

    CK_ID previousOwnerID = m_Owner;
    CKERROR err = CK_OK;

    if (callback) {
        // Notify detachment from current owner
        if (currentOwner) {
            err = CallCallbackFunction(CKM_BEHAVIORDETACH);
            if (err != CK_OK)
                return err;
        }

        m_Owner = owner ? owner->GetID() : 0;

        // Notify attachment to new owner
        if (owner) {
            err = CallCallbackFunction(CKM_BEHAVIORATTACH);
            if (err != CK_OK) {
                // Rollback owner change on failure (CK2.dll restores previous owner and re-attaches).
                if (previousOwnerID) {
                    m_Owner = previousOwnerID;
                    CallCallbackFunction(CKM_BEHAVIORATTACH);
                }
                if (err == CKBR_LOCKED)
                    m_Context->OutputToConsoleExBeep("Cannot add or move behavior, because the behavior is locked.");
                return err;
            }
        }
    } else {
        m_Owner = owner ? owner->GetID() : 0;
    }

    // Update local parameters and sub-behaviors
    for (auto it = m_LocalParameter.Begin(); it != m_LocalParameter.End(); ++it) {
        CKParameterLocal *param = (CKParameterLocal *) *it;
        if (param) {
            param->SetOwner(this);
        }
    }

    CKBeObject *actualOwner = GetOwner();
    err = SetSubBehaviorOwner(actualOwner, callback);
    if (err != CK_OK) {
        // Rollback to previous owner for graph sub-behaviors.
        m_Owner = previousOwnerID;
        CKBeObject *previousOwner = (CKBeObject *)m_Context->GetObject(previousOwnerID);
        SetSubBehaviorOwner(previousOwner, callback);
    }

    if (err == CKBR_LOCKED)
        m_Context->OutputToConsoleExBeep("Cannot add or move behavior, because the behavior is locked.");

    return err;
}

CKERROR CKBehavior::SetSubBehaviorOwner(CKBeObject *owner, CKBOOL callback) {
    if (!m_GraphData)
        return CK_OK;

    for (auto it = m_GraphData->m_SubBehaviors.Begin(); it != m_GraphData->m_SubBehaviors.End(); ++it) {
        CKBehavior *subBeh = (CKBehavior *) *it;
        if (subBeh) {
            CKERROR err = subBeh->SetOwner(owner, callback);
            if (err != CK_OK)
                return err;
        }
    }
    return CK_OK;
}

void CKBehavior::NotifyEdition() {
    CallCallbackFunction(CKM_BEHAVIOREDITED);
}

void CKBehavior::NotifySettingsEdition() {
    CallCallbackFunction(CKM_BEHAVIORSETTINGSEDITED);
}

CKStateChunk *CKBehavior::GetInterfaceChunk() {
    return m_InterfaceChunk;
}

void CKBehavior::SetInterfaceChunk(CKStateChunk *state) {
    if (m_InterfaceChunk)
        delete m_InterfaceChunk;
    m_InterfaceChunk = state;
}

CKBehavior::CKBehavior(CKContext *Context, CKSTRING name) : CKSceneObject(Context, name) {
    m_LastExecutionTime = 0.0f;
    m_Owner = 0;
    m_InputTargetParam = 0;
    m_CompatibleClassID = CKCID_BEOBJECT;
    m_Priority = 0;
    m_Flags = CKBEHAVIOR_BUILDINGBLOCK | CKBEHAVIOR_TOPMOST | CKBEHAVIOR_USEFUNCTION;
    m_GraphData = nullptr;
    m_BlockData = new BehaviorBlockData();
    m_InterfaceChunk = nullptr;
    m_BehParent = 0;
}

CKBehavior::~CKBehavior() {
    // Cleanup interface chunk
    if (m_InterfaceChunk) {
        delete m_InterfaceChunk;
        m_InterfaceChunk = nullptr;
    }

    // Cleanup block data
    delete m_BlockData;
    m_BlockData = nullptr;

    // Cleanup graph data
    if (m_GraphData) {
        delete m_GraphData;
        m_GraphData = nullptr;
    }

    // Clear parameter arrays
    m_LocalParameter.Clear();
    m_OutParameter.Clear();
    m_InParameter.Clear();
    m_OutputArray.Clear();
    m_InputArray.Clear();
}

CK_CLASSID CKBehavior::GetClassID() {
    return m_ClassID;
}

void CKBehavior::PreSave(CKFile *file, CKDWORD flags) {
    CKObject::PreSave(file, flags);

    if (m_GraphData) {
        file->SaveObjects(m_GraphData->m_SubBehaviors.Begin(), m_GraphData->m_SubBehaviors.Size());
        file->SaveObjects(m_GraphData->m_SubBehaviorLinks.Begin(), m_GraphData->m_SubBehaviorLinks.Size());
        file->SaveObjects(m_GraphData->m_Operations.Begin(), m_GraphData->m_Operations.Size());
    }

    file->SaveObjects(m_InputArray.Begin(), m_InputArray.Size());
    file->SaveObjects(m_OutputArray.Begin(), m_OutputArray.Size());
    file->SaveObjects(m_InParameter.Begin(), m_InParameter.Size());
    file->SaveObjects(m_OutParameter.Begin(), m_OutParameter.Size());
    file->SaveObjects(m_LocalParameter.Begin(), m_LocalParameter.Size());

    CKParameterIn *targetObj = GetTargetParameter();
    file->SaveObject(targetObj);
}

CKStateChunk *CKBehavior::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = CKObject::Save(file, flags);
    if (!file && !(flags & CK_STATESAVE_BEHAVIORONLY))
        return baseChunk;

    CKStateChunk *behaviorChunk = CreateCKStateChunk(CKCID_BEHAVIOR, file);
    behaviorChunk->StartWrite();
    behaviorChunk->AddChunkAndDelete(baseChunk);

    if (file) {
        if (m_Context->m_InterfaceMode && !(m_Context->GetFileWriteMode() & CKFILE_FORVIEWER)) {
            behaviorChunk->WriteIdentifier(CK_STATESAVE_BEHAVIORINTERFACE);
            behaviorChunk->WriteSubChunk(m_InterfaceChunk);
        }

        behaviorChunk->WriteIdentifier(CK_STATESAVE_BEHAVIORNEWDATA);
        CKDWORD behaviorFlags = m_Flags;

        if (!(behaviorFlags & CKBEHAVIOR_BUILDINGBLOCK))
            behaviorFlags &= ~CKBEHAVIOR_LOCKED;

        if (m_Priority != 0)
            behaviorFlags |= CKBEHAVIOR_PRIORITY;
        if (m_CompatibleClassID != CKCID_BEOBJECT)
            behaviorFlags |= CKBEHAVIOR_COMPATIBLECLASSID;

        behaviorFlags &= ~CKBEHAVIOR_LAUNCHEDONCE;
        behaviorChunk->WriteDword(behaviorFlags);

        if (behaviorFlags & CKBEHAVIOR_BUILDINGBLOCK) {
            behaviorChunk->WriteGuid(m_BlockData->m_Guid);
            behaviorChunk->WriteDword(m_BlockData->m_Version);
        }

        if (behaviorFlags & CKBEHAVIOR_PRIORITY)
            behaviorChunk->WriteInt(m_Priority);

        if (behaviorFlags & CKBEHAVIOR_COMPATIBLECLASSID)
            behaviorChunk->WriteInt(m_CompatibleClassID);

        if (behaviorFlags & CKBEHAVIOR_TARGETABLE) {
            behaviorChunk->WriteObject(GetTargetParameter());
        }

        CKDWORD saveFlags = 0;
        if (m_GraphData) {
            if (m_GraphData->m_SubBehaviors.Size() > 0)
                saveFlags |= CK_STATESAVE_BEHAVIORSUBBEHAV;
            if (m_GraphData->m_SubBehaviorLinks.Size() > 0)
                saveFlags |= CK_STATESAVE_BEHAVIORSUBLINKS;
            if (m_GraphData->m_Operations.Size() > 0)
                saveFlags |= CK_STATESAVE_BEHAVIOROPERATIONS;
        }
        if (m_InParameter.Size())
            saveFlags |= CK_STATESAVE_BEHAVIORINPARAMS;
        if (m_OutParameter.Size())
            saveFlags |= CK_STATESAVE_BEHAVIOROUTPARAMS;
        if (m_LocalParameter.Size())
            saveFlags |= CK_STATESAVE_BEHAVIORLOCALPARAMS;
        if (m_InputArray.Size())
            saveFlags |= CK_STATESAVE_BEHAVIORINPUTS;
        if (m_OutputArray.Size())
            saveFlags |= CK_STATESAVE_BEHAVIOROUTPUTS;

        behaviorChunk->WriteDword(saveFlags);

        if (m_GraphData) {
            if (saveFlags & CK_STATESAVE_BEHAVIORSUBBEHAV)
                m_GraphData->m_SubBehaviors.Save(behaviorChunk);

            if (saveFlags & CK_STATESAVE_BEHAVIORSUBLINKS)
                m_GraphData->m_SubBehaviorLinks.Save(behaviorChunk);

            if (saveFlags & CK_STATESAVE_BEHAVIOROPERATIONS)
                m_GraphData->m_Operations.Save(behaviorChunk);
        }

        if (saveFlags & CK_STATESAVE_BEHAVIORINPARAMS)
            m_InParameter.Save(behaviorChunk);
        if (saveFlags & CK_STATESAVE_BEHAVIOROUTPARAMS)
            m_OutParameter.Save(behaviorChunk);
        if (saveFlags & CK_STATESAVE_BEHAVIORLOCALPARAMS)
            m_LocalParameter.Save(behaviorChunk);
        if (saveFlags & CK_STATESAVE_BEHAVIORINPUTS)
            m_InputArray.Save(behaviorChunk);
        if (saveFlags & CK_STATESAVE_BEHAVIOROUTPUTS)
            m_OutputArray.Save(behaviorChunk);
    } else {
        if ((flags & CK_STATESAVE_BEHAVIORSUBBEHAV) != 0 && m_GraphData) {
            if (m_GraphData->m_SubBehaviors.Size()) {
                behaviorChunk->WriteIdentifier(CK_STATESAVE_BEHAVIORSUBBEHAV);
                behaviorChunk->WriteInt(m_GraphData->m_SubBehaviors.Size());

                for (XObjectPointerArray::Iterator it = m_GraphData->m_SubBehaviors.Begin();
                     it != m_GraphData->m_SubBehaviors.End(); ++it) {
                    CKObject *subBehavior = *it;
                    CKStateChunk *subChunk = subBehavior ? subBehavior->Save(nullptr, flags) : nullptr;

                    behaviorChunk->WriteObject(subBehavior);
                    behaviorChunk->WriteSubChunk(subChunk);

                    if (subChunk) {
                        delete subChunk;
                    }
                }
            }
        }

        if ((m_Flags & CKBEHAVIOR_BUILDINGBLOCK) != 0 || !(flags & CK_STATESAVE_BEHAVIORLOCALPARAMS)) {
            if (GetClassID() == CKCID_BEHAVIOR) {
                behaviorChunk->CloseChunk();
            } else {
                behaviorChunk->UpdateDataSize();
            }

            return behaviorChunk;
        }

        behaviorChunk->WriteIdentifier(CK_STATESAVE_BEHAVIORLOCALPARAMS);
        if (m_LocalParameter.Size() > 0) {
            behaviorChunk->WriteInt(m_LocalParameter.Size());

            for (XObjectPointerArray::Iterator it = m_LocalParameter.Begin(); it != m_LocalParameter.End(); ++it) {
                CKObject *param = *it;
                CKDWORD paramFlags = (flags == (CK_STATESAVE_BEHAVIORSUBBEHAV | CK_STATESAVE_BEHAVIORLOCALPARAMS))
                                         ? CK_STATESAVE_BEHAVIORFLAGS
                                         : -1;
                CKStateChunk *paramChunk = param ? param->Save(nullptr, paramFlags) : nullptr;

                behaviorChunk->WriteObject(param);
                behaviorChunk->WriteSubChunk(paramChunk);

                if (paramChunk) {
                    delete paramChunk;
                }
            }

            if (GetClassID() == CKCID_BEHAVIOR) {
                behaviorChunk->CloseChunk();
            } else {
                behaviorChunk->UpdateDataSize();
            }

            return behaviorChunk;
        }


        behaviorChunk->WriteDword(0);
    }

    if (file && !file->m_SceneSaved) {
        CKScene *scene = m_Context->GetCurrentScene();
        if (scene) {
            CKSceneObjectDesc *desc = scene->GetSceneObjectDesc(this);
            if (desc) {
                CKDWORD sceneFlags = desc->m_Global;
                if (desc->m_InitialValue)
                    sceneFlags |= CK_SCENEOBJECT_INTERNAL_IC;

                behaviorChunk->WriteIdentifier(CK_STATESAVE_BEHAVIORSINGLEACTIVITY);
                behaviorChunk->WriteDword(sceneFlags);
            }
        }
    }

    if (GetClassID() == CKCID_BEHAVIOR) {
        behaviorChunk->CloseChunk();
    } else {
        behaviorChunk->UpdateDataSize();
    }

    return behaviorChunk;
}

CKERROR CKBehavior::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Load base object data
    CKObject::Load(chunk, file);

    if (!file) {
        // Load sub-behaviors when not in file context
        if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORSUBBEHAV)) {
            int subBehaviorCount = chunk->ReadInt();
            for (int i = 0; i < subBehaviorCount; ++i) {
                CK_ID objId = chunk->ReadObjectID();
                CKObject *obj = m_Context->GetObject(objId);
                CKStateChunk *subChunk = chunk->ReadSubChunk();

                if (subChunk) {
                    if (obj) {
                        obj->Load(subChunk, nullptr);
                    }
                    delete subChunk;
                }
            }
        }

        // Load local parameters
        if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORLOCALPARAMS)) {
            int paramCount = chunk->ReadInt();
            for (int i = 0; i < paramCount; ++i) {
                CK_ID paramId = chunk->ReadObjectID();
                CKObject *param = m_Context->GetObject(paramId);
                CKStateChunk *paramChunk = chunk->ReadSubChunk();

                if (paramChunk) {
                    if (param) {
                        param->Load(paramChunk, nullptr);
                    }
                    delete paramChunk;
                }
            }
        }

        CallCallbackFunction(CKM_BEHAVIORREADSTATE);
    } else {
        // Initialize behavior state
        m_Flags = 0;
        m_CompatibleClassID = CKCID_BEOBJECT;
        m_Priority = 0;
        m_InputTargetParam = 0;

        if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORNEWDATA)) {
            if (chunk->GetDataVersion() >= 5) {
                CKDWORD flags = chunk->ReadDword();
                m_Flags = flags & ~(CKBEHAVIOR_ACTIVE |
                    CKBEHAVIOR_PRIORITY |
                    CKBEHAVIOR_COMPATIBLECLASSID |
                    CKBEHAVIOR_EXECUTEDLASTFRAME |
                    CKBEHAVIOR_DEACTIVATENEXTFRAME |
                    CKBEHAVIOR_RESETNEXTFRAME |
                    CKBEHAVIOR_ACTIVATENEXTFRAME);

                if (!(flags & CKBEHAVIOR_BUILDINGBLOCK)) {
                    UseGraph();
                } else {
                    UseFunction();
                    CKGUID blockGuid = chunk->ReadGuid();
                    m_BlockData->m_Guid = blockGuid;
                    m_BlockData->m_Version = chunk->ReadDword();
                    InitFctPtrFromGuid(blockGuid);
                }

                if (flags & CKBEHAVIOR_PRIORITY)
                    m_Priority = chunk->ReadInt();
                else
                    m_Priority = 0;

                if (flags & CKBEHAVIOR_COMPATIBLECLASSID)
                    m_CompatibleClassID = chunk->ReadDword();
                else
                    m_CompatibleClassID = CKCID_BEOBJECT;

                if (flags & CKBEHAVIOR_TARGETABLE)
                    m_InputTargetParam = chunk->ReadObjectID();
                else
                    m_InputTargetParam = 0;

                CKDWORD saveFlags = chunk->ReadDword();
                if (m_GraphData) {
                    if (saveFlags & CK_STATESAVE_BEHAVIORSUBBEHAV)
                        m_GraphData->m_SubBehaviors.Load(m_Context, chunk);
                    if (saveFlags & CK_STATESAVE_BEHAVIORSUBLINKS)
                        m_GraphData->m_SubBehaviorLinks.Load(m_Context, chunk);
                    if (saveFlags & CK_STATESAVE_BEHAVIOROPERATIONS)
                        m_GraphData->m_Operations.Load(m_Context, chunk);
                }

                if (saveFlags & CK_STATESAVE_BEHAVIORINPARAMS)
                    m_InParameter.Load(m_Context, chunk);
                if (saveFlags & CK_STATESAVE_BEHAVIOROUTPARAMS)
                    m_OutParameter.Load(m_Context, chunk);
                if (saveFlags & CK_STATESAVE_BEHAVIORLOCALPARAMS)
                    m_LocalParameter.Load(m_Context, chunk);
                if (saveFlags & CK_STATESAVE_BEHAVIORINPUTS)
                    m_InputArray.Load(m_Context, chunk);
                if (saveFlags & CK_STATESAVE_BEHAVIOROUTPUTS)
                    m_OutputArray.Load(m_Context, chunk);
            } else {
                CKGUID guid = chunk->ReadGuid();
                CKDWORD flags = chunk->ReadDword();
                m_Flags = flags;

                if (!(flags & CKBEHAVIOR_BUILDINGBLOCK)) {
                    UseGraph();
                } else {
                    UseFunction();
                    m_BlockData->m_Guid = guid;
                    InitFctPtrFromGuid(guid);
                }

                m_Flags &= ~(CKBEHAVIOR_ACTIVATENEXTFRAME | CKBEHAVIOR_RESETNEXTFRAME | CKBEHAVIOR_DEACTIVATENEXTFRAME | CKBEHAVIOR_EXECUTEDLASTFRAME);
                m_CompatibleClassID = chunk->ReadDword();
                CK_BEHAVIOR_TYPE type = (CK_BEHAVIOR_TYPE)chunk->ReadDword();
                SetType(type);
                m_Priority = chunk->ReadInt();
                m_Owner = chunk->ReadObjectID();

                if (!m_BlockData) {
                    chunk->ReadDword();
                } else {
                    m_BlockData->m_Version = chunk->ReadDword();
                    if (m_BlockData->m_Version == 0)
                        m_BlockData->m_Version = 0x10000;
                }
            }
        } else {
            CKGUID guid;
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORPROTOGUID)) {
                guid = chunk->ReadGuid();
            }
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORFLAGS)) {
                m_Flags |= chunk->ReadInt();
                if (m_Flags & CKBEHAVIOR_USEFUNCTION) {
                    UseFunction();
                    m_BlockData->m_Guid = guid;
                    InitFctPtrFromGuid(guid);
                } else {
                    UseGraph();
                }
            }
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORCOMPATIBLECID))
                m_CompatibleClassID = chunk->ReadDword();
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORTYPE)) {
                CK_BEHAVIOR_TYPE type = (CK_BEHAVIOR_TYPE)chunk->ReadDword();
                SetType(type);
            }
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIOROWNER))
                m_Owner = chunk->ReadObjectID();
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORPRIORITY))
                m_Priority = chunk->ReadInt();
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORTARGET))
                m_InputTargetParam = chunk->ReadObjectID();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORINTERFACE) && m_Context->m_InterfaceMode) {
            if (m_InterfaceChunk) {
                delete m_InterfaceChunk;
            }
            m_InterfaceChunk = chunk->ReadSubChunk();
        }

        if (chunk->GetDataVersion() < 5) {
            if (m_GraphData) {
                if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORSUBBEHAV))
                    m_GraphData->m_SubBehaviors.Load(m_Context, chunk);
                if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORSUBLINKS))
                    m_GraphData->m_SubBehaviorLinks.Load(m_Context, chunk);
                if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIOROPERATIONS))
                    m_GraphData->m_Operations.Load(m_Context, chunk);
            }
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORINPARAMS))
                m_InParameter.Load(m_Context, chunk);
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORLOCALPARAMS))
                m_LocalParameter.Load(m_Context, chunk);
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIOROUTPARAMS))
                m_OutParameter.Load(m_Context, chunk);
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORINPUTS))
                m_InputArray.Load(m_Context, chunk);
            if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIOROUTPUTS))
                m_OutputArray.Load(m_Context, chunk);
        }

        CKBehaviorPrototype *proto = GetPrototype();
        if (proto && proto->IsRunTime())
            m_Context->m_RunTime = TRUE;

        m_Flags &= ~CKBEHAVIOR_LAUNCHEDONCE;
    }

    if (chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORSINGLEACTIVITY)) {
        CK_ID id = chunk->ReadDword();
        m_Context->m_ObjectManager->AddSingleObjectActivity(this, id);
    }
    return CK_OK;
}

void CKBehavior::PostLoad() {
    if ((m_Flags & (CKBEHAVIOR_SCRIPT | CKBEHAVIOR_TOPMOST)) != 0)
        HierarchyPostLoad();
    CKObject::PostLoad();
}

void CKBehavior::PreDelete() {
    if (m_Flags & CKBEHAVIOR_SCRIPT) {
        CKBeObject *owner = GetOwner();
        if (owner && !owner->IsToBeDeleted()) {
            owner->m_ScriptArray->Remove(this);
        }
    } else {
        CKBehavior *parent = GetParent();
        if (parent && !parent->IsToBeDeleted()) {
            if (parent->m_GraphData)
                parent->m_GraphData->m_SubBehaviors.Remove(this);
        }
    }
}

int CKBehavior::GetMemoryOccupation() {
    int size = CKSceneObject::GetMemoryOccupation() + (int) (sizeof(CKBehavior) - sizeof(CKSceneObject));

    size += m_InputArray.GetMemoryOccupation(FALSE);
    size += m_OutputArray.GetMemoryOccupation(FALSE);
    size += m_InParameter.GetMemoryOccupation(FALSE);
    size += m_OutParameter.GetMemoryOccupation(FALSE);
    size += m_LocalParameter.GetMemoryOccupation(FALSE);

    if (m_GraphData) {
        size += (int) sizeof(*m_GraphData);
        size += m_GraphData->m_Operations.GetMemoryOccupation(FALSE);
        size += m_GraphData->m_SubBehaviors.GetMemoryOccupation(FALSE);
        size += m_GraphData->m_SubBehaviorLinks.GetMemoryOccupation(FALSE);
        size += m_GraphData->m_Links.GetMemoryOccupation(FALSE);
        if (m_GraphData->m_BehaviorIterators) {
            size += (int) (m_GraphData->m_BehaviorIteratorCount * sizeof(*m_GraphData->m_BehaviorIterators));
        }
    }

    if (m_BlockData) {
        size += (int) sizeof(*m_BlockData);
    }

    return size;
}

CKBOOL CKBehavior::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    switch (cid) {
    case CKCID_PARAMETERIN:
        if (m_InParameter.IsHere(obj) || obj->GetID() == m_InputTargetParam)
            return TRUE;
        break;
    case CKCID_PARAMETEROUT:
        if (m_OutParameter.IsHere(obj))
            return TRUE;
        break;
    case CKCID_PARAMETEROPERATION:
        if (m_GraphData && m_GraphData->m_Operations.IsHere(obj))
            return TRUE;
        break;
    case CKCID_BEHAVIORLINK:
        if (m_GraphData) {
            if (m_GraphData->m_SubBehaviorLinks.IsHere(obj))
                return TRUE;
            if (m_GraphData->m_Links.IsHere(obj))
                return TRUE;
        }
        break;
    case CKCID_BEHAVIOR:
        if (m_GraphData && m_GraphData->m_SubBehaviors.IsHere(obj))
            return TRUE;
        break;
    case CKCID_BEHAVIORIO:
        if (m_InputArray.IsHere(obj) || m_OutputArray.IsHere(obj))
            return TRUE;
        break;
    case CKCID_PARAMETERLOCAL:
        if (m_LocalParameter.IsHere(obj))
            return TRUE;
        break;
    default:
        break;
    }
    return CKObject::IsObjectUsed(obj, cid);
}

CKERROR CKBehavior::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    // Prepare owner dependencies
    CKBeObject *owner = GetOwner();
    if (owner) {
        if (context.IsInMode(CK_DEPENDENCIES_BUILD)) {
            owner->PrepareDependencies(context);
        }
        if (context.IsInMode(CK_DEPENDENCIES_DELETE)) {
            CallCallbackFunction(CKM_BEHAVIORDETACH);
        }
    }

    // Add class dependencies
    context.GetClassDependencies(m_ClassID);

    // Prepare target parameter dependencies
    CKParameterIn *targetParam = GetTargetParameter();
    if (targetParam) {
        targetParam->PrepareDependencies(context);
    }

    // Prepare array dependencies
    m_InputArray.Prepare(context);
    m_OutputArray.Prepare(context);

    if (m_GraphData) {
        m_GraphData->m_SubBehaviorLinks.Prepare(context);
        m_GraphData->m_SubBehaviors.Prepare(context);
        m_GraphData->m_Operations.Prepare(context);
    }

    m_InParameter.Prepare(context);
    m_OutParameter.Prepare(context);
    m_LocalParameter.Prepare(context);

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKBehavior::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    // Remap owner reference
    CK_ID oldOwner = m_Owner;
    CK_ID remappedOwner = context.RemapID(m_Owner);
    if (remappedOwner == oldOwner) {
        m_Owner = 0;
        RemoveFromAllScenes();
    } else {
        m_Owner = remappedOwner;
    }

    // Remap other object references
    m_BehParent = context.RemapID(m_BehParent);
    m_InputTargetParam = context.RemapID(m_InputTargetParam);

    // Remap object arrays
    m_InputArray.Remap(context);
    m_OutputArray.Remap(context);
    m_InParameter.Remap(context);
    m_OutParameter.Remap(context);
    m_LocalParameter.Remap(context);

    // Update local parameter owners
    for (auto it = m_LocalParameter.Begin(); it != m_LocalParameter.End(); ++it) {
        CKParameterLocal *param = (CKParameterLocal *) *it;
        if (param) {
            param->SetOwner(this);
        }
    }

    // Remap graph data if present
    if (m_GraphData) {
        m_GraphData->m_SubBehaviorLinks.Remap(context);
        m_GraphData->m_SubBehaviors.Remap(context);
        m_GraphData->m_Operations.Remap(context);
        m_GraphData->m_Links.Remap(context);
    }

    // Remap interface chunk if needed
    if (m_Context->m_InterfaceMode && m_InterfaceChunk) {
        m_InterfaceChunk->RemapObjects(m_Context, &context);
    }

    // Handle behavior attachment callback
    if ((m_Flags & CKBEHAVIOR_BUILDINGBLOCK) != 0 && m_Owner) {
        CallCallbackFunction(CKM_BEHAVIORATTACH);
    }

    return CK_OK;
}

CKERROR CKBehavior::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    CKBehavior *src = (CKBehavior *) &o;
    if (!src)
        return CKERR_INVALIDPARAMETER;

    // Copy simple members
    m_InputTargetParam = src->m_InputTargetParam;
    m_Owner = src->m_Owner;
    m_BehParent = src->m_BehParent;
    m_CompatibleClassID = src->m_CompatibleClassID;
    m_Priority = src->m_Priority;
    m_Flags = src->m_Flags & ~(CKBEHAVIOR_RESERVED0 | CKBEHAVIOR_EXECUTEDLASTFRAME);

    // Handle BehaviorBlockData
    delete m_BlockData;
    m_BlockData = nullptr;
    if (src->m_BlockData) {
        m_BlockData = new BehaviorBlockData(*src->m_BlockData);
    }

    // Handle BehaviorGraphData
    if (m_GraphData) {
        delete m_GraphData;
        m_GraphData = nullptr;
    }
    if (src->m_GraphData) {
        m_GraphData = new BehaviorGraphData(*src->m_GraphData);
    }

    // Copy object arrays using assignment operators
    m_InputArray = src->m_InputArray;
    m_OutputArray = src->m_OutputArray;
    m_InParameter = src->m_InParameter;
    m_OutParameter = src->m_OutParameter;
    m_LocalParameter = src->m_LocalParameter;

    // Handle interface chunk
    if (src->m_InterfaceChunk) {
        m_InterfaceChunk = new CKStateChunk(*src->m_InterfaceChunk);
    }

    return CK_OK;
}

CKSTRING CKBehavior::GetClassName() {
    return "Behavior";
}

int CKBehavior::GetDependenciesCount(int mode) {
    return CKObject::GetDependenciesCount(mode);
}

CKSTRING CKBehavior::GetDependencies(int i, int mode) {
    return CKObject::GetDependencies(i, mode);
}

void CKBehavior::Register() {
    CKPARAMETERFROMCLASS(CKBehavior, CKPGUID_BEHAVIOR);
    CKCLASSDEFAULTOPTIONS(CKBehavior, CK_DEPENDENCIES_COPY);
}

CKBehavior *CKBehavior::CreateInstance(CKContext *Context) {
    return new CKBehavior(Context);
}

void CKBehavior::Reset() {
    if (m_GraphData) {
        // Reset activation delays for sub-behavior links
        for (XObjectPointerArray::Iterator it = m_GraphData->m_SubBehaviorLinks.Begin();
             it != m_GraphData->m_SubBehaviorLinks.End(); ++it) {
            CKBehaviorLink *link = (CKBehaviorLink *) *it;
            if (link) {
                link->m_ActivationDelay = link->m_InitialActivationDelay;
            }
        }

        for (XObjectPointerArray::Iterator it = m_GraphData->m_Links.Begin();
             it != m_GraphData->m_Links.End(); ++it) {
            CKBehaviorLink *link = (CKBehaviorLink *) *it;
            if (link) {
                link->SetOldFlags(link->GetOldFlags() & ~CKBL_OLD_IN_DELAYED_LIST);
            }
        }
        m_GraphData->m_Links.Resize(0);

        // Reset sub-behaviors
        for (XObjectPointerArray::Iterator it = m_GraphData->m_SubBehaviors.Begin();
             it != m_GraphData->m_SubBehaviors.End(); ++it) {
            CKBehavior *subBeh = (CKBehavior *) *it;
            if (subBeh) {
                subBeh->m_Flags &= ~CKBEHAVIOR_ACTIVE; // Deactivate
                subBeh->Reset();                       // Recursive call
            }
        }
    }

    // Clear activation flags for inputs
    for (XObjectPointerArray::Iterator it = m_InputArray.Begin(); it != m_InputArray.End(); ++it) {
        CKBehaviorIO *io = (CKBehaviorIO *) *it;
        if (io) {
            io->Activate(FALSE);
        }
    }

    // Clear activation flags for outputs
    for (XObjectPointerArray::Iterator it = m_OutputArray.Begin(); it != m_OutputArray.End(); ++it) {
        CKBehaviorIO *io = (CKBehaviorIO *) *it;
        if (io) {
            io->Activate(FALSE);
        }
    }

    ModifyFlags(CKBEHAVIOR_LAUNCHEDONCE,
                CKBEHAVIOR_WAITSFORMESSAGE |
                CKBEHAVIOR_RESETNEXTFRAME |
                CKBEHAVIOR_DEACTIVATENEXTFRAME |
                CKBEHAVIOR_ACTIVATENEXTFRAME);
    if ((m_Flags & CKBEHAVIOR_SCRIPT) != 0)
        ActivateInput(0, TRUE);
}

void CKBehavior::ErrorMessage(CKSTRING Error, CKSTRING Context, CKBOOL ShowOwner, CKBOOL ShowScript) {
    CKBehavior *topBehavior = nullptr;
    if (ShowScript) {
        CKBehavior *parent = GetParent();
        topBehavior = this;
        while (parent) {
            topBehavior = parent;
            parent = topBehavior->GetParent();
        }
    }

    char buffer[1024]; // Assume reasonable buffer size
    const char *behaviorName = m_Name ? m_Name : "Unnamed Behavior";

    strcpy(buffer, behaviorName);
    strcat(buffer, "=>");
    strcat(buffer, Error);

    if (Context) {
        strcat(buffer, "(");
        strcat(buffer, Context);
        strcat(buffer, ")");
    }

    if (ShowScript) {
        strcat(buffer, "\n");
        if (topBehavior) {
            if (topBehavior->m_Name) {
                strcat(buffer, "Script:");
                strcat(buffer, topBehavior->m_Name);
            } else {
                strcat(buffer, "Unnamed Script");
            }
        } else {
            strcat(buffer, "Not within a script");
        }
    }

    if (ShowOwner) {
        strcat(buffer, "\n");
        CKBehavior *targetBehavior = topBehavior ? topBehavior : this;
        CKBeObject *owner = targetBehavior->GetOwner();

        if (owner) {
            if (owner->m_Name) {
                strcat(buffer, "Owner:");
                strcat(buffer, owner->m_Name);
            } else {
                strcat(buffer, "Unnamed Owner");
            }
        } else {
            strcat(buffer, "Not applied to any object");
        }
    }

    m_Context->OutputToConsole(buffer, TRUE);
}

void CKBehavior::ErrorMessage(CKSTRING Error, CKDWORD Context, CKBOOL ShowOwner, CKBOOL ShowScript) {
    const char *contextStr = nullptr;
    switch (Context) {
    case CKM_BEHAVIORPRESAVE:
        contextStr = "Pre Save";
        break;
    case CKM_BEHAVIORATTACH:
        contextStr = "Attach";
        break;
    case CKM_BEHAVIORDETACH:
        contextStr = "Detach";
        break;
    case CKM_BEHAVIORPAUSE:
        contextStr = "Pause";
        break;
    case CKM_BEHAVIORRESUME:
        contextStr = "Resume";
        break;
    case CKM_BEHAVIORRESET:
        contextStr = "Reset";
        break;
    case CKM_BEHAVIORPOSTSAVE:
        contextStr = "Post Save";
        break;
    case CKM_BEHAVIORLOAD:
        contextStr = "Load";
        break;
    case CKM_BEHAVIOREDITED:
        contextStr = "Parameters Edition";
        break;
    case CKM_BEHAVIORSETTINGSEDITED:
        contextStr = "Settings Edition";
        break;
    case CKM_BEHAVIORREADSTATE:
        contextStr = "Read State";
        break;
    case CKM_BEHAVIORNEWSCENE:
        contextStr = "New Scene";
        break;
    case CKM_BEHAVIORACTIVATESCRIPT:
        contextStr = "Activate Script";
        break;
    case CKM_BEHAVIORDEACTIVATESCRIPT:
        contextStr = "Deactivate Script";
        break;
    default:
        contextStr = "Execution";
        break;
    }
    return ErrorMessage(Error, (CKSTRING) contextStr, ShowOwner, ShowScript);
}

void CKBehavior::SetPrototypeGuid(CKGUID ckguid) {
    if (m_BlockData)
        m_BlockData->m_Guid = ckguid;
}

void CKBehavior::SetParent(CKBehavior *parent) {
    if (parent) {
        m_BehParent = parent->GetID();
    } else {
        m_BehParent = 0;
    }
}

void CKBehavior::SortSubs() {
    if (m_GraphData)
        m_GraphData->m_SubBehaviors.Sort(BehaviorPrioritySort);
}

void CKBehavior::ResetExecutionTime() {
    m_Flags &= ~CKBEHAVIOR_EXECUTEDLASTFRAME;
    m_LastExecutionTime = 0.0f;

    if (m_GraphData) {
        for (XObjectPointerArray::Iterator it = m_GraphData->m_SubBehaviors.Begin();
             it != m_GraphData->m_SubBehaviors.End(); ++it) {
            CKBehavior *subBeh = (CKBehavior *) *it;
            if (subBeh) {
                subBeh->ResetExecutionTime();
            }
        }
    }
}

void CKBehavior::ExecuteStepStart() {
    if (!(m_Flags & CKBEHAVIOR_EXECUTEDLASTFRAME)) {
        CKBehaviorManager *behManager = m_Context->GetBehaviorManager();
        m_Flags |= CKBEHAVIOR_EXECUTEDLASTFRAME;
        behManager->m_Behaviors.PushBack(this);
    }

    ++m_Context->m_Stats.BehaviorsExecuted;

    VxTimeProfiler profiler;

    if (!IsUsingFunction()) {
        CheckIOsActivation();
    }

    if (m_Context->m_ProfilingEnabled) {
        m_LastExecutionTime += profiler.Current();
    }
}

int CKBehavior::ExecuteStep(float delta, CKDebugContext *Context) {
    VxTimeProfiler profiler;

    if (m_Flags & CKBEHAVIOR_USEFUNCTION) {
        int result = ExecuteFunction();
        Context->CurrentBehaviorAction = CKDEBUG_BEHEXECUTEDONE;
        return result;
    }

    BehaviorGraphData *graphData = m_GraphData;
    if (graphData->m_BehaviorIteratorIndex <= 0) {
        CheckBehaviorActivity();
        Context->CurrentBehaviorAction = CKDEBUG_BEHEXECUTEDONE;
        return CKBR_OK;
    }

    if (Context->CurrentBehaviorAction == CKDEBUG_BEHEXECUTE) {
        CKBehavior *currentBeh = graphData->m_BehaviorIterators[graphData->m_BehaviorIteratorIndex - 1];
        currentBeh->m_Flags &= ~CKBEHAVIOR_RESERVED0;
        Context->StepInto(currentBeh);
    } else {
        CKBehavior *subBehavior = Context->SubBehavior;
        --graphData->m_BehaviorIteratorIndex;
        Context->CurrentBehaviorAction = CKDEBUG_BEHEXECUTE;
        FindNextBehaviorsToExecute(subBehavior);
    }

    return CKBR_OK;
}

void CKBehavior::WarnInfiniteLoop() {
    char buffer[512];

    CKBeObject *owner = GetOwner();
    const char *ownerName = (owner && owner->m_Name) ? owner->m_Name : "?";
    const char *behaviorName = m_Name ? m_Name : "?";

    snprintf(buffer, sizeof(buffer), "ERROR : Infinite Loop Detected. Object %s : In Behavior %s:", ownerName, behaviorName);

    m_Context->OutputToConsole(buffer, true);
}

int CKBehavior::InternalGetShortestDelay(CKBehavior *beh, XObjectPointerArray &behparsed) {
    if (this == beh)
        return 0;

    // Prevent infinite recursion
    if (!behparsed.AddIfNotHere(this))
        return 1000; // Loop detected

    int shortestDelay = 1000;
    const int outputCount = m_OutputArray.Size();
    for (int i = 0; i < outputCount; ++i) {
        CKBehaviorIO *output = (CKBehaviorIO *) m_OutputArray[i];
        if (!output)
            continue;

        const int linkCount = output->m_Links.Size();
        for (int j = 0; j < linkCount; ++j) {
            CKBehaviorLink *link = (CKBehaviorLink *) output->m_Links[j];
            if (!link || !link->m_OutIO)
                continue;

            CKBehaviorIO *nextIO = link->m_OutIO;
            if (nextIO->m_ObjectFlags & CK_BEHAVIORIO_OUT)
                continue;

            CKBehavior *nextBehavior = nextIO->m_OwnerBehavior;
            if (!nextBehavior)
                continue;

            int delay = link->m_InitialActivationDelay +
                nextBehavior->InternalGetShortestDelay(beh, behparsed);

            if (delay < shortestDelay)
                shortestDelay = delay;
        }
    }

    return shortestDelay;
}

void CKBehavior::CheckIOsActivation() {
    if (!m_GraphData)
        return;

    int deactivateCount = 0;
    int activateCount = 0;

    const int linkCount = m_GraphData->m_SubBehaviorLinks.Size();
    if (linkCount > 0) {
        EnsureIoBuffers(linkCount);

        for (int idx = 0; idx < linkCount; ++idx) {
            ++m_Context->m_Stats.BehaviorLinksParsed;

            CKBehaviorLink *link = (CKBehaviorLink *)m_GraphData->m_SubBehaviorLinks[idx];
            CKBehaviorIO *inIO = link ? link->GetInBehaviorIO() : nullptr;
            if (link)
                link->SetOldFlags(link->GetOldFlags() & ~CKBL_OLD_TRIGGERED_THIS_FRAME);

            if (inIO && inIO->IsActive()) {
                g_IosToDeactivate[deactivateCount++] = inIO;

                const int initialDelay = link->GetInitialActivationDelay();
                link->SetOldFlags(link->GetOldFlags() | CKBL_OLD_TRIGGERED_THIS_FRAME);

                if (initialDelay) {
                    if (link->GetActivationDelay() == 0)
                        link->SetActivationDelay(initialDelay);

                    if ((link->GetOldFlags() & CKBL_OLD_IN_DELAYED_LIST) == 0) {
                        link->SetOldFlags(link->GetOldFlags() | CKBL_OLD_IN_DELAYED_LIST);
                        m_GraphData->m_Links.PushBack(link);
                        ++m_Context->m_Stats.BehaviorDelayedLinks;
                    }
                } else {
                    g_IosToActivate[activateCount++] = link->GetOutBehaviorIO();
                }
            }
        }
    }

    for (int i = 0; i < deactivateCount; ++i) {
        CKBehaviorIO *io = g_IosToDeactivate[i];
        if (io)
            io->Activate(FALSE);
    }

    for (int i = 0; i < activateCount; ++i) {
        CKBehaviorIO *io = g_IosToActivate[i];
        if (!io)
            continue;

        io->Activate(TRUE);
        if (io->GetType() == CK_BEHAVIORIO_IN) {
            CKBehavior *owner = io->GetOwner();
            if (owner)
                owner->ModifyFlags(CKBEHAVIOR_ACTIVE, 0);
        }
    }

    m_GraphData->m_BehaviorIteratorIndex = 0;
    const int subBehaviorCount = m_GraphData->m_SubBehaviors.Size();
    if (subBehaviorCount > m_GraphData->m_BehaviorIteratorCount) {
        delete[] m_GraphData->m_BehaviorIterators;
        m_GraphData->m_BehaviorIterators = new CKBehavior *[subBehaviorCount];
        m_GraphData->m_BehaviorIteratorCount = (CKWORD)subBehaviorCount;
    }

    for (int k = subBehaviorCount - 1; k >= 0; --k) {
        CKBehavior *subBeh = (CKBehavior *)m_GraphData->m_SubBehaviors[k];
        if (subBeh != this) {
            if (subBeh && subBeh->IsActive()) {
                subBeh->ModifyFlags(CKBEHAVIOR_RESERVED0, 0);
                m_GraphData->m_BehaviorIterators[m_GraphData->m_BehaviorIteratorIndex++] = subBeh;
            }
        }
    }
}

void CKBehavior::CheckBehaviorActivity() {
    XObjectPointerArray remaining;

    for (XObjectPointerArray::Iterator it = m_GraphData->m_Links.Begin(); it != m_GraphData->m_Links.End(); ++it) {
        CKBehaviorLink *link = (CKBehaviorLink *)*it;
        if (!link)
            continue;

        link->SetOldFlags(link->GetOldFlags() | CKBL_OLD_TRIGGERED_THIS_FRAME);
        link->SetActivationDelay(link->GetActivationDelay() - 1);

        if (link->GetActivationDelay() > 0) {
            remaining.PushBack(link);
        } else {
            link->SetActivationDelay(0);
            link->SetOldFlags(link->GetOldFlags() & ~CKBL_OLD_IN_DELAYED_LIST);

            CKBehaviorIO *outIO = link->GetOutBehaviorIO();
            if (outIO) {
                CKBehavior *owner = outIO->GetOwner();
                if (owner)
                    owner->ModifyFlags(CKBEHAVIOR_ACTIVE, 0);
                outIO->Activate(TRUE);
            }
        }
    }

    if (!remaining.IsEmpty()) {
        m_GraphData->m_Links.Swap(remaining);
    } else {
        m_GraphData->m_Links.Resize(0);
    }

    ModifyFlags(0, CKBEHAVIOR_ACTIVE);

    if (!m_GraphData->m_Links.IsEmpty()) {
        m_Flags |= CKBEHAVIOR_ACTIVE;
        return;
    }

    for (XObjectPointerArray::Iterator it = m_GraphData->m_SubBehaviors.Begin(); it != m_GraphData->m_SubBehaviors.End(); ++it) {
        CKBehavior *sub = (CKBehavior *)*it;
        if (sub && (sub->GetFlags() & (CKBEHAVIOR_ACTIVE | CKBEHAVIOR_WAITSFORMESSAGE)) != 0) {
            ModifyFlags(CKBEHAVIOR_ACTIVE, 0);
            return;
        }
    }

    if ((m_Flags & CKBEHAVIOR_SCRIPT) != 0) {
        CKScene *scene = m_Context->GetCurrentScene();
        if (scene) {
            CKSceneObjectDesc *desc = scene->GetSceneObjectDesc(this);
            if (desc)
                desc->m_Global &= ~CK_SCENEOBJECT_ACTIVE;
        }
    }
}

int CKBehavior::ExecuteFunction() {
    if (!m_BlockData || !m_BlockData->m_Function)
        return CKBR_GENERICERROR;

    m_Context->m_Stats.BuildingBlockExecuted++;
    m_Context->m_BehaviorContext.Behavior = this;

    int result = 0;

#if defined(_MSC_VER)
    if (m_Context->m_ProfilingEnabled) {
        VxTimeProfiler profiler;
        bool callSucceeded = false;

        __try {
            result = m_BlockData->m_Function(m_Context->m_BehaviorContext);
            callSucceeded = true;
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            ErrorMessage("Error", "Execution", 1, 1);
            result = CKBR_OK;
        }

        if (callSucceeded) {
            m_Context->m_Stats.BehaviorCodeExecution += profiler.Current();
        }
    } else {
        __try {
            result = m_BlockData->m_Function(m_Context->m_BehaviorContext);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            ErrorMessage("Error", "Execution", 1, 1);
            result = CKBR_OK;
        }
    }
#else
    if (m_Context->m_ProfilingEnabled) {
        VxTimeProfiler profiler;
        result = m_BlockData->m_Function(m_Context->m_BehaviorContext);
        m_Context->m_Stats.BehaviorCodeExecution += profiler.Current();
    } else {
        result = m_BlockData->m_Function(m_Context->m_BehaviorContext);
    }
#endif

    if ((result & CKBR_ACTIVATENEXTFRAME) == 0) {
        m_Flags &= ~CKBEHAVIOR_ACTIVE;
    }

    return result;
}

void CKBehavior::FindNextBehaviorsToExecute(CKBehavior *beh) {
    if (!beh || !m_GraphData)
        return;

    for (XObjectPointerArray::Iterator outIt = beh->m_OutputArray.Begin(); outIt != beh->m_OutputArray.End(); ++outIt) {
        CKBehaviorIO *outIO = (CKBehaviorIO *)*outIt;
        if (!outIO)
            continue;

        if (!outIO->IsActive())
            continue;

        outIO->Activate(FALSE);

        for (XObjectPointerArray::Iterator linkIt = outIO->m_Links.Begin(); linkIt != outIO->m_Links.End(); ++linkIt) {
            CKBehaviorLink *link = (CKBehaviorLink *)*linkIt;
            if (!link)
                continue;

            link->SetOldFlags(link->GetOldFlags() | CKBL_OLD_TRIGGERED_THIS_FRAME);

            const int initialDelay = link->GetInitialActivationDelay();
            const CKDWORD oldFlags = link->GetOldFlags();

            if (initialDelay) {
                if (!link->GetActivationDelay())
                    link->SetActivationDelay(initialDelay);

                if ((oldFlags & 1u) == 0) {
                    link->SetOldFlags(oldFlags | CKBL_OLD_IN_DELAYED_LIST);
                    m_GraphData->m_Links.PushBack(link);
                    ++m_Context->m_Stats.BehaviorDelayedLinks;
                }
            } else {
                CKBehaviorIO *dstIO = link->GetOutBehaviorIO();
                if (!dstIO)
                    continue;

                dstIO->Activate(TRUE);

                CKBehavior *dstBeh = dstIO->GetOwner();
                if (dstIO->GetType() == CK_BEHAVIORIO_IN && dstBeh && dstBeh != this) {
                    dstBeh->ModifyFlags(CKBEHAVIOR_ACTIVE, 0);

                    const CKDWORD dstFlags = dstBeh->GetFlags();
                    if ((dstFlags & CKBEHAVIOR_RESERVED0) == 0) {
                        dstBeh->ModifyFlags(CKBEHAVIOR_RESERVED0, 0);

                        const int currentCount = m_GraphData->m_BehaviorIteratorIndex;
                        int insertPos = currentCount - 1;
                        const int priority = dstBeh->GetPriority();

                        while (insertPos >= 0) {
                            if (priority >= m_GraphData->m_BehaviorIterators[insertPos]->GetPriority())
                                break;
                            --insertPos;
                        }

                        const int targetPos = insertPos + 1;
                        for (int i = currentCount; i > targetPos; --i)
                            m_GraphData->m_BehaviorIterators[i] = m_GraphData->m_BehaviorIterators[i - 1];

                        m_GraphData->m_BehaviorIterators[targetPos] = dstBeh;
                        ++m_GraphData->m_BehaviorIteratorIndex;
                    }
                }
            }
        }
    }
}

void CKBehavior::HierarchyPostLoad() {
    // Retrieve prototype and handle naming
    CKBehaviorPrototype *proto = GetPrototype();
    if (proto) {
        CKSTRING protoName = proto->GetName();
        if (!protoName)
            protoName = "";
        if (strcmp(protoName, "Particle Systems") != 0 && strcmp(protoName, "Eval") != 0)
            SetName(protoName, TRUE);
    }

    // Handle target parameter ownership
    if (m_InputTargetParam) {
        CKObject *targetObj = m_Context->GetObject(m_InputTargetParam);
        if (targetObj) {
            if (CKParameterIn *pin = CKParameterIn::Cast(targetObj)) {
                pin->SetOwner(this);
            } else if (CKParameter *p = CKParameter::Cast(targetObj)) {
                p->SetOwner(this);
            }
        }
    }

    // Process input parameters
    int paramIndex = 0;
    for (auto it = m_InParameter.Begin(); it != m_InParameter.End(); ++it, ++paramIndex) {
        CKParameterIn *param = (CKParameterIn *) *it;
        if (!param) continue;

        param->SetOwner(this);

        if (proto) {
            CKSTRING expectedName = proto->GetInParamIndex(paramIndex);
            if (expectedName) {
                CKSTRING paramName = param->GetName();
                if (paramName) {
                    if (!strcmp(paramName, expectedName)) {
                        param->SetName(expectedName, TRUE);
                    }

                    CKParameter *source = param->GetRealSource();
                    if (source && source->GetName() && !strcmp(source->GetName(), expectedName)) {
                        source->SetName(expectedName, TRUE);
                    }
                }
            }
        }
    }

    // Process output parameters
    paramIndex = 0;
    for (auto it = m_OutParameter.Begin(); it != m_OutParameter.End(); ++it, ++paramIndex) {
        CKParameterOut *param = (CKParameterOut *) *it;
        if (!param) continue;

        if ((param->GetObjectFlags() & CK_PARAMETEROUT_PARAMOP) == 0) {
            param->SetOwner(this);

            if (proto) {
                CKSTRING expectedName = proto->GetOutParamIndex(paramIndex);
                if (expectedName && param->GetName() && !strcmp(param->GetName(), expectedName)) {
                    param->SetName(expectedName, TRUE);
                }
            }
        }
    }

    // Process local parameters
    paramIndex = 0;
    for (auto it = m_LocalParameter.Begin(); it != m_LocalParameter.End(); ++it, ++paramIndex) {
        CKParameterLocal *param = (CKParameterLocal *) *it;
        if (!param) continue;

        param->SetOwner(this);

        if (proto) {
            CKSTRING expectedName = proto->GetLocalParamIndex(paramIndex);
            if (expectedName && param->GetName() && !strcmp(param->GetName(), expectedName)) {
                param->SetName(expectedName, TRUE);
            }
        }
    }

    // Process input IOs
    int ioIndex = 0;
    for (auto it = m_InputArray.Begin(); it != m_InputArray.End(); ++it, ++ioIndex) {
        CKBehaviorIO *io = (CKBehaviorIO *) *it;
        if (!io) continue;

        io->SetOwner(this);

        if (proto) {
            CKSTRING expectedName = proto->GetInIOName(ioIndex);
            if (expectedName && io->GetName() && !strcmp(io->GetName(), expectedName)) {
                io->SetName(expectedName, TRUE);
            }
        }
    }

    // Process output IOs
    ioIndex = 0;
    for (auto it = m_OutputArray.Begin(); it != m_OutputArray.End(); ++it, ++ioIndex) {
        CKBehaviorIO *io = (CKBehaviorIO *) *it;
        if (!io) continue;

        io->SetOwner(this);

        if (proto) {
            CKSTRING expectedName = proto->GetOutIOIndex(ioIndex);
            if (expectedName && io->GetName() && !strcmp(io->GetName(), expectedName)) {
                io->SetName(expectedName, TRUE);
            }
        }
    }

    // Process sub-behaviors and operations
    if (m_GraphData) {
        // Handle sub-behaviors
        for (XObjectPointerArray::Iterator it = m_GraphData->m_SubBehaviors.Begin(); it != m_GraphData->m_SubBehaviors.End(); ++it) {
            CKBehavior *subBeh = (CKBehavior *) *it;
            if (subBeh) {
                subBeh->SetParent(this);
                subBeh->m_Flags &= ~CKBEHAVIOR_TOPMOST;
                subBeh->HierarchyPostLoad();
            }
        }

        SortSubs();

        // Update parameter operations
        for (XObjectPointerArray::Iterator it = m_GraphData->m_Operations.Begin(); it != m_GraphData->m_Operations.End(); ++it) {
            CKParameterOperation *op = (CKParameterOperation *) *it;
            if (op) {
                op->SetOwner(this);
                op->Update();
            }
        }
    }
}

int CKBehavior::BehaviorPrioritySort(CKObject *o1, CKObject *o2) {
    CKBehavior *beh1 = (CKBehavior *) o1;
    CKBehavior *beh2 = (CKBehavior *) o2;
    if (beh1 && beh2)
        return beh2->GetPriority() - beh1->GetPriority();
    return 0;
}

int CKBehavior::BehaviorPrioritySort(const void *elem1, const void *elem2) {
    CKBehavior *beh1 = *(CKBehavior **) elem1;
    CKBehavior *beh2 = *(CKBehavior **) elem2;
    if (beh1 && beh2)
        return beh2->GetPriority() - beh1->GetPriority();
    return 0;
}

void CKBehavior::ApplyPatchLoad() {
    if (!m_GraphData)
        return;

    // Iterate through all parameter operations in the behavior graph
    for (XObjectPointerArray::Iterator it = m_GraphData->m_Operations.Begin();
         it != m_GraphData->m_Operations.End(); ++it) {
        CKParameterOperation *operation = (CKParameterOperation *) *it;

        // Check if operation ownership needs correction
        if (operation->GetOwner() != this) {
            // Traverse behavior hierarchy upwards
            CKBehavior *currentBehavior = this;
            while (true) {
                CKBehavior *parent = currentBehavior->GetParent();
                if (!parent)
                    break;

                // Remove operation from parent's list
                parent->RemoveParameterOperation(operation);
                currentBehavior = parent;
            }

            // Claim ownership of the operation
            operation->SetOwner(this);
        }
    }
}
