#include "CKBeObject.h"

#include "CKFile.h"
#include "CKAttributeManager.h"
#include "CKBehaviorManager.h"
#include "CKScene.h"
#include "CKBehavior.h"
#include "CKGroup.h"
#include "CKParameterOut.h"
#include "CKMessage.h"

extern CKBOOL WarningForOlderVersion;

CK_CLASSID CKBeObject::m_ClassID = CKCID_BEOBJECT;

void CKBeObject::ExecuteBehaviors(float delta) {
    if (m_Context->IsProfilingEnable()) {
        ++m_Context->m_Stats.ActiveObjectsExecuted;
        ResetExecutionTime();
    }

    CKBOOL executed = FALSE;

    if (m_ScriptArray) {
        const int count = m_ScriptArray->Size();
        for (int i = 0; i < count; ++i) {
            CKBehavior *beh = (CKBehavior *) m_ScriptArray->GetObject(i);
            if (beh && beh->IsActive()) {
                executed = TRUE;
                beh->Execute(delta);
                m_LastExecutionTime += beh->GetLastExecutionTime();
            }
        }
    }

    if (!executed) {
        if (GetClassID() != CKCID_CHARACTER)
            m_Context->GetBehaviorManager()->RemoveObjectNextFrame(this);
    }
}

CKBOOL CKBeObject::IsInGroup(CKGroup *group) {
    if (!group)
        return FALSE;

    if (group->m_GroupIndex >= m_Groups.Size())
        return FALSE;

    return m_Groups.IsSet(group->m_GroupIndex);
}

CKBOOL CKBeObject::HasAttribute(CKAttributeType AttribType) {
    return m_Attributes.IsHere(AttribType);
}

CKBOOL CKBeObject::SetAttribute(CKAttributeType AttribType, CK_ID parameter) {
    if (AttribType < 0)
        return FALSE;

    if (HasAttribute(AttribType))
        return FALSE;

    CKAttributeManager *am = m_Context->GetAttributeManager();

    CK_CLASSID cid = am->GetAttributeCompatibleClassId(AttribType);
    if (!CKIsChildClassOf(GetClassID(), cid))
        return FALSE;

    CKGUID guid = am->GetAttributeParameterGUID(AttribType);
    if (guid.IsValid()) {
        if (!m_Context->GetObject(parameter)) {
            CKParameterOut *pout = m_Context->CreateCKParameterOut(nullptr, guid, IsDynamic());
            if (pout) {
                CKSTRING defValue = am->GetAttributeDefaultValue(AttribType);
                if (defValue) {
                    pout->SetValue(defValue);
                }
                pout->SetOwner(this);
                parameter = pout->GetID();
            } else {
                parameter = 0;
            }
        }
    }

    CKAttributeVal attrVal = {AttribType, parameter};
    const int count = m_Attributes.Size();
    m_Attributes.Insert(AttribType, attrVal);
    am->AddAttributeToObject(AttribType, this);
    return count != m_Attributes.Size();
}

CKBOOL CKBeObject::RemoveAttribute(CKAttributeType AttribType) {
    XAttributeList::Iterator it = m_Attributes.Find(AttribType);
    if (it == m_Attributes.End())
        return FALSE;

    CKAttributeVal val = *it;
    CK_ID paramToDestroy = val.Parameter;

    m_Context->GetAttributeManager()->RemoveAttributeFromObject(AttribType, this);
    m_Attributes.Remove(AttribType);
    m_Context->DestroyObject(paramToDestroy);
    return TRUE;
}

CKParameterOut *CKBeObject::GetAttributeParameter(CKAttributeType AttribType) {
    XAttributeList::Iterator it = m_Attributes.Find(AttribType);
    if (it == m_Attributes.End())
        return nullptr;

    CKAttributeVal val = *it;
    return (CKParameterOut *) m_Context->GetObject(val.Parameter);
}

int CKBeObject::GetAttributeCount() {
    return m_Attributes.Size();
}

int CKBeObject::GetAttributeType(int index) {
    if (index < 0 || index >= m_Attributes.Size())
        return -1;
    int i = 0;
    for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
        if (index == i++)
            return (*it).AttribType;
    }
    return -1;
}

CKParameterOut *CKBeObject::GetAttributeParameterByIndex(int index) {
    if (index < 0 || index >= m_Attributes.Size())
        return nullptr;
    int i = 0;
    for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
        if (index == i++)
            return (CKParameterOut *) m_Context->GetObject((*it).Parameter);
    }
    return nullptr;
}

void CKBeObject::GetAttributeList(CKAttributeVal *liste) {
    if (!liste) return;
    for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
        liste->AttribType = (*it).AttribType;
        liste->Parameter = (*it).Parameter;
        ++liste;
    }
}

void CKBeObject::RemoveAllAttributes() {
    if (m_Attributes.Size() == 0) return;

    CKAttributeManager *am = m_Context->GetAttributeManager();

    for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
        am->RemoveAttributeFromObject((*it).AttribType, this);
        m_Context->DestroyObject((*it).Parameter);
    }
    m_Attributes.Clear();
}

CKERROR CKBeObject::AddScript(CKBehavior *script) {
    if (!script)
        return CKERR_INVALIDPARAMETER;

    // Check for duplicate script
    if (m_ScriptArray) {
        if (m_ScriptArray->FindObject(script)) {
            return CKERR_ALREADYPRESENT;
        }
    }

    // Check class compatibility
    if (!script->IsUsingTarget()) {
        CK_CLASSID requiredCID = script->GetCompatibleClassID();
        if (!CKIsChildClassOf(this, requiredCID)) {
            return CKERR_INCOMPATIBLECLASSID;
        }
    }

    // Handle ownership transfer
    CKBeObject *previousOwner = script->GetOwner();
    CKERROR err = script->SetOwner(this, TRUE);
    if (err != CK_OK) {
        script->SetOwner(previousOwner, TRUE);
        return err;
    }

    // Create script array if needed
    if (!m_ScriptArray) {
        m_ScriptArray = new XObjectPointerArray();
    }
    if (!m_ScriptArray) return CKERR_OUTOFMEMORY;

    // Add script to array
    m_ScriptArray->PushBack(script);

    // Maintain script order
    SortScripts();

    // Update scene relationships
    AddToSelfScenes(script);

    return CK_OK;
}

CKBehavior *CKBeObject::RemoveScript(CK_ID id) {
    if (!m_ScriptArray)
        return nullptr;

    CKBehavior *script = (CKBehavior *) m_Context->GetObject(id);
    if (!script)
        return nullptr;

    for (int i = 0; i < m_ScriptArray->Size(); ++i) {
        if ((*m_ScriptArray)[i] == script) {
            return RemoveScript(i);
        }
    }

    return nullptr;
}

CKBehavior *CKBeObject::RemoveScript(int pos) {
    if (!m_ScriptArray)
        return nullptr;

    CK_ID scriptId = m_ScriptArray->GetObjectID(pos);
    if (!scriptId)
        return nullptr;

    // Remove from array using XRemove pattern
    m_ScriptArray->RemoveAt(pos);

    CKBehavior *script = (CKBehavior *)m_Context->GetObject(scriptId);
    if (script) {
        script->SetOwner(nullptr, TRUE);

        if (m_ScriptArray->IsEmpty()) {
            delete m_ScriptArray;
            m_ScriptArray = nullptr;
        }

        if (script->GetFlags() & CKBEHAVIOR_SCRIPT)
            RemoveFromSelfScenes(script);
    }

    return script;
}

CKERROR CKBeObject::RemoveAllScripts() {
    if (!m_ScriptArray) return CK_OK;

    // Remove scripts with CKBEHAVIOR_SCRIPT flag from all scenes this object is in
    for (XObjectPointerArray::Iterator it = m_ScriptArray->Begin(); it != m_ScriptArray->End(); ++it) {
        CKBehavior *script = (CKBehavior *) *it;
        if (script && (script->GetFlags() & CKBEHAVIOR_SCRIPT))
            RemoveFromSelfScenes(script);
    }

    // Delete the array
    delete m_ScriptArray;
    m_ScriptArray = nullptr;
    return CK_OK;
}

CKBehavior *CKBeObject::GetScript(int pos) {
    if (m_ScriptArray && pos >= 0 && pos < m_ScriptArray->Size())
        return (CKBehavior *) (*m_ScriptArray)[pos];
    return nullptr;
}

int CKBeObject::GetScriptCount() {
    return m_ScriptArray ? m_ScriptArray->Size() : 0;
}

int CKBeObject::GetPriority() {
    return m_Priority;
}

void CKBeObject::SetPriority(int priority) {
    m_Priority = priority;
    CKScene *scene = m_Context->GetCurrentScene();
    if (IsInScene(scene))
        m_Context->m_BehaviorManager->SortObjects();
}

int CKBeObject::GetLastFrameMessageCount() {
    return m_LastFrameMessages ? m_LastFrameMessages->Size() : 0;
}

CKMessage *CKBeObject::GetLastFrameMessage(int pos) {
    if (m_LastFrameMessages && pos >= 0 && pos < m_LastFrameMessages->Size())
        return (*m_LastFrameMessages)[pos];
    return nullptr;
}

void CKBeObject::SetAsWaitingForMessages(CKBOOL wait) {
    if (wait) {
        ++m_Waiting;
    } else {
        --m_Waiting;
        if (m_Waiting < 0) m_Waiting = 0;
    }
}

CKBOOL CKBeObject::IsWaitingForMessages() {
    return m_Waiting > 0;
}

int CKBeObject::CallBehaviorCallbackFunction(CKDWORD Message, CKGUID *behguid) {
    int result = 0;
    if (!m_ScriptArray) return result;

    for (int i = 0; i < m_ScriptArray->Size(); ++i) {
        CKBehavior *script = (CKBehavior *) (*m_ScriptArray)[i];
        if (script)
            result = script->CallSubBehaviorsCallbackFunction(Message, behguid);
    }
    return result;
}

float CKBeObject::GetLastExecutionTime() {
    return m_LastExecutionTime;
}

void CKBeObject::ApplyPatchForOlderVersion(int NbObject, CKFileObject *FileObjects) {
    if (!m_ScriptArray) return;

    for (XObjectPointerArray::Iterator it = m_ScriptArray->Begin(); it != m_ScriptArray->End();) {
        CKBehavior *script = (CKBehavior *) *it;
        if (script && !(script->GetFlags() & CKBEHAVIOR_SCRIPT)) {
            WarningForOlderVersion = TRUE;
            it = m_ScriptArray->Remove(it);
        } else {
            ++it;
        }
    }
}

CKBeObject::CKBeObject() : CKSceneObject(::GetCKContext(0), nullptr) {
    m_ScriptArray = nullptr;
    m_LastFrameMessages = nullptr;
    m_Waiting = 0;
    m_LastExecutionTime = 0.0f;
    m_Priority = 0;
}

CKBeObject::CKBeObject(CKContext *Context, CKSTRING name) : CKSceneObject(Context, name) {
    m_ScriptArray = nullptr;
    m_LastFrameMessages = nullptr;
    m_Waiting = 0;
    m_LastExecutionTime = 0.0f;
    m_Priority = 0;
}

CKBeObject::~CKBeObject() {
    if (m_ScriptArray) {
        delete m_ScriptArray;
        m_ScriptArray = nullptr;
    }
    RemoveAllLastFrameMessage();
    if (m_LastFrameMessages) {
        delete m_LastFrameMessages;
        m_LastFrameMessages = nullptr;
    }
}

CK_CLASSID CKBeObject::GetClassID() {
    return m_ClassID;
}

void CKBeObject::PreSave(CKFile *file, CKDWORD flags) {
    CKObject::PreSave(file, flags);

    if (m_ScriptArray)
        file->SaveObjects(m_ScriptArray->Begin(), m_ScriptArray->Size());

    for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
        if (!(m_Context->GetAttributeManager()->GetAttributeFlags((*it).AttribType) & CK_ATTRIBUT_DONOTSAVE))
            file->SaveObject(m_Context->GetObject((*it).Parameter));
    }
}

CKStateChunk *CKBeObject::Save(CKFile *file, CKDWORD flags) {
    // Save base object data
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    if (!file && !(flags & CK_STATESAVE_BEOBJECTONLY)) {
        return baseChunk;
    }

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_BEOBJECT, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if (file) {
        // Save scripts
        if (m_ScriptArray) {
            chunk->WriteIdentifier(CK_STATESAVE_SCRIPTS);
            m_ScriptArray->Save(chunk);
        }

        // Save priority data
        if (m_Priority != 0) {
            chunk->WriteIdentifier(CK_STATESAVE_DATAS);
            chunk->WriteDword(0x10000000); // Version identifier
            chunk->WriteInt(m_Priority);
        }
    }

    // Save attributes
    if (m_Attributes.Size() > 0) {
        XArray<CKAttributeVal> attributesToSave;
        CKAttributeManager *am = m_Context->GetAttributeManager();

        // Collect attributes that should be saved
        for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
            if (!(am->GetAttributeFlags((*it).AttribType) & CK_ATTRIBUT_DONOTSAVE)) {
                attributesToSave.PushBack(*it);
            }
        }

        // Write attributes to chunk
        if (attributesToSave.Size() > 0) {
            chunk->WriteIdentifier(CK_STATESAVE_NEWATTRIBUTES);
            chunk->StartObjectIDSequence(attributesToSave.Size());

            // Write object references
            for (int i = 0; i < attributesToSave.Size(); ++i) {
                CKObject *obj = m_Context->GetObject(attributesToSave[i].Parameter);
                chunk->WriteObjectSequence(obj);
            }

            // Write attribute data
            if (!file) {
                chunk->StartSubChunkSequence(attributesToSave.Size());
                for (int i = 0; i < attributesToSave.Size(); ++i) {
                    CKObject *obj = m_Context->GetObject(attributesToSave[i].Parameter);
                    CKStateChunk *paramChunk = obj ? obj->Save(nullptr, flags) : nullptr;
                    chunk->WriteSubChunkSequence(paramChunk);
                    if (paramChunk)
                        delete paramChunk;
                }
            }

            // Write manager sequence
            chunk->StartManagerSequence(ATTRIBUTE_MANAGER_GUID, attributesToSave.Size());
            for (int i = 0; i < attributesToSave.Size(); ++i) {
                chunk->WriteManagerSequence(attributesToSave[i].AttribType);
            }
        }
    }

    // Save scene information
    if (file && !file->m_SceneSaved) {
        CKScene *scene = m_Context->GetCurrentScene();
        if (scene) {
            CKSceneObjectDesc *desc = scene->GetSceneObjectDesc(this);
            if (desc) {
                CKDWORD sceneFlags = desc->m_Flags;
                if (desc->m_InitialValue) sceneFlags |= CK_SCENEOBJECT_INTERNAL_IC;
                chunk->WriteIdentifier(CK_STATESAVE_SINGLEACTIVITY);
                chunk->WriteDword(sceneFlags);
            }
        }
    }

    // Finalize chunk
    if (GetClassID() == CKCID_BEOBJECT) {
        chunk->CloseChunk();
    } else {
        chunk->UpdateDataSize();
    }

    return chunk;
}

CKERROR CKBeObject::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Load base object data
    CKERROR err = CKObject::Load(chunk, file);
    if (err != CK_OK)
        return err;

    CKContext *context = m_Context;
    CKAttributeManager *am = context->GetAttributeManager();

    if (file) {
        // Cleanup existing scripts
        if (m_ScriptArray) {
            delete m_ScriptArray;
            m_ScriptArray = nullptr;
        }

        if (chunk->GetDataVersion() < 5 && chunk->SeekIdentifier(CK_STATESAVE_BEHAVIORS)) {
            if (!m_ScriptArray) {
                m_ScriptArray = new XObjectPointerArray();
            }
            m_ScriptArray->Load(context, chunk);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_SCRIPTS)) {
            if (!m_ScriptArray) {
                m_ScriptArray = new XObjectPointerArray();
            }
            m_ScriptArray->Load(context, chunk);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_DATAS)) {
            CKDWORD versionFlag = chunk->ReadDword();
            if (chunk->GetDataVersion() < 5) {
                chunk->ReadInt();
                chunk->ReadInt();
                chunk->ReadInt();
                m_Priority = chunk->ReadInt();
                m_Waiting = FALSE;
            } else if (versionFlag & 0x10000000) {
                m_Priority = chunk->ReadInt();
                m_Waiting = FALSE;
            } else {
                m_Priority = 0;
                m_Waiting = FALSE;
            }
        }
    } else {
        if (chunk->SeekIdentifier(CK_STATESAVE_DATAS)) {
            chunk->ReadInt();
        }
    }

    // Load attributes
    if (chunk->SeekIdentifier(CK_STATESAVE_NEWATTRIBUTES)) {
        // Handle legacy attribute remapping
        if (chunk->GetChunkVersion() < 6 && !chunk->GetManagers()) {
            am->PatchRemapBeObjectFileChunk(chunk);
        }

        if (file)
            RemoveAllAttributes();

        const int attrCount = chunk->StartReadSequence();
        XArray<CK_ID> attrObjects(attrCount);
        XArray<CKStateChunk *> attrChunks;

        // Read attribute object references
        for (int i = 0; i < attrCount; ++i) {
            attrObjects.PushBack(chunk->ReadObjectID());
        }

        // Read attribute data if not in file mode
        if (!file) {
            attrChunks.Resize(attrCount);
            chunk->StartReadSequence();
            for (int i = 0; i < attrCount; ++i) {
                attrChunks[i] = chunk->ReadSubChunk();
            }
        }

        // Read attribute manager sequence
        CKGUID managerGuid;
        const int seqCount = chunk->StartManagerReadSequence(&managerGuid);
        if (managerGuid == ATTRIBUTE_MANAGER_GUID && seqCount == attrCount) {
            for (int i = 0; i < attrCount; ++i) {
                CKAttributeType attrType = chunk->ReadDword();
                CKObject *paramObj = context->GetObject(attrObjects[i]);

                SetAttribute(attrType, paramObj ? paramObj->GetID() : 0);

                if (!file) {
                    CKParameter *param = GetAttributeParameter(attrType);
                    if (param) {
                        param->Load(attrChunks[i], nullptr);
                        CKParameterType paramType = am->GetAttributeParameterType(attrType);
                        if (param->GetType() != paramType) {
                            param->SetType(paramType);
                        }
                    }
                }
            }
        } else {
            context->OutputToConsoleEx("Load:%s :Invalid Attribute Data", m_Name);
        }

        // Cleanup temporary chunks
        for (int i = 0; i < attrChunks.Size(); ++i) {
            delete attrChunks[i];
            attrChunks[i] = nullptr;
        }
    } else {
        CKBOOL oldVersion = FALSE;

        if (chunk->SeekIdentifier(CK_STATESAVE_ATTRIBUTES) && chunk->ReadInt() > 0) {
            int a = chunk->ReadInt();
            int b = chunk->ReadInt();
            if (a < 0 || a >= 0x36 || b <= 0 || b > 0x41) {
                oldVersion = TRUE;
                WarningForOlderVersion = TRUE;
            }
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_ATTRIBUTES)) {
            if (file)
                RemoveAllAttributes();

            const int attrCount = chunk->ReadInt();
            for (int i = 0; i < attrCount; ++i) {
                CK_CLASSID compatibleCid = CKCID_BEOBJECT;
                if (!oldVersion)
                    compatibleCid = chunk->ReadInt();

                CKSTRING name = nullptr;
                if (CKIsChildClassOf(compatibleCid, CKCID_OBJECT) && (chunk->ReadString(&name), name)) {
                    CKSTRING category = nullptr;
                    chunk->ReadString(&category);
                    CKGUID paramGuid = chunk->ReadGuid();

                    CKAttributeType attrType = am->RegisterNewAttributeType(name, paramGuid, compatibleCid);
                    if (category) {
                        am->AddCategory(category);
                        am->SetAttributeCategory(attrType, category);
                    }
                    SetAttribute(attrType);

                    CK_ID paramId = chunk->ReadObjectID();
                    CKParameter *param = (CKParameter *) context->GetObject(paramId);
                    if (param) {
                        for (auto it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
                            if ((*it).AttribType == attrType) {
                                if ((*it).Parameter != paramId) {
                                    m_Context->DestroyObject((*it).Parameter);
                                    (*it).Parameter = paramId;
                                }
                                break;
                            }
                        }
                    } else {
                        param = GetAttributeParameter(attrType);
                    }

                    if (!file) {
                        CKStateChunk *subChunk = chunk->ReadSubChunk();
                        if (param && subChunk) {
                            param->Load(subChunk, nullptr);
                            delete subChunk;
                        }
                    }

                    delete[] name;
                    delete[] category;
                } else {
                    context->OutputToConsoleEx("Load:%s :Invalid Attribute Data", m_Name);
                }
            }
        }
    }

    if (chunk->SeekIdentifier(CK_STATESAVE_SINGLEACTIVITY)) {
        CKDWORD flags = chunk->ReadDword();
        if (flags)
            m_Context->m_ObjectManager->AddSingleObjectActivity(this, flags);
    }

    return CK_OK;
}

void CKBeObject::PreDelete() {
    CKObject::PreDelete();

    CKAttributeManager *am = m_Context->GetAttributeManager();
    for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
        am->RemoveAttributeFromObject((*it).AttribType, this);
    }
    m_Attributes.Clear();
}

int CKBeObject::GetMemoryOccupation() {
    // Base object + in-object members + owned buffers.
    int size = CKSceneObject::GetMemoryOccupation();
    size += static_cast<int>(sizeof(CKBeObject) - sizeof(CKSceneObject));
    size += m_Groups.GetMemoryOccupation(FALSE);
    size += m_Attributes.GetMemoryOccupation(FALSE);

    if (m_ScriptArray) {
        size += m_ScriptArray->GetMemoryOccupation(TRUE);
    }

    if (m_LastFrameMessages) {
        size += m_LastFrameMessages->GetMemoryOccupation(TRUE);
    }

    return size;
}

CKBOOL CKBeObject::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (cid == CKCID_BEHAVIOR && m_ScriptArray && m_ScriptArray->FindObject(obj)) {
        return TRUE;
    }
    return CKObject::IsObjectUsed(obj, cid);
}

CKERROR CKBeObject::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::PrepareDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    CKDWORD classDeps = context.GetClassDependencies(m_ClassID);
    if (classDeps & 1 || context.IsInMode(CK_DEPENDENCIES_DELETE)) {
        if (context.IsInMode(CK_DEPENDENCIES_COPY)) {
            // Incremental mode: Process each script individually
            for (int i = 0; i < GetScriptCount(); ++i) {
                CKBehavior *script = GetScript(i);
                if (script && !(script->GetFlags() & CKBEHAVIOR_LOCKED)) {
                    script->PrepareDependencies(context);
                }
            }
        } else {
            // Full mode: Process entire script array
            if (m_ScriptArray) {
                m_ScriptArray->Prepare(context);
            }
        }

    }

    if ((classDeps & 2) || context.IsInMode(CK_DEPENDENCIES_DELETE)) {
        CKAttributeManager *am = m_Context->GetAttributeManager();
        for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
            CKAttributeVal &val = *it;
            if (!(am->GetAttributeFlags(val.AttribType) & CK_ATTRIBUT_DONOTCOPY)) {
                CKParameter *param = (CKParameter *) m_Context->GetObject(val.Parameter);
                if (param) {
                    param->PrepareDependencies(context);
                }
            }
        }
    }

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR CKBeObject::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    CKDWORD classDeps = context.GetClassDependencies(m_ClassID);
    if (classDeps & 1) {
        if (m_ScriptArray) m_ScriptArray->Remap(context);
    }
    if (classDeps & 2) {
        for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
            (*it).Parameter = context.RemapID((*it).Parameter);
        }
    }

    return CK_OK;
}

CKERROR CKBeObject::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    CKBeObject *beo = (CKBeObject *) &o;

    // Get dependency flags for BeObject class
    const CKDWORD classDeps = context.GetClassDependencies(m_ClassID);

    // Copy scripts if requested
    if (classDeps & 1) {
        RemoveAllScripts();

        // Copy scripts from source
        if (beo->m_ScriptArray && beo->m_ScriptArray->Size() > 0) {
            m_ScriptArray = new XObjectPointerArray();

            // Clone valid scripts
            for (XObjectPointerArray::Iterator it = beo->m_ScriptArray->Begin(); it != beo->m_ScriptArray->End(); ++it) {
                CKBehavior *srcScript = (CKBehavior *) *it;
                if (!(srcScript->GetFlags() & CKBEHAVIOR_LOCKED)) {
                    m_ScriptArray->PushBack(srcScript);
                }
            }
        }
    }

    // Copy attributes if requested
    if (classDeps & 2) {
        CKAttributeManager *am = m_Context->GetAttributeManager();

        for (XAttributeList::Iterator it = beo->m_Attributes.Begin(); it != beo->m_Attributes.End(); ++it) {
            const CKAttributeVal &val = *it;
            if (!(am->GetAttributeFlags(val.AttribType) & CK_ATTRIBUT_DONOTCOPY)) {
                SetAttribute(val.AttribType, val.Parameter);
            }
        }
    }

    if (classDeps & 4) {
        const int count = m_Context->GetObjectsCountByClassID(CKCID_GROUP);
        CK_ID *ids = m_Context->GetObjectsListByClassID(CKCID_GROUP);
        for (int i = 0; i < count; ++i) {
            CKGroup *group = (CKGroup *) m_Context->GetObject(ids[i]);
            if (group && beo->IsInGroup(group)) {
                group->AddObject(this);
            }
        }
    }

    SetPriority(beo->GetPriority());

    return CK_OK;
}

void CKBeObject::AddToScene(CKScene *scene, CKBOOL dependencies) {
    if (!scene) return;
    CKSceneObject::AddToScene(scene, dependencies);
    if (dependencies && m_ScriptArray) {
        for (int i = 0; i < m_ScriptArray->Size(); ++i) {
            CKBehavior *beh = (CKBehavior *) m_ScriptArray->GetObject(i);
            if (beh && (beh->GetFlags() & CKBEHAVIOR_SCRIPT))
                scene->AddObject(beh);
        }
    }
}

void CKBeObject::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    if (!scene) return;
    if (dependencies && m_ScriptArray) {
        for (int i = 0; i < m_ScriptArray->Size(); ++i) {
            CKBehavior *beh = (CKBehavior *) m_ScriptArray->GetObject(i);
            if (beh && (beh->GetFlags() & CKBEHAVIOR_SCRIPT))
                scene->RemoveObject(beh);
        }
    }
    CKSceneObject::RemoveFromScene(scene, dependencies);
}

void CKBeObject::AddToGroup(CKGroup *group) {
    if (group) m_Groups.Set(group->m_GroupIndex);
}

void CKBeObject::RemoveFromGroup(CKGroup *group) {
    if (group) m_Groups.Unset(group->m_GroupIndex);
}

CKSTRING CKBeObject::GetClassName() {
    return "Behavioral Object";
}

int CKBeObject::GetDependenciesCount(int mode) {
    if (mode == CK_DEPENDENCIES_COPY || mode == CK_DEPENDENCIES_SAVE)
        return 3;
    return 0;
}

CKSTRING CKBeObject::GetDependencies(int i, int mode) {
    switch (i) {
    case 0:
        return "Scripts";
    case 1:
        return "Attributes";
    case 2:
        return "Groups Belonging";
    default:
        return nullptr;
    }
}

void CKBeObject::Register() {
    CKPARAMETERFROMCLASS(CKBeObject, CKPGUID_BEOBJECT);
    CKCLASSDEFAULTCOPYDEPENDENCIES(CKBeObject, CK_DEPENDENCIES_COPY_OBJECT_NAME);
}

CKBeObject *CKBeObject::CreateInstance(CKContext *Context) {
    return new CKBeObject(Context);
}

void CKBeObject::AddToSelfScenes(CKSceneObject *o) {
    int count = m_Context->GetObjectsCountByClassID(CKCID_SCENE);
    CK_ID *ids = m_Context->GetObjectsListByClassID(CKCID_SCENE);
    for (int i = 0; i < count; ++i) {
        CKScene *scene = (CKScene *) m_Context->GetObject(ids[i]);
        if (scene && IsInScene(scene)) {
            o->AddToScene(scene, TRUE);
        }
    }
}

void CKBeObject::RemoveFromSelfScenes(CKSceneObject *o) {
    int count = m_Context->GetObjectsCountByClassID(CKCID_SCENE);
    CK_ID *ids = m_Context->GetObjectsListByClassID(CKCID_SCENE);
    for (int i = 0; i < count; ++i) {
        CKScene *scene = (CKScene *) m_Context->GetObject(ids[i]);
        if (scene && IsInScene(scene)) {
            o->RemoveFromScene(scene, TRUE);
        }
    }
}

void CKBeObject::SortScripts() {
    if (m_ScriptArray) {
        m_ScriptArray->BubbleSort(CKBehavior::BehaviorPrioritySort);
    }
}

void CKBeObject::RemoveFromAllGroups() {
    if (m_Groups.BitSet()) {
        int count = m_Context->GetObjectsCountByClassID(CKCID_GROUP);
        CK_ID *ids = m_Context->GetObjectsListByClassID(CKCID_GROUP);
        for (int i = 0; i < count; ++i) {
            CKGroup *group = (CKGroup *) m_Context->GetObject(ids[i]);
            if (group && m_Groups.IsSet(group->m_GroupIndex)) {
                group->m_ObjectArray.Remove(this);
            }
        }
    }
}

CKERROR CKBeObject::AddLastFrameMessage(CKMessage *msg) {
    if (!msg) return CKERR_INVALIDPARAMETER;
    if (!m_LastFrameMessages) m_LastFrameMessages = new XArray<CKMessage *>();
    if (!m_LastFrameMessages) return CKERR_OUTOFMEMORY;
    if (!m_LastFrameMessages->IsHere(msg)) {
        m_LastFrameMessages->PushBack(msg);
        msg->AddRef();
    }
    return CK_OK;
}

CKERROR CKBeObject::RemoveAllLastFrameMessage() {
    if (m_LastFrameMessages) {
        for (XArray<CKMessage *>::Iterator it = m_LastFrameMessages->Begin(); it != m_LastFrameMessages->End(); ++it) {
            if (*it) (*it)->Release();
        }
        m_LastFrameMessages->Resize(0);
    }
    return CK_OK;
}

void CKBeObject::ApplyOwner() {
    if (m_ScriptArray) {
        for (int i = 0; i < m_ScriptArray->Size(); ++i) {
            CKBehavior *beh = (CKBehavior *) m_ScriptArray->GetObject(i);
            if (beh) beh->SetOwner(this, FALSE);
        }
    }
    for (XAttributeList::Iterator it = m_Attributes.Begin(); it != m_Attributes.End(); ++it) {
        CKParameterOut *param = (CKParameterOut *) m_Context->GetObject((*it).Parameter);
        if (param) param->SetOwner(this);
    }
}

void CKBeObject::ResetExecutionTime() {
    m_LastExecutionTime = 0.0f;
    if (m_ScriptArray) {
        XObjectPointerArray &array = *m_ScriptArray;
        for (int i = 0; i < array.Size(); ++i) {
            CKBehavior *beh = (CKBehavior *) array[i];
            if (beh)
                beh->ResetExecutionTime();
        }
    }
}

int CKBeObject::BeObjectPrioritySort(const void *o1, const void *o2) {
    CKBeObject *obj1 = *(CKBeObject **) o1;
    CKBeObject *obj2 = *(CKBeObject **) o2;
    if (obj1 && obj2) {
        return obj2->GetPriority() - obj1->GetPriority();
    }
    return 0;
}
