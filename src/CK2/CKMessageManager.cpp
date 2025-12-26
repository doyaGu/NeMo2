#include "CKMessageManager.h"

#include "CKFile.h"
#include "CKBehavior.h"
#include "CKBehaviorIO.h"
#include "CKScene.h"
#include "CKGroup.h"
#include "CKBodyPart.h"
#include "CKCharacter.h"
#include "CKDebugContext.h"
#include "CKParameterManager.h"

extern int g_MaxClassID;

CKMessageType CKMessageManager::AddMessageType(CKSTRING MsgName) {
    if (!MsgName || MsgName[0] == '\0') return -1;

    XString name(MsgName);
    int i = m_RegisteredMessageTypes.GetPosition(name);
    if (i >= 0) {
        return i;
    }

    int newIndex = m_RegisteredMessageTypes.Size();
    m_RegisteredMessageTypes.PushBack(name);

    CKWaitingObjectArray **newList = new CKWaitingObjectArray *[m_RegisteredMessageTypes.Size()];
    if (m_MsgWaitingList) {
        memcpy(newList, m_MsgWaitingList, (m_RegisteredMessageTypes.Size() - 1) * sizeof(CKWaitingObjectArray *));
        delete[] m_MsgWaitingList;
    }
    newList[newIndex] = nullptr;
    m_MsgWaitingList = newList;

    return newIndex;
}

CKSTRING CKMessageManager::GetMessageTypeName(CKMessageType MsgType) {
    if (MsgType >= 0 && MsgType < m_RegisteredMessageTypes.Size())
        return m_RegisteredMessageTypes[MsgType].Str() ? m_RegisteredMessageTypes[MsgType].Str() : (CKSTRING)"";
    return nullptr;
}

int CKMessageManager::GetMessageTypeCount() {
    return m_RegisteredMessageTypes.Size();
}

void CKMessageManager::RenameMessageType(CKMessageType MsgType, CKSTRING NewName) {
    if (MsgType >= 0 && MsgType < m_RegisteredMessageTypes.Size())
        m_RegisteredMessageTypes[MsgType] = NewName ? NewName : "";
}

void CKMessageManager::RenameMessageType(CKSTRING OldName, CKSTRING NewName) {
    if (OldName && NewName)
        RenameMessageType(AddMessageType(OldName), NewName);
}

CKERROR CKMessageManager::SendMessage(CKMessage *msg) {
    if (!msg || msg->m_MessageType < 0) return CKERR_INVALIDPARAMETER;

    msg->AddRef();
    m_ReceivedMsgThisFrame.PushBack(msg);

    CKContext *ctx = m_Context;
    if (ctx->m_DebugContext && ctx->m_DebugContext->InDebug) {
        if (ctx->m_UICallBackFct) {
            CKUICallbackStruct cbData;
            cbData.Reason = CKUIM_DEBUGMESSAGESEND;
            cbData.DebugMessageSent = msg;
            ctx->m_UICallBackFct(cbData, ctx->m_InterfaceModeData);
        }
    }

    return CK_OK;
}

CKMessage *CKMessageManager::SendMessageSingle(CKMessageType MsgType, CKBeObject *dest, CKBeObject *sender) {
    if (MsgType < 0)
        return nullptr;

    CKMessage *msg = CreateMessageSingle(MsgType, dest, sender);
    if (msg) {
        SendMessage(msg);
        msg->Release();
    }
    return msg;
}

CKMessage *CKMessageManager::SendMessageGroup(CKMessageType MsgType, CKGroup *group, CKBeObject *sender) {
    if (!group || MsgType < 0)
        return nullptr;

    CKMessage *msg = CreateMessageGroup(MsgType, group, sender);
    if (msg) {
        SendMessage(msg);
        msg->Release();
    }
    return msg;
}

CKMessage *CKMessageManager::SendMessageBroadcast(CKMessageType MsgType, CK_CLASSID cid, CKBeObject *sender) {
    CKMessage *msg = CreateMessageBroadcast(MsgType, cid, sender);
    if (msg) {
        SendMessage(msg);
        msg->Release();
    }
    return msg;
}

CKERROR CKMessageManager::RegisterWait(CKMessageType MsgType, CKBehavior *behav, int OutputToActivate, CKBeObject *obj) {
    if (MsgType < 0 || MsgType >= m_RegisteredMessageTypes.Size()) return CKERR_INVALIDMESSAGE;
    if (!behav) return CKERR_INVALIDPARAMETER;

    CKBehaviorIO *output = behav->GetOutput(OutputToActivate);

    behav->ModifyFlags(CKBEHAVIOR_WAITSFORMESSAGE, 0);

    CKWaitingObjectArray *&waitList = m_MsgWaitingList[MsgType];
    if (!waitList) {
        waitList = new CKWaitingObjectArray;
        if (!waitList) return CKERR_OUTOFMEMORY;
    }

    for (CKWaitingObjectArray::Iterator it = waitList->Begin(); it != waitList->End(); ++it) {
        if (it->m_Behavior == behav && it->m_Output == output)
            return CKERR_ALREADYPRESENT;
    }

    CKWaitingObject newEntry;
    newEntry.m_Behavior = behav;
    newEntry.m_Output = output;
    newEntry.m_BeObject = obj;
    waitList->PushBack(newEntry);

    return CK_OK;
}

CKERROR CKMessageManager::RegisterWait(CKSTRING MsgName, CKBehavior *behav, int OutputToActivate, CKBeObject *obj) {
    return RegisterWait(AddMessageType(MsgName), behav, OutputToActivate, obj);
}

CKERROR CKMessageManager::UnRegisterWait(CKMessageType MsgType, CKBehavior *behav, int OutputToActivate) {
    if (MsgType < 0 || MsgType >= m_RegisteredMessageTypes.Size()) return CKERR_INVALIDMESSAGE;
    CKWaitingObjectArray *waitList = m_MsgWaitingList[MsgType];
    if (!waitList) return CK_OK;

    CKBehaviorIO *output = nullptr;
    if (behav && OutputToActivate != -1)
        output = behav->GetOutput(OutputToActivate);

    for (CKWaitingObjectArray::Iterator it = waitList->Begin(); it != waitList->End(); ++it) {
        if (it->m_Behavior == behav && (output == nullptr || it->m_Output == output)) {
            waitList->Remove(it);
            if (behav)
                behav->ModifyFlags(0, CKBEHAVIOR_WAITSFORMESSAGE);
            break;
        }
    }

    return CK_OK;
}

CKERROR CKMessageManager::UnRegisterWait(CKSTRING MsgName, CKBehavior *behav, int OutputToActivate) {
    return UnRegisterWait(AddMessageType(MsgName), behav, OutputToActivate);
}

CKERROR CKMessageManager::RegisterDefaultMessages() {
    AddMessageType("OnClick");
    AddMessageType("OnDblClick");
    return CK_OK;
}

CKStateChunk *CKMessageManager::SaveData(CKFile *SavedFile) {
    XBitArray usedMessageTypes(m_RegisteredMessageTypes.Size());
    CKParameterType msgParamType = m_Context->GetParameterManager()->ParameterGuidToType(CKPGUID_MESSAGE);

    // Check for messages used in output parameters
    const int paramOutCount = SavedFile->m_IndexByClassId[CKCID_PARAMETEROUT].Size();
    for (int i = 0; i < paramOutCount; ++i) {
        CKParameterOut *param = (CKParameterOut *) SavedFile->m_FileObjects[SavedFile->m_IndexByClassId[CKCID_PARAMETEROUT][i]].ObjPtr;
        if (param && param->GetType() == msgParamType) {
            int msgType;
            param->GetValue(&msgType);
            if (msgType >= 0) usedMessageTypes.Set(msgType);
        }
    }

    // Check for messages used in local parameters
    const int paramLocalCount = SavedFile->m_IndexByClassId[CKCID_PARAMETERLOCAL].Size();
    for (int i = 0; i < paramLocalCount; ++i) {
        CKParameterLocal *param = (CKParameterLocal *) SavedFile->m_FileObjects[SavedFile->m_IndexByClassId[CKCID_PARAMETERLOCAL][i]].ObjPtr;
        if (param && param->GetType() == msgParamType) {
            int msgType;
            param->GetValue(&msgType);
            if (msgType >= 0) usedMessageTypes.Set(msgType);
        }
    }

    if (usedMessageTypes.BitSet() > 0) {
        CKStateChunk *chunk = CreateCKStateChunk(CKCID_MESSAGEMANAGER, SavedFile);
        chunk->StartWrite();
        chunk->WriteIdentifier(0x53);
        chunk->WriteInt(m_RegisteredMessageTypes.Size());
        for (int i = 0; i < m_RegisteredMessageTypes.Size(); ++i) {
            chunk->WriteString(usedMessageTypes.IsSet(i) ? m_RegisteredMessageTypes[i].Str() : (CKSTRING) "");
        }
        chunk->CloseChunk();
        return chunk;
    }
    return nullptr;
}

CKERROR CKMessageManager::LoadData(CKStateChunk *chunk, CKFile *LoadedFile) {
    if (!chunk) return CKERR_INVALIDPARAMETER;
    chunk->StartRead();
    if (!chunk->SeekIdentifier(0x53)) return CK_OK;

    const int typeCount = chunk->ReadInt();
    if (typeCount <= 0) return CK_OK;

    XArray<CKMessageType> typeIDs(typeCount);
    for (int i = 0; i < typeCount; ++i) {
        CKSTRING name = nullptr;
        chunk->ReadString(&name);
        typeIDs.PushBack(AddMessageType(name));
        delete[] name;
    }
    LoadedFile->RemapManagerInt(MESSAGE_MANAGER_GUID, typeIDs.Begin(), typeCount);
    return CK_OK;
}

CKERROR CKMessageManager::PreClearAll() {
    if (m_MsgWaitingList) {
        for (int i = 0; i < m_RegisteredMessageTypes.Size(); ++i) {
            CKWaitingObjectArray *waitingArray = m_MsgWaitingList[i];
            if (waitingArray) {
                delete waitingArray;
                m_MsgWaitingList[i] = nullptr;
            }
        }
        delete[] m_MsgWaitingList;
        m_MsgWaitingList = nullptr;
    }
    m_RegisteredMessageTypes.Clear();
    for (int i = 0; i < m_ReceivedMsgThisFrame.Size(); ++i) {
        CKMessage *msg = m_ReceivedMsgThisFrame[i];
        if (msg) {
            msg->Release();
            m_ReceivedMsgThisFrame[i] = nullptr;
        }
    }

    m_ReceivedMsgThisFrame.Resize(0);
    m_LastFrameObjects.Resize(0);
    return RegisterDefaultMessages();
}

CKERROR CKMessageManager::PostProcess() {
    // Clear last frame messages from all objects that received messages
    for (XObjectPointerArray::Iterator it = m_LastFrameObjects.Begin(); it != m_LastFrameObjects.End(); ++it) {
        CKBeObject *beObject = static_cast<CKBeObject *>(*it);
        if (beObject) {
            beObject->RemoveAllLastFrameMessage();
        }
    }

    // Clear the list of objects that received messages last frame (keep allocation)
    m_LastFrameObjects.Resize(0);

    // Get current scene for message processing
    CKScene *currentScene = m_Context->GetCurrentScene();
    CKObjectManager *objectManager = m_Context->m_ObjectManager;

    // Process all messages received this frame
    for (int msgIndex = 0; msgIndex < m_ReceivedMsgThisFrame.Size(); ++msgIndex) {
        CKMessage *message = m_ReceivedMsgThisFrame[msgIndex];
        if (!message)
            continue;

        CKMessageType messageType = message->GetMsgType();
        CK_MESSAGE_SENDINGTYPE sendingType = message->GetSendingType();
        CKObject *recipient = NULL;

        // Handle different sending types
        if (sendingType == CK_MESSAGE_BROADCAST) {
            // Broadcast message to all objects of specified class hierarchy
            CK_CLASSID broadcastClassId = message->m_BroadcastCid;

            // Iterate through all registered class IDs
            for (CK_CLASSID classId = CKCID_OBJECT; classId < g_MaxClassID; ++classId) {
                if (CKIsChildClassOf(classId, broadcastClassId)) {
                    int objectCount = objectManager->GetObjectsCountByClassID(classId);
                    if (objectCount > 0) {
                        CK_ID *objectsList = objectManager->GetObjectsListByClassID(classId);
                        if (objectsList) {
                            for (int i = 0; i < objectCount; ++i) {
                                CKObject *obj = m_Context->GetObject(objectsList[i]);
                                if (obj) {
                                    AddMessageToObject(obj, message, currentScene, FALSE);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Single or Group message
            recipient = message->GetRecipient();
            if (recipient) {
                AddMessageToObject(recipient, message, currentScene, TRUE);
            }
        }

        // Process waiting behaviors for this message type
        if (messageType >= 0 && messageType < GetMessageTypeCount()) {
            CKWaitingObjectArray *waitingList = m_MsgWaitingList[messageType];
            if (waitingList) {
                for (CKWaitingObjectArray::Iterator it = waitingList->Begin(); it != waitingList->End();) {
                    CKBeObject *waitingBeObject = it->m_BeObject;
                    if (!waitingBeObject) {
                        ++it;
                        continue;
                    }

                    CKBOOL shouldActivate = FALSE;

                    // Check if this waiting object should receive the message
                    if (sendingType == CK_MESSAGE_BROADCAST) {
                        // For broadcast, check if waiting object is of the right class
                        if (CKIsChildClassOf(waitingBeObject, message->m_BroadcastCid)) {
                            shouldActivate = TRUE;
                        }
                    } else if (sendingType == CK_MESSAGE_SINGLE) {
                        // For single message, check if it's the same object
                        if (waitingBeObject == recipient) {
                            shouldActivate = TRUE;
                        }
                    } else if (sendingType == CK_MESSAGE_GROUP) {
                        // For group message, check various conditions
                        if (waitingBeObject == recipient) {
                            shouldActivate = TRUE;
                        } else if (CKIsChildClassOf(recipient, CKCID_GROUP)) {
                            // If recipient is a group, check if waiting object is in that group
                            CKGroup *group = static_cast<CKGroup *>(recipient);
                            if (waitingBeObject->IsInGroup(group)) {
                                shouldActivate = TRUE;
                            }
                        } else if (CKIsChildClassOf(recipient, CKCID_BODYPART)) {
                            // If recipient is a body part, check if waiting object is the character
                            CKBodyPart *bodyPart = static_cast<CKBodyPart *>(recipient);
                            CKCharacter *character = bodyPart->GetCharacter();
                            if (waitingBeObject == character) {
                                shouldActivate = TRUE;
                            }
                        }
                    }

                    if (shouldActivate) {
                        // Add message to the waiting object
                        waitingBeObject->AddLastFrameMessage(message);
                        m_LastFrameObjects.AddIfNotHere(waitingBeObject);

                        // Activate the behavior output if specified
                        if (it->m_Output) {
                            if (it->m_Behavior->GetFlags() & CKBEHAVIOR_WAITSFORMESSAGE) {
                                it->m_Output->Activate();
                            }
                        }

                        // Mark behavior for processing
                        if (it->m_Behavior) {
                            it->m_Behavior->ModifyFlags(0, CKBEHAVIOR_WAITSFORMESSAGE);
                        }

                        // Remove from waiting list (behavior waits for one message only)
                        it = waitingList->Remove(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
    }

    // Release all processed messages and keep allocation
    for (int i = 0; i < m_ReceivedMsgThisFrame.Size(); ++i) {
        CKMessage *message = m_ReceivedMsgThisFrame[i];
        if (message) {
            message->Release();
            m_ReceivedMsgThisFrame[i] = nullptr;
        }
    }

    m_ReceivedMsgThisFrame.Resize(0);

    return CK_OK;
}

CKERROR CKMessageManager::OnCKReset() {
    for (int i = 0; i < m_RegisteredMessageTypes.Size(); ++i) {
        if (m_MsgWaitingList[i]) {
            m_MsgWaitingList[i]->Clear();
        }
    }

    // Discards any pending messages here; keep allocation and just clear the list.
    for (int i = 0; i < m_ReceivedMsgThisFrame.Size(); ++i)
        m_ReceivedMsgThisFrame[i] = nullptr;
    m_ReceivedMsgThisFrame.Resize(0);

    for (int i = 0; i < m_LastFrameObjects.Size(); ++i) {
        CKBeObject *obj = (CKBeObject *)m_LastFrameObjects[i];
        if (obj) {
            obj->RemoveAllLastFrameMessage();
        }
    }
    m_LastFrameObjects.Clear();

    return CK_OK;
}

CKERROR CKMessageManager::SequenceToBeDeleted(CK_ID *objids, int count) {
    for (int i = 0; i < m_RegisteredMessageTypes.Size(); ++i) {
        CKWaitingObjectArray *waitingList = m_MsgWaitingList[i];
        if (!waitingList) continue;

        for (auto it = waitingList->Begin(); it != waitingList->End();) {
            CKWaitingObject &wo = *it;

            CKBOOL shouldRemove = FALSE;
            if (wo.m_BeObject && wo.m_BeObject->IsToBeDeleted())
                shouldRemove = TRUE;
            if (wo.m_Output && wo.m_Output->IsToBeDeleted())
                shouldRemove = TRUE;

            if (shouldRemove) {
                it = waitingList->Remove(it);
            } else {
                ++it;
            }
        }
    }

    m_LastFrameObjects.Check();
    return CK_OK;
}

CKMessageManager::~CKMessageManager() {
    if (m_MsgWaitingList) {
        for (int i = 0; i < m_RegisteredMessageTypes.Size(); ++i) {
            CKWaitingObjectArray *waitingList = m_MsgWaitingList[i];
            if (waitingList) {
                waitingList->Clear();
                delete waitingList;
            }
        }

        delete[] m_MsgWaitingList;
        m_MsgWaitingList = nullptr;
    }

    for (int i = 0; i < m_ReceivedMsgThisFrame.Size(); ++i) {
        CKMessage *msg = m_ReceivedMsgThisFrame[i];
        if (msg) {
            msg->Release();
        }
    }
    m_ReceivedMsgThisFrame.Clear();
    m_LastFrameObjects.Clear();
    m_RegisteredMessageTypes.Clear();
}

CKMessageManager::CKMessageManager(CKContext *Context)
    : CKBaseManager(Context, MESSAGE_MANAGER_GUID, (CKSTRING)"Message Manager"),
      m_ReceivedMsgThisFrame(100) {
    m_MsgWaitingList = nullptr;
    RegisterDefaultMessages();
    Context->RegisterNewManager(this);
}

void CKMessageManager::AddMessageToObject(CKObject *obj, CKMessage *msg, CKScene *currentscene, CKBOOL recurse) {
    if (!obj || !msg) return;

    if (CKIsChildClassOf(obj, CKCID_BEOBJECT)) {
        CKBeObject *beo = (CKBeObject *) obj;
        if (beo->IsWaitingForMessages() && currentscene->IsObjectActive(beo)) {
            beo->AddLastFrameMessage(msg);
            m_LastFrameObjects.AddIfNotHere(beo);
        }

        if (recurse) {
            if (obj->GetClassID() == CKCID_BODYPART) {
                CKBodyPart *bodyPart = (CKBodyPart *) obj;
                CKCharacter *character = bodyPart->GetCharacter();
                if (character) {
                    AddMessageToObject(character, msg, currentscene, TRUE);
                }
            }
            if (msg->GetSendingType() == CK_MESSAGE_GROUP && obj->GetClassID() == CKCID_GROUP) {
                CKGroup *group = (CKGroup *) obj;
                for (int i = 0; i < group->GetObjectCount(); ++i) {
                    AddMessageToObject(group->GetObject(i), msg, currentscene, TRUE);
                }
            }
        }
    }
}

CKMessage *CKMessageManager::CreateMessageSingle(CKMessageType type, CKBeObject *dest, CKBeObject *sender) {
    CKMessage *msg = new CKMessage(m_Context);
    if (msg) {
        msg->m_MessageType = type;
        msg->m_Sender = sender ? sender->GetID() : 0;
        msg->m_Recipient = dest ? dest->GetID() : 0;
        msg->m_SendingType = CK_MESSAGE_SINGLE;
    }
    return msg;
}

CKMessage *CKMessageManager::CreateMessageGroup(CKMessageType type, CKGroup *group, CKBeObject *sender) {
    CKMessage *msg = new CKMessage(m_Context);
    if (msg) {
        msg->m_MessageType = type;
        msg->m_Sender = sender ? sender->GetID() : 0;
        msg->m_Recipient = group ? group->GetID() : 0;
        msg->m_SendingType = CK_MESSAGE_GROUP;
    }
    return msg;
}

CKMessage *CKMessageManager::CreateMessageBroadcast(CKMessageType type, CK_CLASSID objType, CKBeObject *sender) {
    CKMessage *msg = new CKMessage(m_Context);
    if (msg) {
        msg->m_MessageType = type;
        msg->m_Sender = sender ? sender->GetID() : 0;
        msg->m_SendingType = CK_MESSAGE_BROADCAST;
        msg->SetBroadcastObjectType(objType);
    }
    return msg;
}
