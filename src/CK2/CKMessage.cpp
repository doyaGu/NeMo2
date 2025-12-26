#include "CKMessage.h"

#include "CKParameter.h"

CKERROR CKMessage::SetBroadcastObjectType(CK_CLASSID type) {
    if (!CKIsChildClassOf(type, CKCID_BEOBJECT))
        return CKERR_INVALIDPARAMETER;
    m_BroadcastCid = type;
    return CK_OK;
}

CKERROR CKMessage::AddParameter(CKParameter *param, CKBOOL DeleteParameterWithMessage) {
    if (!param)
        return CKERR_INVALIDPARAMETER;

    if (!m_Parameters)
        m_Parameters = new XObjectArray();

    param->MessageDeleteAfterUse(DeleteParameterWithMessage);
    m_Parameters->PushBack(param->GetID());
    return CK_OK;
}

CKERROR CKMessage::RemoveParameter(CKParameter *param) {
    if (!param || !m_Parameters)
        return CKERR_INVALIDPARAMETER;

    m_Parameters->RemoveObject(param);
    if (m_Parameters->Size() == 0) {
        delete m_Parameters;
        m_Parameters = nullptr;
    }

    return CK_OK;
}

int CKMessage::GetParameterCount() {
    return m_Parameters ? m_Parameters->Size() : 0;
}

CKParameter *CKMessage::GetParameter(int pos) {
    if (!m_Parameters)
        return nullptr;
    return (CKParameter *) m_Context->GetObject((*m_Parameters)[pos]);
}

CKMessage::CKMessage(CKContext *context) {
    m_MessageType = -1;
    m_SendingType = CK_MESSAGE_SINGLE;
    m_Sender = 0;
    m_Parameters = nullptr;
    m_RefCount = 1;
    m_Context = context;
    m_BroadcastCid = 0;
}

CKMessage::~CKMessage() {
    if (m_Parameters) {
        for (XObjectArray::Iterator it = m_Parameters->Begin(); it != m_Parameters->End(); ++it) {
            CKObject *obj = m_Context->GetObject(*it);
            if (obj && (obj->GetObjectFlags() & CK_PARAMETEROUT_DELETEAFTERUSE)) {
                CKDestroyObject(obj, CK_DESTROY_TEMPOBJECT);
            }
        }
        delete m_Parameters;
    }
}

int CKMessage::AddRef() {
    return ++m_RefCount;
}

int CKMessage::Release() {
    --m_RefCount;
    if (m_RefCount <= 0) {
        delete this;
        return 0;
    }
    return m_RefCount;
}
