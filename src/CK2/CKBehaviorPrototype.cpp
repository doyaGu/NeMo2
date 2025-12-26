#include "CKBehaviorPrototype.h"

CKPARAMETER_DESC::CKPARAMETER_DESC() {
    Type = 0;
    Name = NULL;
    DefaultValueString = NULL;
    DefaultValue = NULL;
    DefaultValueSize = 0;
    Guid = CKGUID();
    Owner = -1;
}

CKPARAMETER_DESC::CKPARAMETER_DESC(const CKPARAMETER_DESC &d) {
    Type = d.Type;
    Name = d.Name ? MAKESTRING(d.Name) : NULL;
    DefaultValueString = d.DefaultValueString ? MAKESTRING(d.DefaultValueString) : NULL;
    DefaultValue = NULL;
    DefaultValueSize = 0;
    Guid = d.Guid;
    Owner = d.Owner;
    if (d.DefaultValue && d.DefaultValueSize) {
        DefaultValue = new CKBYTE[d.DefaultValueSize];
        DefaultValueSize = d.DefaultValueSize;
        memcpy(DefaultValue, d.DefaultValue, DefaultValueSize);
    }
}

CKPARAMETER_DESC::~CKPARAMETER_DESC() {
    DELETESTRING(Name);
    DELETESTRING(DefaultValueString);
    delete[] DefaultValue;
}

CKPARAMETER_DESC &CKPARAMETER_DESC::operator=(const CKPARAMETER_DESC &d) {
    if (this == &d)
        return *this;

    DELETESTRING(Name);
    DELETESTRING(DefaultValueString);
    delete[] DefaultValue;

    Type = d.Type;
    Name = d.Name ? MAKESTRING(d.Name) : NULL;
    DefaultValueString = d.DefaultValueString ? MAKESTRING(d.DefaultValueString) : NULL;
    DefaultValue = NULL;
    DefaultValueSize = 0;
    Guid = d.Guid;
    Owner = d.Owner;
    if (d.DefaultValue && d.DefaultValueSize) {
        DefaultValue = new CKBYTE[d.DefaultValueSize];
        DefaultValueSize = d.DefaultValueSize;
        memcpy(DefaultValue, d.DefaultValue, DefaultValueSize);
    }

    return *this;
}

CKBEHAVIORIO_DESC::CKBEHAVIORIO_DESC() : Name(0) {}

CKBEHAVIORIO_DESC::~CKBEHAVIORIO_DESC() {
    DELETESTRING(Name);
}

int CKBehaviorPrototype::DeclareInput(CKSTRING name) {
    CKBEHAVIORIO_DESC **newList = new CKBEHAVIORIO_DESC *[m_InIOCount + 1];

    if (m_InIOList) {
        memcpy(newList, m_InIOList, sizeof(CKBEHAVIORIO_DESC *) * m_InIOCount);
        delete[] m_InIOList;
    }

    m_InIOList = newList;
    m_InIOList[m_InIOCount] = new CKBEHAVIORIO_DESC();

    if (name) {
        m_InIOList[m_InIOCount]->Name = MAKESTRING(name);
    } else {
        m_InIOList[m_InIOCount]->Name = nullptr;
    }

    m_InIOList[m_InIOCount]->Flags = CK_BEHAVIORIO_IN;

    return m_InIOCount++;
}

int CKBehaviorPrototype::DeclareOutput(CKSTRING name) {
    CKBEHAVIORIO_DESC **newList = new CKBEHAVIORIO_DESC *[m_OutIOCount + 1];

    if (m_OutIOList) {
        memcpy(newList, m_OutIOList, sizeof(CKBEHAVIORIO_DESC *) * m_OutIOCount);
        delete[] m_OutIOList;
    }
    m_OutIOList = newList;

    m_OutIOList[m_OutIOCount] = new CKBEHAVIORIO_DESC();

    if (name) {
        m_OutIOList[m_OutIOCount]->Name = MAKESTRING(name);
    } else {
        m_OutIOList[m_OutIOCount]->Name = nullptr;
    }

    m_OutIOList[m_OutIOCount]->Flags = CK_BEHAVIORIO_OUT;

    return m_OutIOCount++;
}

int CKBehaviorPrototype::DeclareInParameter(CKSTRING name, CKGUID guid_type, CKSTRING defaultval) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_InParameterCount + 1];

    if (m_InParameterList) {
        memcpy(newList, m_InParameterList, sizeof(CKPARAMETER_DESC *) * m_InParameterCount);
        delete[] m_InParameterList;
    }
    m_InParameterList = newList;

    m_InParameterList[m_InParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_InParameterList[m_InParameterCount]->Name = MAKESTRING(name);

    m_InParameterList[m_InParameterCount]->Guid = guid_type;
    m_InParameterList[m_InParameterCount]->Owner = -1;
    m_InParameterList[m_InParameterCount]->Type = 1;

    if (defaultval)
        m_InParameterList[m_InParameterCount]->DefaultValueString = MAKESTRING(defaultval);

    return m_InParameterCount++;
}

int CKBehaviorPrototype::DeclareInParameter(CKSTRING name, CKGUID guid_type, void *defaultval, int valsize) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_InParameterCount + 1];

    if (m_InParameterList) {
        memcpy(newList, m_InParameterList, sizeof(CKPARAMETER_DESC *) * m_InParameterCount);
        delete[] m_InParameterList;
    }
    m_InParameterList = newList;

    m_InParameterList[m_InParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_InParameterList[m_InParameterCount]->Name = MAKESTRING(name);

    m_InParameterList[m_InParameterCount]->Guid = guid_type;
    m_InParameterList[m_InParameterCount]->Owner = -1;
    m_InParameterList[m_InParameterCount]->Type = 1;

    if (defaultval && valsize > 0) {
        m_InParameterList[m_InParameterCount]->DefaultValue = new CKBYTE[valsize];
        m_InParameterList[m_InParameterCount]->DefaultValueSize = valsize;
        memcpy(m_InParameterList[m_InParameterCount]->DefaultValue, defaultval, valsize);
    }

    return m_InParameterCount++;
}

int CKBehaviorPrototype::DeclareOutParameter(CKSTRING name, CKGUID guid_type, CKSTRING defaultval) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_OutParameterCount + 1];

    if (m_OutParameterList) {
        memcpy(newList, m_OutParameterList, sizeof(CKPARAMETER_DESC *) * m_OutParameterCount);
        delete[] m_OutParameterList;
    }
    m_OutParameterList = newList;

    m_OutParameterList[m_OutParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_OutParameterList[m_OutParameterCount]->Name = MAKESTRING(name);

    m_OutParameterList[m_OutParameterCount]->Guid = guid_type;
    m_OutParameterList[m_OutParameterCount]->Owner = -1;
    m_OutParameterList[m_OutParameterCount]->Type = 2;

    if (defaultval)
        m_OutParameterList[m_OutParameterCount]->DefaultValueString = MAKESTRING(defaultval);

    return m_OutParameterCount++;
}

int CKBehaviorPrototype::DeclareOutParameter(CKSTRING name, CKGUID guid_type, void *defaultval, int valsize) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_OutParameterCount + 1];

    if (m_OutParameterList) {
        memcpy(newList, m_OutParameterList, sizeof(CKPARAMETER_DESC *) * m_OutParameterCount);
        delete[] m_OutParameterList;
    }
    m_OutParameterList = newList;

    m_OutParameterList[m_OutParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_OutParameterList[m_OutParameterCount]->Name = MAKESTRING(name);

    m_OutParameterList[m_OutParameterCount]->Guid = guid_type;
    m_OutParameterList[m_OutParameterCount]->Type = 2;
    m_OutParameterList[m_OutParameterCount]->Owner = -1;

    if (defaultval && valsize > 0) {
        m_OutParameterList[m_OutParameterCount]->DefaultValue = new CKBYTE[valsize];
        m_OutParameterList[m_OutParameterCount]->DefaultValueSize = valsize;
        memcpy(m_OutParameterList[m_OutParameterCount]->DefaultValue, defaultval, valsize);
    }

    return m_OutParameterCount++;
}

int CKBehaviorPrototype::DeclareLocalParameter(CKSTRING name, CKGUID guid_type, CKSTRING defaultval) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_LocalParameterCount + 1];

    if (m_LocalParameterList) {
        memcpy(newList, m_LocalParameterList, sizeof(CKPARAMETER_DESC *) * m_LocalParameterCount);
        delete[] m_LocalParameterList;
    }
    m_LocalParameterList = newList;

    m_LocalParameterList[m_LocalParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_LocalParameterList[m_LocalParameterCount]->Name = MAKESTRING(name);

    m_LocalParameterList[m_LocalParameterCount]->Guid = guid_type;
    m_LocalParameterList[m_LocalParameterCount]->Type = 0;

    if (defaultval)
        m_LocalParameterList[m_LocalParameterCount]->DefaultValueString = MAKESTRING(defaultval);

    return m_LocalParameterCount++;
}

int CKBehaviorPrototype::DeclareLocalParameter(CKSTRING name, CKGUID guid_type, void *defaultval, int valsize) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_LocalParameterCount + 1];

    if (m_LocalParameterList) {
        memcpy(newList, m_LocalParameterList, sizeof(CKPARAMETER_DESC *) * m_LocalParameterCount);
        delete[] m_LocalParameterList;
    }
    m_LocalParameterList = newList;

    m_LocalParameterList[m_LocalParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_LocalParameterList[m_LocalParameterCount]->Name = MAKESTRING(name);

    m_LocalParameterList[m_LocalParameterCount]->Guid = guid_type;
    m_LocalParameterList[m_LocalParameterCount]->Type = 0;

    if (defaultval && valsize > 0) {
        m_LocalParameterList[m_LocalParameterCount]->DefaultValue = new CKBYTE[valsize];
        m_LocalParameterList[m_LocalParameterCount]->DefaultValueSize = valsize;
        memcpy(m_LocalParameterList[m_LocalParameterCount]->DefaultValue, defaultval, valsize);
    }

    return m_LocalParameterCount++;
}

int CKBehaviorPrototype::DeclareSetting(CKSTRING name, CKGUID guid_type, CKSTRING defaultval) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_LocalParameterCount + 1];

    if (m_LocalParameterList) {
        memcpy(newList, m_LocalParameterList, sizeof(CKPARAMETER_DESC *) * m_LocalParameterCount);
        delete[] m_LocalParameterList;
    }
    m_LocalParameterList = newList;

    m_LocalParameterList[m_LocalParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_LocalParameterList[m_LocalParameterCount]->Name = MAKESTRING(name);

    m_LocalParameterList[m_LocalParameterCount]->Guid = guid_type;
    m_LocalParameterList[m_LocalParameterCount]->Type = 3;

    if (defaultval)
        m_LocalParameterList[m_LocalParameterCount]->DefaultValueString = MAKESTRING(defaultval);

    return m_LocalParameterCount++;
}

int CKBehaviorPrototype::DeclareSetting(CKSTRING name, CKGUID guid_type, void *defaultval, int valsize) {
    CKPARAMETER_DESC **newList = new CKPARAMETER_DESC *[m_LocalParameterCount + 1];

    if (m_LocalParameterList) {
        memcpy(newList, m_LocalParameterList, sizeof(CKPARAMETER_DESC *) * m_LocalParameterCount);
        delete[] m_LocalParameterList;
    }
    m_LocalParameterList = newList;

    m_LocalParameterList[m_LocalParameterCount] = new CKPARAMETER_DESC();

    if (name)
        m_LocalParameterList[m_LocalParameterCount]->Name = MAKESTRING(name);

    m_LocalParameterList[m_LocalParameterCount]->Guid = guid_type;
    m_LocalParameterList[m_LocalParameterCount]->Type = 3;

    if (defaultval && valsize > 0) {
        m_LocalParameterList[m_LocalParameterCount]->DefaultValue = new CKBYTE[valsize];
        m_LocalParameterList[m_LocalParameterCount]->DefaultValueSize = valsize;
        memcpy(m_LocalParameterList[m_LocalParameterCount]->DefaultValue, defaultval, valsize);
    }

    return m_LocalParameterCount++;
}

void CKBehaviorPrototype::SetGuid(CKGUID guid) {
    m_Guid = guid;
}

CKGUID CKBehaviorPrototype::GetGuid() {
    return m_Guid;
}

void CKBehaviorPrototype::SetFlags(CK_BEHAVIORPROTOTYPE_FLAGS flags) {
    m_bFlags = flags;
}

CK_BEHAVIORPROTOTYPE_FLAGS CKBehaviorPrototype::GetFlags() {
    return m_bFlags;
}

void CKBehaviorPrototype::SetApplyToClassID(CK_CLASSID cid) {
    m_ApplyTo = cid;
}

CK_CLASSID CKBehaviorPrototype::GetApplyToClassID() {
    return m_ApplyTo;
}

void CKBehaviorPrototype::SetFunction(CKBEHAVIORFCT fct) {
    m_FctPtr = fct;
}

CKBEHAVIORFCT CKBehaviorPrototype::GetFunction() {
    return m_FctPtr;
}

void CKBehaviorPrototype::SetBehaviorCallbackFct(CKBEHAVIORCALLBACKFCT fct, CKDWORD CallbackMask, void *param) {
    m_CallbackFctPtr = fct;
    m_CallBackMask = CallbackMask;
    m_CallBackParam = param;
}

CKBEHAVIORCALLBACKFCT CKBehaviorPrototype::GetBehaviorCallbackFct() {
    return m_CallbackFctPtr;
}

void CKBehaviorPrototype::SetBehaviorFlags(CK_BEHAVIOR_FLAGS flags) {
    m_BehaviorFlags = flags;
}

CK_BEHAVIOR_FLAGS CKBehaviorPrototype::GetBehaviorFlags() {
    return m_BehaviorFlags;
}

CKObjectDeclaration *CKBehaviorPrototype::GetSoureObjectDeclaration() {
    return m_SourceObjectDeclaration;
}

int CKBehaviorPrototype::GetInIOIndex(CKSTRING name) {
    if (!name)
        return -1;
    for (int i = 0; i < m_InIOCount; ++i) {
        if (strcmp(m_InIOList[i]->Name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int CKBehaviorPrototype::GetOutIOIndex(CKSTRING name) {
    if (!name)
        return -1;
    for (int i = 0; i < m_OutIOCount; ++i) {
        if (strcmp(m_OutIOList[i]->Name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int CKBehaviorPrototype::GetInParamIndex(CKSTRING name) {
    if (!name)
        return -1;
    for (int i = 0; i < m_InParameterCount; ++i) {
        if (strcmp(m_InParameterList[i]->Name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int CKBehaviorPrototype::GetOutParamIndex(CKSTRING name) {
    if (!name)
        return -1;
    for (int i = 0; i < m_OutParameterCount; ++i) {
        if (strcmp(m_OutParameterList[i]->Name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int CKBehaviorPrototype::GetLocalParamIndex(CKSTRING name) {
    if (!name)
        return -1;

    for (int i = 0; i < m_LocalParameterCount; ++i) {
        if (strcmp(m_LocalParameterList[i]->Name, name) == 0) {
            return i;
        }
    }
    return -1;
}

CKSTRING CKBehaviorPrototype::GetInIOName(int idx) {
    if (idx < m_InIOCount) {
        return m_InIOList[idx]->Name;
    }
    return nullptr;
}

CKSTRING CKBehaviorPrototype::GetOutIOIndex(int idx) {
    if (idx < m_OutIOCount) {
        return m_OutIOList[idx]->Name;
    }
    return nullptr;
}

CKSTRING CKBehaviorPrototype::GetInParamIndex(int idx) {
    if (idx < m_InParameterCount) {
        return m_InParameterList[idx]->Name;
    }
    return nullptr;
}

CKSTRING CKBehaviorPrototype::GetOutParamIndex(int idx) {
    if (idx < m_OutParameterCount) {
        return m_OutParameterList[idx]->Name;
    }
    return nullptr;
}

CKSTRING CKBehaviorPrototype::GetLocalParamIndex(int idx) {
    if (idx < m_LocalParameterCount) {
        return m_LocalParameterList[idx]->Name;
    }
    return nullptr;
}

CKBehaviorPrototype::~CKBehaviorPrototype() {
    for (int i = 0; i < m_InIOCount; ++i) {
        if (m_InIOList && m_InIOList[i]) {
            delete m_InIOList[i];
        }
    }
    delete[] m_InIOList;

    for (int i = 0; i < m_OutIOCount; ++i) {
        if (m_OutIOList && m_OutIOList[i]) {
            delete m_OutIOList[i];
        }
    }
    delete[] m_OutIOList;

    for (int i = 0; i < m_InParameterCount; ++i) {
        delete m_InParameterList[i];
    }
    delete[] m_InParameterList;

    for (int i = 0; i < m_OutParameterCount; ++i) {
        delete m_OutParameterList[i];
    }
    delete[] m_OutParameterList;

    for (int i = 0; i < m_LocalParameterCount; ++i) {
        delete m_LocalParameterList[i];
    }
    delete[] m_LocalParameterList;
}

CKBehaviorPrototype::CKBehaviorPrototype(CKSTRING Name)
    : m_Guid(),
      m_bFlags(CK_BEHAVIORPROTOTYPE_NORMAL),
      m_BehaviorFlags(CKBEHAVIOR_NONE),
      m_ApplyTo(0),
      m_FctPtr(nullptr),
      m_CallbackFctPtr(nullptr),
      m_CallBackMask(CKCB_BEHAVIORALL),
      m_InIOCount(0),
      m_OutIOCount(0),
      m_InParameterCount(0),
      m_LocalParameterCount(0),
      m_OutParameterCount(0),
      m_InIOList(nullptr),
      m_OutIOList(nullptr),
      m_InParameterList(nullptr),
      m_OutParameterList(nullptr),
      m_LocalParameterList(nullptr),
      m_BehaviorType(CKBEHAVIORTYPE_BASE),
      m_CallBackParam(nullptr),
      m_SourceObjectDeclaration(nullptr),
      m_Name(Name ? Name : "") {}
