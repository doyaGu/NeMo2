#include "CKObjectDeclaration.h"

void CKObjectDeclaration::SetDescription(CKSTRING Description) {
    delete[] m_Description;
    m_Description = CKStrdup(Description);
}

CKSTRING CKObjectDeclaration::GetDescription() {
    return m_Description;
}

void CKObjectDeclaration::SetGuid(CKGUID guid) {
    m_Guid = guid;
}

CKGUID CKObjectDeclaration::GetGuid() {
    return m_Guid;
}

void CKObjectDeclaration::SetType(int type) {
    m_Type = type;
}

int CKObjectDeclaration::GetType() {
    return m_Type;
}

void CKObjectDeclaration::NeedManager(CKGUID Manager) {
    CKGUID *it;
    for (it = m_ManagersGuid.Begin(); it != m_ManagersGuid.End(); ++it) {
        if (*it == Manager)
            break;
    }
    if (it == m_ManagersGuid.End())
        m_ManagersGuid.PushBack(Manager);
}

void CKObjectDeclaration::SetCreationFunction(CKDLL_CREATEPROTOFUNCTION f) {
    m_CreationFunction = f;
}

CKDLL_CREATEPROTOFUNCTION CKObjectDeclaration::GetCreationFunction() {
    return m_CreationFunction;
}

void CKObjectDeclaration::SetAuthorGuid(CKGUID guid) {
    m_AuthorGuid = guid;
}

CKGUID CKObjectDeclaration::GetAuthorGuid() {
    return m_AuthorGuid;
}

void CKObjectDeclaration::SetAuthorName(CKSTRING Name) {
    delete[] m_AuthorName;
    m_AuthorName = CKStrdup(Name);
}

CKSTRING CKObjectDeclaration::GetAuthorName() {
    return m_AuthorName;
}

void CKObjectDeclaration::SetVersion(CKDWORD verion) {
    m_Version = verion;
}

CKDWORD CKObjectDeclaration::GetVersion() {
    return m_Version;
}

void CKObjectDeclaration::SetCompatibleClassId(CK_CLASSID id) {
    m_CompatibleClassID = id;
}

CK_CLASSID CKObjectDeclaration::GetCompatibleClassId() {
    return m_CompatibleClassID;
}

void CKObjectDeclaration::SetCategory(CKSTRING cat) {
    delete[] m_Category;
    m_Category = CKStrdup(cat);
}

CKSTRING CKObjectDeclaration::GetCategory() {
    return m_Category;
}

CKObjectDeclaration::CKObjectDeclaration(CKSTRING Name) {
    m_Version = -1;
    m_Description = nullptr;
    m_Guid = CKGUID();
    m_Type = CKDLL_BEHAVIORPROTOTYPE;
    m_CreationFunction = nullptr;
    m_AuthorGuid = CKGUID();
    m_AuthorName = nullptr;
    m_Proto = nullptr;
    m_CompatibleClassID = CKCID_OBJECT;
    m_Category = nullptr;
    m_Name = Name;
    m_PluginIndex = -1;
}

CKObjectDeclaration::~CKObjectDeclaration() {
    delete[] m_Description;
    delete[] m_AuthorName;
    delete[] m_Category;
}
