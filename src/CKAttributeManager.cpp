#include "CKAttributeManager.h"

#include "CKContext.h"
#include "CKFile.h"
#include "CKStateChunk.h"
#include "CKPathManager.h"
#include "CKPluginManager.h"
#include "CKBeObject.h"
#include "CKParameterManager.h"

extern CKPluginManager g_ThePluginManager;
extern CKPluginEntry *g_TheCurrentPluginEntry;

CKAttributeType CKAttributeManager::RegisterNewAttributeType(CKSTRING Name, CKGUID ParameterType,
                                                             CK_CLASSID CompatibleCid, CK_ATTRIBUT_FLAGS flags) {
    if (!Name)
        return -1;

    char secureName[64];
    strncpy(secureName, Name, 63);
    secureName[63] = '\0';

    int freeSlot = -1;
    for (CKAttributeType i = 0; i < m_AttributeInfoCount; ++i) {
        if (!m_AttributeInfos[i]) {
            if (freeSlot == -1) freeSlot = i;
            continue;
        }

        if (strcmp(secureName, m_AttributeInfos[i]->Name) == 0) {
            return i; // Return existing index
        }
    }

    if (freeSlot == -1) {
        CKAttributeDesc **newInfos = new CKAttributeDesc *[m_AttributeInfoCount + 1];
        if (m_AttributeInfos) {
            memcpy(newInfos, m_AttributeInfos, sizeof(CKAttributeDesc *) * m_AttributeInfoCount);
            delete[] m_AttributeInfos;
        }
        freeSlot = m_AttributeInfoCount;
        m_AttributeInfos = newInfos;
        m_AttributeInfoCount++;
    }

    CKAttributeDesc *desc = new CKAttributeDesc();
    memset(desc, 0, sizeof(CKAttributeDesc));

    strcpy(desc->Name, secureName);
    desc->ParameterType = ParameterType;
    desc->CompatibleCid = CompatibleCid;
    desc->AttributeCategory = -1;
    desc->Flags = flags;

    if (!(desc->Flags & (CK_ATTRIBUT_USER | CK_ATTRIBUT_SYSTEM))) {
        desc->Flags |= CK_ATTRIBUT_USER;
    }

    if (!g_TheCurrentPluginEntry) {
        CKBaseManager *man = m_Context->m_CurrentManager;
        if (man && man != this) {
            g_ThePluginManager.FindComponent(man->GetGuid(), CKPLUGIN_MANAGER_DLL);
        }
    }

    desc->CreatorDll = nullptr;

    m_AttributeInfos[freeSlot] = desc;
    return freeSlot;
}

void CKAttributeManager::UnRegisterAttribute(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount)
        return;

    CKAttributeDesc *attrDesc = m_AttributeInfos[AttribType];
    if (!attrDesc)
        return;

    // Remove from front of list - RemoveAttribute modifies the list so we can't use iterator
    while (attrDesc->GlobalAttributeList.Size() > 0) {
        CKBeObject *beo = (CKBeObject *)attrDesc->GlobalAttributeList[0];
        if (beo) {
            beo->RemoveAttribute(AttribType);
        }
        attrDesc = m_AttributeInfos[AttribType];
        if (!attrDesc)
            return;
    }

    delete[] attrDesc->DefaultValue;
    attrDesc->DefaultValue = nullptr;

    delete attrDesc;
    m_AttributeInfos[AttribType] = nullptr;
}

void CKAttributeManager::UnRegisterAttribute(CKSTRING atname) {
    CKAttributeType type = GetAttributeTypeByName(atname);
    if (type >= 0) {
        UnRegisterAttribute(type);
    }
}

CKSTRING CKAttributeManager::GetAttributeNameByType(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return nullptr;

    if (!m_AttributeInfos[AttribType])
        return nullptr;

    return m_AttributeInfos[AttribType]->Name;
}

CKAttributeType CKAttributeManager::GetAttributeTypeByName(CKSTRING AttribName) {
    if (!AttribName || m_AttributeInfoCount == 0 || !m_AttributeInfos)
        return -1;

    char secureName[64];
    strncpy(secureName, AttribName, 63);
    secureName[63] = '\0';

    for (CKAttributeType i = 0; i < m_AttributeInfoCount; ++i) {
        if (!m_AttributeInfos[i])
            continue;

        if (strcmp(secureName, m_AttributeInfos[i]->Name) == 0) {
            return i;
        }
    }

    return -1;
}

void CKAttributeManager::SetAttributeNameByType(CKAttributeType AttribType, CKSTRING name) {
    if (!name || AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return;
    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return;

    strcpy(desc->Name, name);
}

int CKAttributeManager::GetAttributeCount() {
    return m_AttributeInfoCount;
}

CKGUID CKAttributeManager::GetAttributeParameterGUID(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return CKGUID();

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return CKGUID();

    return desc->ParameterType;
}

CKParameterType CKAttributeManager::GetAttributeParameterType(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return -1;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return -1;

    CKParameterManager *pm = m_Context->GetParameterManager();
    if (!pm)
        return -1;

    return pm->ParameterGuidToType(desc->ParameterType);
}

CK_CLASSID CKAttributeManager::GetAttributeCompatibleClassId(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return -1;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return -1;

    return desc->CompatibleCid;
}

CKBOOL CKAttributeManager::IsAttributeIndexValid(CKAttributeType index) {
    if (index < 0 || index >= m_AttributeInfoCount || !m_AttributeInfos)
        return FALSE;
    return m_AttributeInfos[index] != nullptr;
}

CKBOOL CKAttributeManager::IsCategoryIndexValid(CKAttributeCategory index) {
    if (index < 0 || index >= m_AttributeCategoryCount || !m_AttributeCategories)
        return FALSE;
    return m_AttributeCategories[index] != nullptr;
}

CK_ATTRIBUT_FLAGS CKAttributeManager::GetAttributeFlags(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return (CK_ATTRIBUT_FLAGS) 0;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return (CK_ATTRIBUT_FLAGS) 0;

    return (CK_ATTRIBUT_FLAGS) desc->Flags;
}

void CKAttributeManager::SetAttributeCallbackFunction(CKAttributeType AttribType, CKATTRIBUTECALLBACK fct, void *arg) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return;

    desc->CallbackFct = fct;
    desc->CallbackArg = arg;
}

void CKAttributeManager::SetAttributeDefaultValue(CKAttributeType AttribType, CKSTRING DefaultVal) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return;

    if (desc->DefaultValue) {
        delete[] desc->DefaultValue;
        desc->DefaultValue = nullptr;
    }

    if (DefaultVal) {
        desc->DefaultValue = new char[strlen(DefaultVal) + 1];
        strcpy(desc->DefaultValue, DefaultVal);
    }
}

CKSTRING CKAttributeManager::GetAttributeDefaultValue(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return nullptr;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return nullptr;

    return desc->DefaultValue;
}

const XObjectPointerArray &CKAttributeManager::GetAttributeListPtr(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return m_Context->m_GlobalAttributeList;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return m_Context->m_GlobalAttributeList;

    if (m_Context->GetCurrentScene()) {
        return desc->AttributeList;
    } else {
        return desc->GlobalAttributeList;
    }
}

const XObjectPointerArray &CKAttributeManager::GetGlobalAttributeListPtr(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return m_Context->m_GlobalAttributeList;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return m_Context->m_GlobalAttributeList;

    return desc->GlobalAttributeList;
}

const XObjectPointerArray &CKAttributeManager::FillListByAttributes(CKAttributeType *ListAttrib, int AttribCount) {
    m_AttributeList.Resize(0);
    if (!ListAttrib || AttribCount == 0 || !m_AttributeInfos)
        return m_AttributeList;

    if (m_Context->GetCurrentScene()) {
        for (CKAttributeType i = 0; i < AttribCount; ++i) {
            CKAttributeType attrType = ListAttrib[i];
            if (attrType >= 0 && attrType < m_AttributeInfoCount) {
                CKAttributeDesc *desc = m_AttributeInfos[attrType];
                if (desc) {
                    m_AttributeList += desc->AttributeList;
                }
            }
        }
    } else {
        for (CKAttributeType i = 0; i < AttribCount; ++i) {
            CKAttributeType attrType = ListAttrib[i];
            if (attrType >= 0 && attrType < m_AttributeInfoCount) {
                CKAttributeDesc *desc = m_AttributeInfos[attrType];
                if (desc) {
                    m_AttributeList += desc->GlobalAttributeList;
                }
            }
        }
    }

    return m_AttributeList;
}

const XObjectPointerArray &CKAttributeManager::FillListByGlobalAttributes(CKAttributeType *ListAttrib, int AttribCount) {
    m_AttributeList.Resize(0);
    if (!ListAttrib)
        return m_AttributeList;

    for (CKAttributeType i = 0; i < AttribCount; ++i) {
        CKAttributeType attrType = ListAttrib[i];
        if (attrType >= 0 && attrType < m_AttributeInfoCount) {
            CKAttributeDesc *desc = m_AttributeInfos[attrType];
            if (desc) {
                m_AttributeList += desc->GlobalAttributeList;
            }
        }
    }

    return m_AttributeList;
}

int CKAttributeManager::GetCategoriesCount() {
    return m_AttributeCategoryCount;
}

CKSTRING CKAttributeManager::GetCategoryName(CKAttributeCategory index) {
    if (index < 0 || index >= m_AttributeCategoryCount || !m_AttributeCategories)
        return nullptr;
    CKAttributeCategoryDesc *desc = m_AttributeCategories[index];
    if (!desc)
        return nullptr;
    return desc->Name;
}

CKAttributeCategory CKAttributeManager::GetCategoryByName(CKSTRING Name) {
    if (!Name)
        return -1;
    if (m_AttributeCategoryCount <= 0)
        return -1;

    for (int i = 0; i < m_AttributeCategoryCount; ++i) {
        CKAttributeCategoryDesc *desc = m_AttributeCategories[i];
        if (desc && strcmp(Name, desc->Name) == 0) {
            return i;
        }
    }
    return -1;
}

void CKAttributeManager::SetCategoryName(CKAttributeCategory catType, CKSTRING name) {
    if (!name || catType < 0 || catType >= m_AttributeCategoryCount || !m_AttributeCategories)
        return;

    CKAttributeCategoryDesc *desc = m_AttributeCategories[catType];
    if (!desc)
        return;

    if (desc->Name) {
        delete[] desc->Name;
    }
    desc->Name = new char[strlen(name) + 1];
    strcpy(desc->Name, name);
}

CKAttributeCategory CKAttributeManager::AddCategory(CKSTRING Category, CKDWORD flags) {
    if (!Category)
        return -1;

    int existingIndex = GetCategoryByName(Category);
    if (existingIndex >= 0)
        return existingIndex;

    CKAttributeCategoryDesc **newCategories = new CKAttributeCategoryDesc *[m_AttributeCategoryCount + 1];
    if (m_AttributeCategories) {
        memcpy(newCategories, m_AttributeCategories, sizeof(CKAttributeCategoryDesc *) * m_AttributeCategoryCount);
        delete[] m_AttributeCategories;
    }
    m_AttributeCategories = newCategories;

    CKAttributeCategoryDesc *newDesc = new CKAttributeCategoryDesc();
    newDesc->Name = new char[strlen(Category) + 1];
    strcpy(newDesc->Name, Category);
    newDesc->Flags = flags;

    m_AttributeCategories[m_AttributeCategoryCount] = newDesc;
    return m_AttributeCategoryCount++;
}

void CKAttributeManager::RemoveCategory(CKSTRING Category) {
    int categoryIndex = GetCategoryByName(Category);
    if (categoryIndex < 0 || !m_AttributeCategories)
        return;

    for (CKAttributeType i = 0; i < m_AttributeInfoCount; ++i) {
        CKAttributeDesc *attrDesc = m_AttributeInfos[i];
        if (!attrDesc)
            continue;

        if (attrDesc->AttributeCategory == categoryIndex) {
            attrDesc->AttributeCategory = -1;
        } else if (attrDesc->AttributeCategory > categoryIndex) {
            attrDesc->AttributeCategory--;
        }
    }

    CKAttributeCategoryDesc *categoryDesc = m_AttributeCategories[categoryIndex];

    delete[] categoryDesc->Name;
    delete categoryDesc;

    const int newCount = m_AttributeCategoryCount - 1;
    if (newCount < 0) return; // Should not happen but for safety.
    CKAttributeCategoryDesc **newCategories = (newCount > 0) ? new CKAttributeCategoryDesc *[newCount] : nullptr;

    if (newCount > 0) {
        if (categoryIndex > 0) {
            memcpy(newCategories,
                   m_AttributeCategories,
                   sizeof(CKAttributeCategoryDesc *) * categoryIndex);
        }

        if (categoryIndex < newCount) {
            memcpy(newCategories + categoryIndex,
                   m_AttributeCategories + categoryIndex + 1,
                   sizeof(CKAttributeCategoryDesc *) * (newCount - categoryIndex));
        }
    }

    delete[] m_AttributeCategories;
    m_AttributeCategories = newCategories;
    m_AttributeCategoryCount = newCount;
}

CKDWORD CKAttributeManager::GetCategoryFlags(CKAttributeCategory cat) {
    if (cat < 0 || cat >= m_AttributeCategoryCount || !m_AttributeCategories)
        return 0;

    CKAttributeCategoryDesc *desc = m_AttributeCategories[cat];
    if (!desc)
        return 0;

    return desc->Flags;
}

CKDWORD CKAttributeManager::GetCategoryFlags(CKSTRING cat) {
    if (cat)
        return GetCategoryFlags(GetCategoryByName(cat));
    return 0;
}

void CKAttributeManager::SetAttributeCategory(CKAttributeType AttribType, CKSTRING Category) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return;

    desc->AttributeCategory = AddCategory(Category);
}

CKSTRING CKAttributeManager::GetAttributeCategory(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return nullptr;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return nullptr;

    return GetCategoryName(desc->AttributeCategory);
}

CKAttributeCategory CKAttributeManager::GetAttributeCategoryIndex(CKAttributeType AttribType) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return 0;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (!desc)
        return 0;

    return desc->AttributeCategory;
}

void CKAttributeManager::AddAttributeToObject(CKAttributeType AttribType, CKBeObject *beo) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos)
        return;
    if (!m_AttributeInfos[AttribType] || !beo)
        return;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    CKScene *currentScene = m_Context->GetCurrentScene();

    if (beo->IsInScene(currentScene) || beo->IsPrivate()) {
        desc->AttributeList.AddIfNotHere(beo);
    }

    if (desc->GlobalAttributeList.AddIfNotHere(beo)) {
        if (desc->CallbackFct) {
            desc->CallbackFct(AttribType, TRUE, beo, desc->CallbackArg);
        }
    }
}

void CKAttributeManager::RemoveAttributeFromObject(CKAttributeType AttribType, CKBeObject *beo) {
    if (AttribType < 0 || AttribType >= m_AttributeInfoCount || !m_AttributeInfos || !beo)
        return;

    CKAttributeDesc *desc = m_AttributeInfos[AttribType];
    if (desc) {
        desc->AttributeList.Remove(beo);
        if (desc->GlobalAttributeList.Remove(beo)) {
            if (desc->CallbackFct) {
                desc->CallbackFct(AttribType, FALSE, beo, desc->CallbackArg);
            }
        }
    }
}

void CKAttributeManager::RefreshList(CKObject *obj, CKScene *scene) {
    if (!obj || !scene || !m_AttributeInfos)
        return;

    if (CKIsChildClassOf(obj, CKCID_BEOBJECT) && scene == m_Context->GetCurrentScene()) {
        CKBeObject *beo = (CKBeObject *) obj;
        const int count = beo->GetAttributeCount();
        for (int i = 0; i < count; ++i) {
            CKAttributeType type = beo->GetAttributeType(i);
            if (type >= 0 && type < m_AttributeInfoCount) {
                CKAttributeDesc *desc = m_AttributeInfos[type];
                if (desc)
                    desc->AttributeList.AddIfNotHere(beo);
            }
        }
    }
}

CKPluginEntry *CKAttributeManager::GetCreatorDll(int pos) {
    if (pos < 0 || pos >= m_AttributeInfoCount)
        return nullptr;

    CKAttributeDesc *desc = m_AttributeInfos[pos];
    if (!desc)
        return nullptr;

    return desc->CreatorDll;
}

void CKAttributeManager::NewActiveScene(CKScene *scene) {
    for (int i = 0; i < m_AttributeInfoCount; ++i) {
        CKAttributeDesc *desc = m_AttributeInfos[i];
        if (!desc)
            continue;

        desc->AttributeList.Clear();
        if (scene) {
            for (auto it = desc->GlobalAttributeList.Begin(); it != desc->GlobalAttributeList.End(); ++it) {
                CKBeObject *beo = (CKBeObject *) *it;
                if (beo && (beo->IsInScene(scene) || beo->IsPrivate())) {
                    desc->AttributeList.PushBack(beo);
                }
            }
        }
    }
}

void CKAttributeManager::ObjectAddedToScene(CKBeObject *beo, CKScene *scene) {
    if (!beo || !scene)
        return;

    if (m_Context->GetCurrentScene() == scene) {
        const int count = beo->GetAttributeCount();
        for (CKAttributeType i = 0; i < count; ++i) {
            CKAttributeType type = beo->GetAttributeType(i);
            if (type != -1) {
                CKAttributeDesc *desc = m_AttributeInfos[type];
                if (desc)
                    desc->AttributeList.AddIfNotHere(beo);
            }
        }
    }
}

void CKAttributeManager::ObjectRemovedFromScene(CKBeObject *beo, CKScene *scene) {
    if (!beo || !scene)
        return;

    CKScene *currentScene = m_Context->GetCurrentScene();
    if (currentScene != scene)
        return;

    const int attributeCount = beo->GetAttributeCount();
    for (int i = 0; i < attributeCount; ++i) {
        CKAttributeType attrType = beo->GetAttributeType(i);
        if (attrType < 0 || attrType >= m_AttributeInfoCount)
            continue;

        CKAttributeDesc *attrDesc = m_AttributeInfos[attrType];
        if (!attrDesc)
            continue;

        attrDesc->AttributeList.Remove(beo);
    }
}

void CKAttributeManager::PatchRemapBeObjectFileChunk(CKStateChunk *chunk) {
    if (m_ConversionTable && m_ConversionTableCount > 0) {
        chunk->AttributePatch(1, m_ConversionTable, m_ConversionTableCount);
    }
}

void CKAttributeManager::PatchRemapBeObjectStateChunk(CKStateChunk *chunk) {
    if (m_ConversionTable && m_ConversionTableCount > 0) {
        chunk->AttributePatch(0, m_ConversionTable, m_ConversionTableCount);
    }
}

CKAttributeManager::CKAttributeManager(CKContext *Context): CKBaseManager(
    Context, ATTRIBUTE_MANAGER_GUID, "Attribute Manager") {
    m_AttributeInfoCount = 0;
    m_AttributeInfos = nullptr;
    m_AttributeCategoryCount = 0;
    m_AttributeCategories = nullptr;
    m_ConversionTableCount = 0;
    m_ConversionTable = nullptr;
    m_Saving = FALSE;
    m_Context->RegisterNewManager(this);
}

CKAttributeManager::~CKAttributeManager() {
    for (int i = 0; i < m_AttributeInfoCount; ++i) {
        CKAttributeDesc *desc = m_AttributeInfos[i];
        if (desc) {
            delete[] desc->DefaultValue;
            desc->DefaultValue = nullptr;
            delete desc;
        }
    }
    delete[] m_AttributeInfos;
    m_AttributeInfos = nullptr;
    m_AttributeInfoCount = 0;

    for (int i = 0; i < m_AttributeCategoryCount; ++i) {
        CKAttributeCategoryDesc *category = m_AttributeCategories[i];
        if (category) {
            delete[] category->Name;
            delete category;
        }
    }
    delete[] m_AttributeCategories;
    m_AttributeCategories = nullptr;
    m_AttributeCategoryCount = 0;

    delete[] m_ConversionTable;
    m_ConversionTable = nullptr;
    m_ConversionTableCount = 0;

    m_AttributeList.Clear();
    m_AttributeMask.Clear();
}

CKERROR CKAttributeManager::PreClearAll() {
    delete[] m_ConversionTable;
    m_ConversionTable = nullptr;
    m_ConversionTableCount = 0;

    for (int i = 0; i < m_AttributeInfoCount; ++i) {
        CKAttributeDesc *desc = m_AttributeInfos[i];
        if (!desc)
            continue;

        if (desc->CallbackFct) {
            XObjectPointerArray &globalList = desc->GlobalAttributeList;
            for (XObjectPointerArray::Iterator it = globalList.Begin(); it != globalList.End(); ++it) {
                CKBeObject *obj = (CKBeObject *) *it;
                if (obj) {
                    desc->CallbackFct(i, FALSE, obj, desc->CallbackArg);
                }
            }
        }

        desc->AttributeList.Clear();
        desc->GlobalAttributeList.Clear();

        if (!(desc->Flags & CK_ATTRIBUT_SYSTEM)) {
            delete[] desc->DefaultValue;
            delete desc;
            m_AttributeInfos[i] = nullptr;
        }
    }

    for (int catIdx = 0; catIdx < m_AttributeCategoryCount;) {
        CKAttributeCategoryDesc *category = m_AttributeCategories ? m_AttributeCategories[catIdx] : nullptr;
        if (!category) {
            catIdx++;
            continue;
        }

        bool used = false;
        for (int attrIdx = 0; attrIdx < m_AttributeInfoCount; ++attrIdx) {
            CKAttributeDesc *attrDesc = m_AttributeInfos[attrIdx];
            if (attrDesc && attrDesc->AttributeCategory == catIdx) {
                used = true;
                break;
            }
        }

        if (!used) {
            RemoveCategory(category->Name);
        } else {
            catIdx++;
        }
    }

    return CK_OK;
}

CKERROR CKAttributeManager::PostLoad() {
    for (int catIdx = 0; catIdx < m_AttributeCategoryCount;) {
        CKAttributeCategoryDesc *category = m_AttributeCategories[catIdx];

        if (!category || (category->Flags & CK_ATTRIBUT_USER)) {
            catIdx++;
            continue;
        }

        bool categoryUsed = false;
        for (int attrIdx = 0; attrIdx < m_AttributeInfoCount; ++attrIdx) {
            if (m_AttributeInfos[attrIdx] && m_AttributeInfos[attrIdx]->AttributeCategory == catIdx) {
                categoryUsed = true;
                break;
            }
        }

        if (!categoryUsed) {
            RemoveCategory(category->Name);
        } else {
            catIdx++;
        }
    }

    return CK_OK;
}

CKERROR CKAttributeManager::LoadData(CKStateChunk *chunk, CKFile *LoadedFile) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    chunk->StartRead();

    if (!chunk->SeekIdentifier(0x52))
        return CK_OK;

    int categoryCount = chunk->ReadInt();
    m_ConversionTableCount = chunk->ReadInt();
    delete[] m_ConversionTable; // Clean up any existing table
    m_ConversionTable = new int[m_ConversionTableCount];

    XArray<CKAttributeCategoryDesc> categories;
    categories.Resize(categoryCount);
    XArray<CKAttributeCategory> categoryConversions;
    categoryConversions.Resize(categoryCount);

    for (int i = 0; i < categoryCount; ++i) {
        if (chunk->ReadInt()) {
            CKAttributeCategoryDesc &desc = categories[i];
            desc.Name = nullptr;
            chunk->ReadString(&desc.Name);
            CKDWORD flags = chunk->ReadDword();
            if (GetCategoryByName(desc.Name) < 0) {
                flags = (flags & ~0x20) | 0x13;
            }
            desc.Flags = flags;

            int newCatIdx = AddCategory(desc.Name, flags);
            categoryConversions.PushBack(newCatIdx);

            delete[] desc.Name;
            desc.Name = nullptr;
        } else {
            categoryConversions.PushBack(-1);
        }
    }

    // Load attribute conversions
    for (int i = 0; i < m_ConversionTableCount; ++i) {
        if (chunk->ReadInt()) {
            CKSTRING name = nullptr;
            chunk->ReadString(&name);
            CKGUID paramType = chunk->ReadGuid();
            int oldCategoryIdx = chunk->ReadInt();
            CK_CLASSID compatCid = chunk->ReadInt();
            CKDWORD flags = chunk->ReadDword();

            int newCategoryIdx = -1;
            if (oldCategoryIdx >= 0 && oldCategoryIdx < categoryCount) {
                newCategoryIdx = categoryConversions[oldCategoryIdx];
            }

            if (GetAttributeTypeByName(name) < 0) {
                flags = (flags & ~0x20) | 0x13;
            }

            m_ConversionTable[i] = RegisterNewAttributeType(name, paramType, compatCid, (CK_ATTRIBUT_FLAGS) flags);
            if (m_ConversionTable[i] >= 0 && m_ConversionTable[i] < m_AttributeInfoCount) {
                CKAttributeDesc *desc = m_AttributeInfos[m_ConversionTable[i]];
                if (desc && !(desc->Flags & CK_ATTRIBUT_SYSTEM)) {
                    desc->AttributeCategory = newCategoryIdx;
                }
            }

            delete[] name;
        } else {
            m_ConversionTable[i] = -1;
        }
    }

    LoadedFile->RemapManagerInt(ATTRIBUTE_MANAGER_GUID, m_ConversionTable, m_ConversionTableCount);
    return CK_OK;
}

CKStateChunk *CKAttributeManager::SaveData(CKFile *SavedFile) {
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_ATTRIBUTEMANAGER, SavedFile);
    chunk->StartWrite();
    chunk->WriteIdentifier(0x52);

    chunk->WriteInt(m_AttributeCategoryCount);
    chunk->WriteInt(m_AttributeInfoCount);

    const int attributeBitBlocks = (m_AttributeInfoCount >> 5) + 1;
    const int categoryBitBlocks = (m_AttributeCategoryCount >> 5) + 1;
    XBitArray attributeMask(attributeBitBlocks);
    XBitArray categoryMask(categoryBitBlocks);

    CKParameterManager *pm = m_Context->GetParameterManager();
    CKParameterType paramAttrType = pm->ParameterGuidToType(CKPGUID_ATTRIBUTE);

    const int paramOutCount = SavedFile->m_IndexByClassId[CKCID_PARAMETEROUT].Size();
    const int paramLocalCount = SavedFile->m_IndexByClassId[CKCID_PARAMETERLOCAL].Size();

    for (int i = 0; i < paramOutCount; ++i) {
        int fileObjIdx = SavedFile->m_IndexByClassId[CKCID_PARAMETEROUT][i];
        CKParameterOut *param = (CKParameterOut *) SavedFile->m_FileObjects[fileObjIdx].ObjPtr;
        if (param && param->GetType() == paramAttrType) {
            CKAttributeType attrType;
            param->GetValue(&attrType);
            if (attrType >= 0 && attrType < m_AttributeInfoCount && m_AttributeInfos[attrType]) {
                attributeMask.Set(attrType);
            }
        }
    }

    for (int i = 0; i < paramLocalCount; ++i) {
        int fileObjIdx = SavedFile->m_IndexByClassId[CKCID_PARAMETERLOCAL][i];
        CKParameterLocal *param = (CKParameterLocal *) SavedFile->m_FileObjects[fileObjIdx].ObjPtr;
        if (param && param->GetType() == paramAttrType) {
            CKAttributeType attrType;
            param->GetValue(&attrType);
            if (attrType >= 0 && attrType < m_AttributeInfoCount && m_AttributeInfos[attrType]) {
                attributeMask.Set(attrType);
            }
        }
    }

    for (CKAttributeType i = 0; i < m_AttributeInfoCount; ++i) {
        CKAttributeDesc *desc = m_AttributeInfos[i];
        if (desc && !(desc->Flags & CK_ATTRIBUT_DONOTSAVE) && desc->GlobalAttributeList.Size() > 0) {
            attributeMask.Set(i);
        }
    }

    for (CKAttributeType i = 0; i < m_AttributeInfoCount; ++i) {
        CKAttributeDesc *desc = m_AttributeInfos[i];
        if (desc) {
            if (desc->Flags & CK_ATTRIBUT_DONOTSAVE) {
                attributeMask.Unset(i);
            } else if (!m_Saving && (desc->Flags & CK_ATTRIBUT_USER)) {
                attributeMask.Set(i);
            }
        }
    }

    for (CKAttributeType i = 0; i < m_AttributeInfoCount; ++i) {
        CKAttributeDesc *desc = m_AttributeInfos[i];
        if (desc && attributeMask.IsSet(i) && desc->AttributeCategory >= 0) {
            categoryMask.Set(desc->AttributeCategory);
        }
    }

    for (CKAttributeCategory i = 0; i < m_AttributeCategoryCount; ++i) {
        CKAttributeCategoryDesc *cat = m_AttributeCategories[i];
        if (cat && (cat->Flags & CK_ATTRIBUT_DONOTSAVE) && i < categoryMask.Size()) {
            categoryMask.Unset(i);
        }
    }

    for (CKAttributeCategory i = 0; i < m_AttributeCategoryCount; ++i) {
        if (categoryMask.IsSet(i)) {
            CKAttributeCategoryDesc *cat = m_AttributeCategories[i];
            chunk->WriteInt(TRUE);
            chunk->WriteString(cat->Name);
            chunk->WriteDword(cat->Flags);
        } else {
            chunk->WriteInt(FALSE);
        }
    }

    for (CKAttributeType i = 0; i < m_AttributeInfoCount; ++i) {
        if (attributeMask.IsSet(i)) {
            CKAttributeDesc *desc = m_AttributeInfos[i];
            chunk->WriteInt(TRUE);
            chunk->WriteString(desc->Name);
            chunk->WriteGuid(desc->ParameterType);
            chunk->WriteInt(desc->AttributeCategory);
            chunk->WriteInt(desc->CompatibleCid);
            chunk->WriteDword(desc->Flags);
        } else {
            chunk->WriteInt(FALSE);
        }
    }

    m_AttributeMask = attributeMask;
    chunk->CloseChunk();
    return chunk;
}

CKERROR CKAttributeManager::SequenceAddedToScene(CKScene *scn, CK_ID *objid, int count) {
    for (int i = 0; i < count; ++i) {
        CKBeObject *beo = (CKBeObject *) m_Context->GetObject(objid[i]);
        if (beo && CKIsChildClassOf(beo, CKCID_BEOBJECT)) {
            ObjectAddedToScene(beo, scn);
        }
    }
    return CK_OK;
}

CKERROR CKAttributeManager::SequenceRemovedFromScene(CKScene *scn, CK_ID *objid, int count) {
    for (int i = 0; i < count; ++i) {
        CKBeObject *beo = (CKBeObject *) m_Context->GetObject(objid[i]);
        if (beo && CKIsChildClassOf(beo, CKCID_BEOBJECT)) {
            ObjectRemovedFromScene(beo, scn);
        }
    }
    return CK_OK;
}