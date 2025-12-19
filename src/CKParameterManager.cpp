#include "CKParameterManager.h"

#include "CKPluginManager.h"
#include "CKInterfaceManager.h"
#include "CKStateChunk.h"
#include "CKCallbackFunctions.h"

extern CKPluginManager g_ThePluginManager;
extern CKPluginEntry *g_TheCurrentPluginEntry;
extern XClassInfoArray g_CKClassInfo;

CKERROR CKParameterManager::RegisterParameterType(CKParameterTypeDesc *paramType) {
    if (ParameterGuidToType(paramType->Guid) >= 0)
        return CKERR_INVALIDPARAMETERTYPE;

    m_DerivationMasksUpToDate = FALSE;

    int freeSlot = -1;
    XArray<CKParameterTypeDesc *> &regTypes = m_RegisteredTypes;
    const int typeCount = regTypes.Size();

    for (int i = 0; i < typeCount; ++i) {
        if (regTypes[i] == nullptr) {
            freeSlot = i;
            break;
        }
    }

    if (freeSlot == -1) {
        freeSlot = regTypes.Size();
        regTypes.Resize(freeSlot + 1);
    }

    CKParameterTypeDesc *newDesc = new CKParameterTypeDesc();
    if (!newDesc)
        return CKERR_OUTOFMEMORY;

    *newDesc = *paramType;
    newDesc->Index = freeSlot;
    newDesc->Valid = TRUE;
    regTypes[freeSlot] = newDesc;

    CKPluginEntry *creator = g_TheCurrentPluginEntry;
    if (!creator) {
        CKBaseManager *currentMgr = m_Context->m_CurrentManager;
        if (currentMgr && currentMgr != this) {
            creator = g_ThePluginManager.FindComponent(
                currentMgr->m_ManagerGuid,
                CKPLUGIN_MANAGER_DLL
            );
        }
    }
    newDesc->CreatorDll = creator;

    if (!newDesc->CopyFunction) {
        if (newDesc->SaveLoadFunction) {
            newDesc->CopyFunction = CK_ParameterCopier_SaveLoad;
        } else {
            newDesc->CopyFunction = CK_ParameterCopier_SetValue;
        }
    }

    m_ParamGuids.InsertUnique(paramType->Guid, freeSlot);

    m_ParameterTypeEnumUpToDate = FALSE;

    return CK_OK;
}

CKERROR CKParameterManager::UnRegisterParameterType(CKGUID guid) {
    m_DerivationMasksUpToDate = FALSE;

    CKParameterTypeDesc *foundDesc = nullptr;
    for (XArray<CKParameterTypeDesc *>::Iterator it = m_RegisteredTypes.Begin(); it != m_RegisteredTypes.End(); ++it) {
        CKParameterTypeDesc *desc = *it;
        if (desc && desc->Guid == guid) {
            foundDesc = desc;
            if (desc) {
                delete desc;
            }
            *it = nullptr;
            break;
        }
    }

    m_ParamGuids.Remove(guid);

    if (!foundDesc)
        return CKERR_INVALIDPARAMETER;

    CKBOOL parametersAffected = FALSE;

    int paramInCount = m_Context->GetObjectsCountByClassID(CKCID_PARAMETERIN);
    CK_ID *paramInList = m_Context->GetObjectsListByClassID(CKCID_PARAMETERIN);
    for (int i = 0; i < paramInCount; ++i) {
        CKParameterIn *param = (CKParameterIn *) m_Context->GetObject(paramInList[i]);
        if (param->m_ParamType == foundDesc) {
            param->m_ParamType = nullptr;
            parametersAffected = TRUE;
        }
    }

    int paramCount = m_Context->GetObjectsCountByClassID(CKCID_PARAMETER);
    CK_ID *paramList = m_Context->GetObjectsListByClassID(CKCID_PARAMETER);
    for (int i = 0; i < paramCount; ++i) {
        CKParameter *param = (CKParameter *) m_Context->GetObject(paramList[i]);
        if (param && param->m_ParamType == foundDesc) {
            param->m_ParamType = nullptr;
            parametersAffected = TRUE;
        }
    }

    int paramOutCount = m_Context->GetObjectsCountByClassID(CKCID_PARAMETEROUT);
    CK_ID *paramOutList = m_Context->GetObjectsListByClassID(CKCID_PARAMETEROUT);
    for (int i = 0; i < paramOutCount; ++i) {
        CKParameterOut *param = (CKParameterOut *) m_Context->GetObject(paramOutList[i]);
        if (param && param->m_ParamType == foundDesc) {
            param->m_ParamType = nullptr;
            parametersAffected = TRUE;
        }
    }

    int localParamCount = m_Context->GetObjectsCountByClassID(CKCID_PARAMETERLOCAL);
    CK_ID *localParamList = m_Context->GetObjectsListByClassID(CKCID_PARAMETERLOCAL);
    for (int i = 0; i < localParamCount; ++i) {
        CKParameterLocal *param = (CKParameterLocal *) m_Context->GetObject(localParamList[i]);
        if (param && param->m_ParamType == foundDesc) {
            param->m_ParamType = nullptr;
            parametersAffected = TRUE;
        }
    }

    if (parametersAffected) {
        m_Context->OutputToConsole("Warning : Some parameters are not valid any more.");
    }

    for (int i = 0; i < m_NbOperations; ++i) {
        OperationCell &opCell = m_OperationTree[i];
        if (opCell.Tree) {
            for (int j = 0; j < opCell.CellCount; ++j) {
                RecurseDeleteParam(&opCell.Tree[j], guid);
            }
        }
    }

    for (XArray<CKParameterTypeDesc *>::Iterator it = m_RegisteredTypes.Begin(); it != m_RegisteredTypes.End(); ++it) {
        CKParameterTypeDesc *desc = *it;
        if (desc && desc->DerivedFrom == guid) {
            UnRegisterParameterType(desc->Guid);
        }
    }

    m_ParameterTypeEnumUpToDate = FALSE;
    return CK_OK;
}

CKParameterTypeDesc *CKParameterManager::GetParameterTypeDescription(int type) {
    if (type < 0 || type >= m_RegisteredTypes.Size())
        return nullptr;
    return m_RegisteredTypes[type];
}

CKParameterTypeDesc *CKParameterManager::GetParameterTypeDescription(CKGUID guid) {
    auto type = ParameterGuidToType(guid);
    if (type == -1)
        return nullptr;
    return m_RegisteredTypes[type];
}

int CKParameterManager::GetParameterSize(CKParameterType type) {
    if (CheckParamTypeValidity(type))
        return m_RegisteredTypes[type]->DefaultSize;
    return 0;
}

int CKParameterManager::GetParameterTypesCount() {
    return m_RegisteredTypes.Size();
}

CKParameterType CKParameterManager::ParameterGuidToType(CKGUID guid) {
    auto *pType = m_ParamGuids.FindPtr(guid);
    if (pType)
        return *pType;
    return -1;
}

CKSTRING CKParameterManager::ParameterGuidToName(CKGUID guid) {
    CKParameterType type = ParameterGuidToType(guid);
    return ParameterTypeToName(type);
}

CKGUID CKParameterManager::ParameterTypeToGuid(CKParameterType type) {
    if (CheckParamTypeValidity(type)) {
        return m_RegisteredTypes[type]->Guid;
    }
    return CKGUID();
}

CKSTRING CKParameterManager::ParameterTypeToName(CKParameterType type) {
    if (CheckParamTypeValidity(type)) {
        return m_RegisteredTypes[type]->TypeName.Str();
    }
    return nullptr;
}

CKGUID CKParameterManager::ParameterNameToGuid(CKSTRING name) {
    if (!name || name[0] == '\0') {
        return CKGUID();
    }

    const int typeCount = m_RegisteredTypes.Size();
    for (int i = 0; i < typeCount; ++i) {
        CKParameterTypeDesc *typeDesc = m_RegisteredTypes[i];
        if (!typeDesc) continue;

        const char *registeredName = typeDesc->TypeName.CStr();
        if (!registeredName) registeredName = "";

        if (strcmp(registeredName, name) == 0) {
            return typeDesc->Guid;
        }
    }

    return CKGUID();
}

CKParameterType CKParameterManager::ParameterNameToType(CKSTRING name) {
    if (!name) return -1;

    const int typeCount = m_RegisteredTypes.Size();
    for (int i = 0; i < typeCount; ++i) {
        CKParameterTypeDesc *desc = m_RegisteredTypes[i];
        if (!desc) continue;

        const char *registeredName = desc->TypeName.CStr();
        if (!registeredName) registeredName = "";

        if (strcmp(registeredName, name) == 0) {
            return i;
        }
    }
    return -1;
}

CKBOOL CKParameterManager::IsDerivedFrom(CKGUID guid1, CKGUID parent) {
    CKParameterType type1 = ParameterGuidToType(guid1);
    CKParameterType type2 = ParameterGuidToType(parent);
    return IsDerivedFrom(type1, type2);
}

CKBOOL CKParameterManager::IsDerivedFrom(CKParameterType child, CKParameterType parent) {
    if (child == parent)
        return true;

    if (!m_DerivationMasksUpToDate) {
        UpdateDerivationTables();
    }

    CKParameterTypeDesc *childDesc = GetParameterTypeDescription(child);
    if (!childDesc || parent >= childDesc->DerivationMask.Size()) {
        return false;
    }

    return childDesc->DerivationMask.IsSet(parent) != 0;
}

CKBOOL CKParameterManager::IsTypeCompatible(CKGUID guid1, CKGUID guid2) {
    CKParameterType type1 = ParameterGuidToType(guid1);
    CKParameterType type2 = ParameterGuidToType(guid2);
    return IsTypeCompatible(type1, type2);
}

CKBOOL CKParameterManager::IsTypeCompatible(CKParameterType type1, CKParameterType type2) {
    if (type1 == type2)
        return true;

    if (!m_DerivationMasksUpToDate) {
        UpdateDerivationTables();
    }

    CKParameterTypeDesc *desc1 = GetParameterTypeDescription(type1);
    CKParameterTypeDesc *desc2 = GetParameterTypeDescription(type2);
    if (!desc1 || !desc2)
        return false;

    return IsDerivedFrom(type1, type2) || IsDerivedFrom(type2, type1);
}

CK_CLASSID CKParameterManager::TypeToClassID(CKParameterType type) {
    if (CheckParamTypeValidity(type))
        return m_RegisteredTypes[type]->Cid;
    return 0;
}

CK_CLASSID CKParameterManager::GuidToClassID(CKGUID guid) {
    CKParameterType type = ParameterGuidToType(guid);
    if (type >= 0)
        return m_RegisteredTypes[type]->Cid;
    return 0;
}

CKParameterType CKParameterManager::ClassIDToType(CK_CLASSID cid) {
    CKGUID guid = ClassIDToGuid(cid);
    return ParameterGuidToType(guid);
}

CKGUID CKParameterManager::ClassIDToGuid(CK_CLASSID cid) {
    if (cid < 0 || cid >= g_CKClassInfo.Size())
        return CKGUID();
    return g_CKClassInfo[cid].Parameter;
}

CKERROR CKParameterManager::RegisterNewFlags(CKGUID FlagsGuid, CKSTRING FlagsName, CKSTRING FlagsData) {
    if (!FlagsData || !FlagsName)
        return CKERR_INVALIDPARAMETER;

    if (ParameterGuidToType(FlagsGuid) >= 0) {
        return CKERR_INVALIDGUID;
    }

    int flagCount = 1;
    const char *ptr = FlagsData;
    while ((ptr = strchr(ptr, ','))) {
        flagCount++;
        ptr++;
    }

    CKFlagsStruct flagStruct;
    flagStruct.NbData = flagCount;
    flagStruct.Vals = new int[flagCount];
    flagStruct.Desc = new CKSTRING[flagCount];
    for (int i = 0; i < flagCount; ++i) flagStruct.Desc[i] = nullptr;

    // Parse the flags data
    int currentValue = 1;
    char buffer[512];
    const char *currentPos = FlagsData;

    for (int i = 0; i < flagCount; ++i) {
        const char *end = strchr(currentPos, ',');
        size_t len = end ? (end - currentPos) : strlen(currentPos);
        strncpy(buffer, currentPos, len);
        buffer[len] = '\0';

        char *equalSign = strchr(buffer, '=');
        if (equalSign) {
            *equalSign = '\0';
            sscanf(equalSign + 1, "%d", &currentValue);
            len = equalSign - buffer;
        }

        flagStruct.Desc[i] = new char[len + 1];
        strncpy(flagStruct.Desc[i], buffer, len);
        flagStruct.Desc[i][len] = '\0';
        flagStruct.Vals[i] = currentValue;

        currentValue <<= 1;
        if (currentValue == 0) currentValue = 1;

        currentPos = end ? end + 1 : currentPos + len;
    }

    CKParameterTypeDesc flagDesc;
    flagDesc.Guid = FlagsGuid;
    flagDesc.TypeName = FlagsName;
    flagDesc.DerivedFrom = CKPGUID_FLAGS;
    flagDesc.DefaultSize = sizeof(CKDWORD);
    flagDesc.CopyFunction = CK_ParameterCopier_Dword;
    flagDesc.StringFunction = CKFlagsStringFunc;
    flagDesc.UICreatorFunction = GetUICreatorFunction(m_Context, &flagDesc);
    flagDesc.dwParam = m_NbFlagsDefined;
    flagDesc.dwFlags = CKPARAMETERTYPE_FLAGS;

    CKERROR err = RegisterParameterType(&flagDesc);
    if (err != CK_OK) {
        // Cleanup on failure
        for (int i = 0; i < flagCount; ++i) delete[] flagStruct.Desc[i];
        delete[] flagStruct.Desc;
        delete[] flagStruct.Vals;
        return err;
    }

    CKFlagsStruct *newFlags = new CKFlagsStruct[m_NbFlagsDefined + 1];
    if (m_Flags) {
        memcpy(newFlags, m_Flags, sizeof(CKFlagsStruct) * m_NbFlagsDefined);
        delete[] m_Flags;
    }
    m_Flags = newFlags;
    m_Flags[m_NbFlagsDefined] = flagStruct;
    m_NbFlagsDefined++;

    return CK_OK;
}

CKERROR CKParameterManager::RegisterNewEnum(CKGUID EnumGuid, CKSTRING EnumName, CKSTRING EnumData) {
    if (!EnumData || !EnumName || EnumName[0] == '\0')
        return CKERR_INVALIDPARAMETER;

    if (ParameterGuidToType(EnumGuid) >= 0) {
        return CKERR_INVALIDGUID;
    }

    int entryCount = 1;
    const char *counterPtr = EnumData;
    while ((counterPtr = strchr(counterPtr, ','))) {
        entryCount++;
        counterPtr++;
    }

    CKEnumStruct enumStruct;
    enumStruct.NbData = entryCount;
    enumStruct.Vals = new int[entryCount];
    enumStruct.Desc = new CKSTRING[entryCount];

    if (!enumStruct.Vals || !enumStruct.Desc) {
        delete[] enumStruct.Vals;
        delete[] enumStruct.Desc;
        return CKERR_OUTOFMEMORY;
    }
    for (int i = 0; i < entryCount; ++i) enumStruct.Desc[i] = nullptr;

    // Parse data
    int currentValueForNextDefault = 0;
    char buffer[512];
    const char *currentPosInEnumData = EnumData;

    for (int i = 0; i < entryCount; ++i) {
        const char *entryEnd = strchr(currentPosInEnumData, ',');
        size_t currentEntryLength = entryEnd ? entryEnd - currentPosInEnumData : strlen(currentPosInEnumData);
        size_t copyLength = XMin((size_t) 511, currentEntryLength);
        strncpy(buffer, currentPosInEnumData, copyLength);
        buffer[copyLength] = '\0';

        char *equalSign = strchr(buffer, '=');
        int parsedValue;
        size_t nameLength;

        if (equalSign) {
            *equalSign = '\0';
            nameLength = equalSign - buffer;
            if (sscanf(equalSign + 1, "%d", &parsedValue) != 1) {
                parsedValue = currentValueForNextDefault;
            }
        } else {
            nameLength = strlen(buffer);
            parsedValue = currentValueForNextDefault;
        }

        enumStruct.Desc[i] = new char[nameLength + 1];
        strncpy(enumStruct.Desc[i], buffer, nameLength);
        enumStruct.Desc[i][nameLength] = '\0';

        enumStruct.Vals[i] = parsedValue;
        currentValueForNextDefault = parsedValue + 1;

        currentPosInEnumData = entryEnd ? entryEnd + 1 : currentPosInEnumData + currentEntryLength;
    }

    CKParameterTypeDesc enumDesc;
    enumDesc.Guid = EnumGuid;
    enumDesc.TypeName = EnumName;
    enumDesc.DerivedFrom = CKPGUID_INT;
    enumDesc.DefaultSize = sizeof(CKDWORD);
    enumDesc.CopyFunction = CK_ParameterCopier_Dword;
    enumDesc.StringFunction = CKEnumStringFunc;
    enumDesc.UICreatorFunction = GetUICreatorFunction(m_Context, &enumDesc);
    enumDesc.dwParam = m_NbEnumsDefined;
    enumDesc.dwFlags = CKPARAMETERTYPE_ENUMS;

    CKERROR err = RegisterParameterType(&enumDesc);
    if (err != CK_OK) {
        // Cleanup on failure
        for (int i = 0; i < entryCount; ++i) delete[] enumStruct.Desc[i];
        delete[] enumStruct.Desc;
        delete[] enumStruct.Vals;
        return err;
    }

    CKEnumStruct *newEnums = new CKEnumStruct[m_NbEnumsDefined + 1];
    if (m_Enums) {
        memcpy(newEnums, m_Enums, sizeof(CKEnumStruct) * m_NbEnumsDefined);
        delete[] m_Enums;
    }
    m_Enums = newEnums;
    m_Enums[m_NbEnumsDefined] = enumStruct;
    m_NbEnumsDefined++;

    return CK_OK;
}

CKERROR CKParameterManager::ChangeEnumDeclaration(CKGUID EnumGuid, CKSTRING EnumData) {
    if (!m_Enums || !EnumData || EnumData[0] == '\0')
        return CKERR_INVALIDPARAMETER;

    CKParameterType type = ParameterGuidToType(EnumGuid);
    if (type < 0)
        return CKERR_INVALIDGUID;

    CKParameterTypeDesc *typeDesc = GetParameterTypeDescription(type);
    if (!typeDesc)
        return CKERR_INVALIDGUID;

    const int enumIndex = (int) typeDesc->dwParam;
    if (enumIndex >= m_NbEnumsDefined)
        return CKERR_INVALIDPARAMETER;

    int entryCount = 1;
    const char *counterPtr = EnumData;
    while ((counterPtr = strchr(counterPtr, ','))) {
        entryCount++;
        counterPtr++; // Skip the comma
    }

    CKEnumStruct &enumStruct = m_Enums[enumIndex];
    for (int i = 0; i < enumStruct.NbData; ++i) delete[] enumStruct.Desc[i];
    delete[] enumStruct.Desc;
    delete[] enumStruct.Vals;

    enumStruct.NbData = entryCount;
    enumStruct.Vals = new int[entryCount];
    enumStruct.Desc = new CKSTRING[entryCount];
    for (int i = 0; i < entryCount; ++i) enumStruct.Desc[i] = nullptr;

    int currentValueForNextDefault = 0;
    char buffer[512];
    const char *currentPosInEnumData = EnumData;

    for (int i = 0; i < entryCount; ++i) {
        const char *entryEnd = strchr(currentPosInEnumData, ',');
        size_t currentEntryLength = entryEnd ? entryEnd - currentPosInEnumData : strlen(currentPosInEnumData);
        size_t copyLength = XMin((size_t) 511, currentEntryLength);
        strncpy(buffer, currentPosInEnumData, copyLength);
        buffer[copyLength] = '\0';

        char *equalSign = strchr(buffer, '=');
        int parsedValue;
        size_t nameLength;

        if (equalSign) {
            *equalSign = '\0';
            nameLength = equalSign - buffer;
            if (sscanf(equalSign + 1, "%d", &parsedValue) != 1) {
                parsedValue = currentValueForNextDefault;
            }
        } else {
            nameLength = strlen(buffer);
            parsedValue = currentValueForNextDefault;
        }

        enumStruct.Desc[i] = new char[nameLength + 1];
        strncpy(enumStruct.Desc[i], buffer, nameLength);
        enumStruct.Desc[i][nameLength] = '\0';

        enumStruct.Vals[i] = parsedValue;
        currentValueForNextDefault = parsedValue + 1;

        currentPosInEnumData = entryEnd ? entryEnd + 1 : currentPosInEnumData + currentEntryLength;
    }

    return CK_OK;
}

CKERROR CKParameterManager::ChangeFlagsDeclaration(CKGUID FlagsGuid, CKSTRING FlagsData) {
    if (!m_Flags || !FlagsData)
        return CKERR_INVALIDPARAMETER;

    CKParameterType type = ParameterGuidToType(FlagsGuid);
    if (type < 0)
        return CKERR_INVALIDGUID;

    CKParameterTypeDesc *typeDesc = GetParameterTypeDescription(type);
    if (!typeDesc)
        return CKERR_INVALIDGUID;

    const int flagsIndex = typeDesc->dwParam;
    if (flagsIndex >= m_NbFlagsDefined)
        return CKERR_INVALIDPARAMETER;

    int entryCount = 1;
    const char *ptr = FlagsData;
    while ((ptr = strchr(ptr, ','))) {
        entryCount++;
        ptr++;
    }

    CKFlagsStruct &flagsStruct = m_Flags[flagsIndex];
    for (int i = 0; i < flagsStruct.NbData; ++i) delete[] flagsStruct.Desc[i];
    delete[] flagsStruct.Desc;
    delete[] flagsStruct.Vals;

    flagsStruct.NbData = entryCount;
    flagsStruct.Vals = new int[entryCount];
    flagsStruct.Desc = new CKSTRING[entryCount];
    for (int i = 0; i < entryCount; ++i) flagsStruct.Desc[i] = nullptr;

    int currentValue = 1;
    char buffer[512];
    const char *currentPos = FlagsData;

    for (int i = 0; i < entryCount; ++i) {
        const char *end = strchr(currentPos, ',');
        size_t len = end ? (end - currentPos) : strlen(currentPos);
        strncpy(buffer, currentPos, len);
        buffer[len] = '\0';

        char *equalSign = strchr(buffer, '=');
        if (equalSign) {
            *equalSign = '\0';
            sscanf(equalSign + 1, "%d", &currentValue);
            len = equalSign - buffer;
        }

        flagsStruct.Desc[i] = new char[len + 1];
        strncpy(flagsStruct.Desc[i], buffer, len);
        flagsStruct.Desc[i][len] = '\0';
        flagsStruct.Vals[i] = currentValue;

        currentValue <<= 1;
        if (currentValue == 0) currentValue = 1;

        currentPos = end ? end + 1 : currentPos + len;
    }

    return CK_OK;
}

CKERROR CKParameterManager::RegisterNewStructure(CKGUID StructGuid, CKSTRING StructName, CKSTRING StructData, ...) {
    if (!StructName || !StructData)
        return CKERR_INVALIDPARAMETER;

    if (ParameterGuidToType(StructGuid) >= 0)
        return CKERR_INVALIDGUID;

    int memberCount = 1;
    const char *ptr = StructData;
    while ((ptr = strchr(ptr, ','))) {
        memberCount++;
        ptr++;
    }

    CKStructStruct structStruct;
    structStruct.NbData = memberCount;
    structStruct.Guids = new CKGUID[memberCount];
    structStruct.Desc = new CKSTRING[memberCount];
    for (int i = 0; i < memberCount; ++i) structStruct.Desc[i] = nullptr;

    const char *currentPos = StructData;
    for (int i = 0; i < memberCount; ++i) {
        const char *end = strchr(currentPos, ',');
        int len = end ? (end - currentPos) : (int) strlen(currentPos);
        structStruct.Desc[i] = new char[len + 1];
        strncpy(structStruct.Desc[i], currentPos, len);
        structStruct.Desc[i][len] = '\0';
        currentPos = end ? end + 1 : currentPos + len;
    }

    va_list args;
    va_start(args, StructData);
    for (int i = 0; i < memberCount; ++i) {
        structStruct.Guids[i] = va_arg(args, CKGUID);
    }
    va_end(args);

    CKParameterTypeDesc structDesc;
    structDesc.Guid = StructGuid;
    structDesc.TypeName = StructName;
    structDesc.DefaultSize = sizeof(CKDWORD) * memberCount;
    structDesc.CreateDefaultFunction = CKStructCreator;
    structDesc.DeleteFunction = CKStructDestructor;
    structDesc.SaveLoadFunction = CKStructSaver;
    structDesc.CopyFunction = CKStructCopier;
    structDesc.StringFunction = CKStructStringFunc;
    structDesc.UICreatorFunction = GetUICreatorFunction(m_Context, &structDesc);
    structDesc.dwFlags = CKPARAMETERTYPE_STRUCT;
    structDesc.dwParam = m_NbStructDefined;

    CKERROR err = RegisterParameterType(&structDesc);
    if (err != CK_OK) {
        // Cleanup on failure
        delete[] structStruct.Guids;
        for (int i = 0; i < memberCount; ++i) delete[] structStruct.Desc[i];
        delete[] structStruct.Desc;
        return err;
    }

    CKStructStruct *newStructs = new CKStructStruct[m_NbStructDefined + 1];
    if (m_Structs) {
        memcpy(newStructs, m_Structs, sizeof(CKStructStruct) * m_NbStructDefined);
        delete[] m_Structs;
    }
    m_Structs = newStructs;
    m_Structs[m_NbStructDefined] = structStruct;
    m_NbStructDefined++;

    return CK_OK;
}

CKERROR CKParameterManager::RegisterNewStructure(CKGUID StructGuid, CKSTRING StructName, CKSTRING StructData,
                                                 XArray<CKGUID> &ListGuid) {
    if (!StructData || !StructName[0])
        return CKERR_INVALIDPARAMETER;

    if (ParameterGuidToType(StructGuid) >= 0)
        return CKERR_INVALIDGUID;

    int memberCount = 1;
    const char *ptr = StructData;
    while ((ptr = strchr(ptr, ','))) {
        memberCount++;
        ptr++;
    }

    if (ListGuid.Size() != memberCount) {
        return CKERR_INVALIDPARAMETER;
    }

    CKStructStruct structStruct;
    structStruct.NbData = memberCount;
    structStruct.Guids = new CKGUID[memberCount];
    structStruct.Desc = new CKSTRING[memberCount];
    for (int i = 0; i < memberCount; ++i) structStruct.Desc[i] = nullptr;

    const char *currentPos = StructData;
    for (int i = 0; i < memberCount; ++i) {
        const char *end = strchr(currentPos, ',');
        int len = end ? (end - currentPos) : (int) strlen(currentPos);
        structStruct.Desc[i] = new char[len + 1];
        strncpy(structStruct.Desc[i], currentPos, len);
        structStruct.Desc[i][len] = '\0';
        structStruct.Guids[i] = ListGuid[i];
        currentPos = end ? end + 1 : currentPos + len;
    }

    CKParameterTypeDesc structDesc;
    structDesc.Guid = StructGuid;
    structDesc.TypeName = StructName;
    structDesc.DefaultSize = sizeof(CKDWORD) * memberCount;
    structDesc.CreateDefaultFunction = CKStructCreator;
    structDesc.DeleteFunction = CKStructDestructor;
    structDesc.SaveLoadFunction = CKStructSaver;
    structDesc.CopyFunction = CKStructCopier;
    structDesc.StringFunction = CKStructStringFunc;
    structDesc.UICreatorFunction = GetUICreatorFunction(m_Context, &structDesc);
    structDesc.dwFlags = CKPARAMETERTYPE_STRUCT;
    structDesc.dwParam = m_NbStructDefined;

    CKERROR err = RegisterParameterType(&structDesc);
    if (err != CK_OK) {
        // Cleanup on failure
        delete[] structStruct.Guids;
        for (int i = 0; i < memberCount; ++i) delete[] structStruct.Desc[i];
        delete[] structStruct.Desc;
        return err;
    }

    CKStructStruct *newStructs = new CKStructStruct[m_NbStructDefined + 1];
    if (m_Structs) {
        memcpy(newStructs, m_Structs, sizeof(CKStructStruct) * m_NbStructDefined);
        delete[] m_Structs;
    }
    m_Structs = newStructs;
    m_Structs[m_NbStructDefined] = structStruct;
    m_NbStructDefined++;

    return CK_OK;
}

int CKParameterManager::GetNbFlagDefined() {
    return m_NbFlagsDefined;
}

int CKParameterManager::GetNbEnumDefined() {
    return m_NbEnumsDefined;
}

int CKParameterManager::GetNbStructDefined() {
    return m_NbStructDefined;
}

CKFlagsStruct *CKParameterManager::GetFlagsDescByType(CKParameterType pType) {
    CKParameterTypeDesc *desc = GetParameterTypeDescription(pType);
    if (desc && desc->dwFlags & CKPARAMETERTYPE_FLAGS && desc->dwParam < m_NbFlagsDefined) {
        return &m_Flags[desc->dwParam];
    }
    return nullptr;
}

CKEnumStruct *CKParameterManager::GetEnumDescByType(CKParameterType pType) {
    CKParameterTypeDesc *desc = GetParameterTypeDescription(pType);
    if (desc && desc->dwFlags & CKPARAMETERTYPE_ENUMS && desc->dwParam < m_NbEnumsDefined) {
        return &m_Enums[desc->dwParam];
    }
    return nullptr;
}

CKStructStruct *CKParameterManager::GetStructDescByType(CKParameterType pType) {
    CKParameterTypeDesc *desc = GetParameterTypeDescription(pType);
    if (desc && desc->dwFlags & CKPARAMETERTYPE_STRUCT && desc->dwParam < m_NbStructDefined) {
        return &m_Structs[desc->dwParam];
    }
    return nullptr;
}

CKOperationType CKParameterManager::RegisterOperationType(CKGUID OpCode, CKSTRING name) {
    CKOperationType existingCode = OperationNameToCode(name);
    if (existingCode >= 0)
        return existingCode;

    existingCode = OperationGuidToCode(OpCode);
    if (existingCode >= 0)
        return existingCode;

    CKOperationType opType = -1;
    const int currentCount = m_NbOperations;

    for (int i = 0; i < currentCount; ++i) {
        if (!m_OperationTree[i].IsActive) {
            opType = i;
            break;
        }
    }

    if (opType == 0 && currentCount > 0) {
        // This case should be when no inactive slot is found. opType remains 0.
        // It should then proceed to expand array.
        // The original code had a bug here where it would not find a free slot and then not expand.
    }

    if (opType == -1) {
        opType = m_NbOperations;
    }

    if (opType >= m_NbAllocatedOperations) {
        const int newAllocation = m_NbAllocatedOperations + 40;
        OperationCell *newTree = new OperationCell[newAllocation];
        if (!newTree) return -1;
        memset(newTree, 0, sizeof(OperationCell) * newAllocation);
        if (m_OperationTree) {
            memcpy(newTree, m_OperationTree, sizeof(OperationCell) * m_NbOperations);
            delete[] m_OperationTree;
        }
        m_OperationTree = newTree;
        m_NbAllocatedOperations = newAllocation;
    }

    if (opType == m_NbOperations)
        m_NbOperations++;

    OperationCell &cell = m_OperationTree[opType];
    cell.IsActive = TRUE;
    strncpy(cell.Name, name, 29);
    cell.Name[29] = '\0';
    cell.OperationGuid = OpCode;
    cell.CellCount = 0;
    cell.Tree = nullptr;

    m_OpGuids.InsertUnique(OpCode, opType);
    return opType;
}

CKERROR CKParameterManager::UnRegisterOperationType(CKGUID opguid) {
    CKOperationType type = OperationGuidToCode(opguid);
    return UnRegisterOperationType(type);
}

CKERROR CKParameterManager::UnRegisterOperationType(CKOperationType opcode) {
    if (opcode < 0 || opcode >= m_NbOperations || !m_OperationTree)
        return CKERR_INVALIDOPERATION;

    OperationCell &cell = m_OperationTree[opcode];
    if (!cell.IsActive)
        return CKERR_INVALIDOPERATION;

    m_OpGuids.Remove(cell.OperationGuid);

    cell.IsActive = FALSE;
    if (cell.CellCount > 0 && cell.Tree) {
        for (int i = 0; i < cell.CellCount; ++i) {
            RecurseDelete(&cell.Tree[i]);
        }
        delete[] cell.Tree;
        cell.Tree = nullptr;
        cell.CellCount = 0;
    }

    memset(cell.Name, 0, sizeof(cell.Name));
    cell.OperationGuid = CKGUID();
    return CK_OK;
}

CKERROR CKParameterManager::RegisterOperationFunction(CKGUID &operation, CKGUID &type_paramres, CKGUID &type_param1,
                                                      CKGUID &type_param2, CK_PARAMETEROPERATION op) {
    if (!m_OperationTree)
        return CKERR_INVALIDOPERATION;

    int opCode = OperationGuidToCode(operation);
    if (opCode < 0)
        return CKERR_INVALIDOPERATION;

    OperationCell &opCell = m_OperationTree[opCode];

    TreeCell *param1Cell = FindOrCreateGuidCell(opCell.Tree, opCell.CellCount, type_param1);
    if (!param1Cell) return CKERR_OUTOFMEMORY;
    TreeCell *param2Cell = FindOrCreateGuidCell(param1Cell->Children, param1Cell->ChildCount, type_param2);
    if (!param2Cell) return CKERR_OUTOFMEMORY;
    TreeCell *resultCell = FindOrCreateGuidCell(param2Cell->Children, param2Cell->ChildCount, type_paramres);
    if (!resultCell) return CKERR_OUTOFMEMORY;

    resultCell->Operation = op;
    return CK_OK;
}

CK_PARAMETEROPERATION CKParameterManager::GetOperationFunction(CKGUID &operation, CKGUID &type_paramres,
                                                               CKGUID &type_param1, CKGUID &type_param2) {
    CKOperationType type = OperationGuidToCode(operation);
    if (type < 0 || type >= m_NbOperations || !m_OperationTree)
        return nullptr;

    // 1. Try direct lookup first
    OperationCell &opCell = m_OperationTree[type];
    int pos1 = DichotomicSearch(0, opCell.CellCount - 1, opCell.Tree, type_param1);
    if (pos1 >= 0) {
        TreeCell &param1Cell = opCell.Tree[pos1];
        int pos2 = DichotomicSearch(0, param1Cell.ChildCount - 1, param1Cell.Children, type_param2);
        if (pos2 >= 0) {
            TreeCell &param2Cell = param1Cell.Children[pos2];
            int posRes = DichotomicSearch(0, param2Cell.ChildCount - 1, param2Cell.Children, type_paramres);
            if (posRes >= 0) {
                TreeCell &resultCell = param2Cell.Children[posRes];
                if (resultCell.Operation)
                    return resultCell.Operation;
            }
        }
    }

    // 2. Prepare for type hierarchy fallback
    CKGUID parentRes, parentP1, parentP2;
    CKBOOL hasParentRes = GetParameterGuidParentGuid(type_paramres, parentRes);
    CKBOOL hasParentP1 = GetParameterGuidParentGuid(type_param1, parentP1);
    CKBOOL hasParentP2 = GetParameterGuidParentGuid(type_param2, parentP2);

    // 3. Try different combinations of parent types

    // Combination 1: All parents
    if (hasParentP1 && hasParentP2 && hasParentRes) {
        CK_PARAMETEROPERATION result = GetOperationFunction(operation, parentRes, parentP1, parentP2);
        if (result) return result;
    }

    // Combination 2: Parent param1 and param2
    if (hasParentP1 && hasParentP2) {
        CK_PARAMETEROPERATION result = GetOperationFunction(operation, type_paramres, parentP1, parentP2);
        if (result) return result;
    }

    // Combination 3: Parent param1 and result
    if (hasParentP1 && hasParentRes) {
        CK_PARAMETEROPERATION result = GetOperationFunction(operation, parentRes, parentP1, type_param2);
        if (result) return result;
    }

    // Combination 4: Parent param1
    if (hasParentP1) {
        CK_PARAMETEROPERATION result = GetOperationFunction(operation, type_paramres, parentP1, type_param2);
        if (result) return result;
    }

    // Combination 5: Parent param2 and result
    if (hasParentP2 && hasParentRes) {
        CK_PARAMETEROPERATION result = GetOperationFunction(operation, parentRes, type_param1, parentP2);
        if (result) return result;
    }

    // Combination 6: Parent param2
    if (hasParentP2) {
        CK_PARAMETEROPERATION result = GetOperationFunction(operation, type_paramres, type_param1, parentP2);
        if (result) return result;
    }

    // Combination 7: Parent result
    if (hasParentRes) {
        return GetOperationFunction(operation, parentRes, type_param1, type_param2);
    }

    return nullptr;
}

CKERROR CKParameterManager::UnRegisterOperationFunction(CKGUID &operation, CKGUID &type_paramres, CKGUID &type_param1,
                                                        CKGUID &type_param2) {
    if (!m_OperationTree)
        return CKERR_INVALIDOPERATION;

    int opCode = OperationGuidToCode(operation);
    if (opCode < 0 || opCode >= m_NbOperations)
        return CKERR_INVALIDOPERATION;

    OperationCell &opCell = m_OperationTree[opCode];
    int pos1 = DichotomicSearch(0, opCell.CellCount - 1, opCell.Tree, type_param1);
    if (pos1 < 0) return CKERR_OPERATIONNOTIMPLEMENTED;
    TreeCell &param1Cell = opCell.Tree[pos1];
    int pos2 = DichotomicSearch(0, param1Cell.ChildCount - 1, param1Cell.Children, type_param2);
    if (pos2 < 0) return CKERR_OPERATIONNOTIMPLEMENTED;
    TreeCell &param2Cell = param1Cell.Children[pos2];
    int posRes = DichotomicSearch(0, param2Cell.ChildCount - 1, param2Cell.Children, type_paramres);
    if (posRes < 0) return CKERR_OPERATIONNOTIMPLEMENTED;
    TreeCell &resultCell = param2Cell.Children[posRes];
    resultCell.Operation = nullptr;

    return CK_OK;
}

CKGUID CKParameterManager::OperationCodeToGuid(CKOperationType type) {
    if (CheckOpCodeValidity(type))
        return m_OperationTree[type].OperationGuid;
    return CKGUID();
}

CKSTRING CKParameterManager::OperationCodeToName(CKOperationType type) {
    if (CheckOpCodeValidity(type))
        return m_OperationTree[type].Name;
    return nullptr;
}

CKOperationType CKParameterManager::OperationGuidToCode(CKGUID guid) {
    XHashGuidToType::Iterator it = m_OpGuids.Find(guid);
    if (it != m_OpGuids.End())
        return *it;
    return -1;
}

CKSTRING CKParameterManager::OperationGuidToName(CKGUID guid) {
    CKOperationType type = OperationGuidToCode(guid);
    return OperationCodeToName(type);
}

CKGUID CKParameterManager::OperationNameToGuid(CKSTRING name) {
    if (!name || !m_OperationTree || m_NbOperations <= 0) {
        return CKGUID();
    }

    for (int i = 0; i < m_NbOperations; ++i) {
        const OperationCell &cell = m_OperationTree[i];
        if (cell.IsActive && strcmp(name, cell.Name) == 0) {
            return cell.OperationGuid;
        }
    }
    return CKGUID();
}

CKOperationType CKParameterManager::OperationNameToCode(CKSTRING name) {
    if (m_NbOperations == 0 || !name || !m_OperationTree)
        return -1;

    for (int i = 0; i < m_NbOperations; ++i) {
        const OperationCell &cell = m_OperationTree[i];
        if (cell.IsActive && strcmp(name, cell.Name) == 0) {
            return i;
        }
    }
    return -1;
}

int CKParameterManager::GetAvailableOperationsDesc(const CKGUID &opGuid, CKParameterOut *res, CKParameterIn *p1,
                                                   CKParameterIn *p2, CKOperationDesc *list) {
    if (!m_DerivationMasksUpToDate) {
        UpdateDerivationTables();
    }

    // Initialize GUID buffers for parameter hierarchies
    CKGUID resParents[128], p1Parents[128], p2Parents[128];
    int resParentCount = 0, p1ParentCount = 0, p2ParentCount = 0;

    // 1. Build parameter type hierarchies
    if (p1) p1ParentCount = BuildGuidHierarchy(p1->GetGUID(), p1Parents, 128);
    if (p2) p2ParentCount = BuildGuidHierarchy(p2->GetGUID(), p2Parents, 128);
    if (res) resParentCount = BuildGuidHierarchy(res->GetGUID(), resParents, 128);

    // 2. Find target operation code
    int opCount = 0;
    int startOp = opGuid.IsValid() ? OperationGuidToCode(opGuid) : 0;
    int endOp = opGuid.IsValid() ? startOp + 1 : m_NbOperations;
    if (startOp < 0) startOp = 0;

    // 3. Iterate through relevant operations
    for (int opIdx = startOp; opIdx < endOp; ++opIdx) {
        OperationCell &opCell = m_OperationTree[opIdx];
        if (!opCell.IsActive) continue;

        // 4. Handle parameter type combinations
        ProcessParameterCombinations(
            opCell,
            (p1 ? p1Parents : nullptr), p1ParentCount,
            (p2 ? p2Parents : nullptr), p2ParentCount,
            (res ? resParents : nullptr), resParentCount,
            list,
            opCount
        );
    }
    return opCount;
}

int CKParameterManager::GetParameterOperationCount() {
    return m_NbOperations;
}

void CKParameterManager::ProcessParameterCombinations(OperationCell &opCell, CKGUID *p1Hierarchy, int p1Count,
                                                      CKGUID *p2Hierarchy, int p2Count, CKGUID *resHierarchy,
                                                      int resCount, CKOperationDesc *list, int &opCount) {
    for (int p1Idx = 0; p1Idx < (p1Hierarchy ? p1Count : opCell.CellCount); ++p1Idx) {
        CKGUID p1Guid = p1Hierarchy ? p1Hierarchy[p1Idx] : opCell.Tree[p1Idx].Guid;
        int p1CellIdx = DichotomicSearch(0, opCell.CellCount - 1, opCell.Tree, p1Guid);
        if (p1CellIdx < 0) continue;
        ProcessSecondParameterLevel(*&opCell.Tree[p1CellIdx], p2Hierarchy, p2Count, resHierarchy, resCount, list,
                                    opCount, opCell.OperationGuid, p1Guid);
    }
}

void CKParameterManager::ProcessSecondParameterLevel(TreeCell &p1Cell, CKGUID *p2Hierarchy, int p2Count,
                                                     CKGUID *resHierarchy, int resCount, CKOperationDesc *list,
                                                     int &opCount, const CKGUID &opGuid, const CKGUID &p1Guid) {
    for (int p2Idx = 0; p2Idx < (p2Hierarchy ? p2Count : p1Cell.ChildCount); ++p2Idx) {
        CKGUID p2Guid = p2Hierarchy ? p2Hierarchy[p2Idx] : p1Cell.Children[p2Idx].Guid;
        int p2CellIdx = DichotomicSearch(0, p1Cell.ChildCount - 1, p1Cell.Children, p2Guid);
        if (p2CellIdx < 0) continue;
        ProcessResultTypeLevel(*&p1Cell.Children[p2CellIdx], resHierarchy, resCount, list, opCount, opGuid, p1Guid,
                               p2Guid);
    }
}

void CKParameterManager::ProcessResultTypeLevel(TreeCell &p2Cell, CKGUID *resHierarchy, int resCount,
                                                CKOperationDesc *list, int &opCount, const CKGUID &opGuid,
                                                const CKGUID &p1Guid, const CKGUID &p2Guid) {
    for (int resIdx = 0; resIdx < (resHierarchy ? resCount : p2Cell.ChildCount); ++resIdx) {
        CKGUID resGuid = resHierarchy ? resHierarchy[resIdx] : p2Cell.Children[resIdx].Guid;
        int resCellIdx = DichotomicSearch(0, p2Cell.ChildCount - 1, p2Cell.Children, resGuid);
        if (resCellIdx < 0) continue;
        TreeCell &resCell = p2Cell.Children[resCellIdx];
        if (!resCell.Operation) continue;

        if (list) {
            CKOperationDesc &desc = list[opCount];
            desc.OpGuid = opGuid;
            desc.P1Guid = p1Guid;
            desc.P2Guid = p2Guid;
            desc.ResGuid = resCell.Guid;
            desc.Fct = resCell.Operation;
        }
        opCount++;
        if (!resHierarchy) {
            AddCompatibleDerivedOperations(resCell, list, opCount, opGuid, p1Guid, p2Guid);
        }
    }
}

void CKParameterManager::AddCompatibleDerivedOperations(TreeCell &resCell, CKOperationDesc *list, int &opCount,
                                                        const CKGUID &opGuid, const CKGUID &p1Guid, const CKGUID &p2Guid) {
    for (int i = 0; i < m_RegisteredTypes.Size(); ++i) {
        CKParameterTypeDesc *typeDesc = m_RegisteredTypes[i];
        if (!typeDesc || (typeDesc->dwFlags & (CKPARAMETERTYPE_FLAGS | CKPARAMETERTYPE_STRUCT | CKPARAMETERTYPE_ENUMS)))
            continue;
        if (IsDerivedFrom(typeDesc->Guid, resCell.Guid)) {
            if (list) {
                CKOperationDesc &desc = list[opCount];
                desc.OpGuid = opGuid;
                desc.P1Guid = p1Guid;
                desc.P2Guid = p2Guid;
                desc.ResGuid = typeDesc->Guid;
                desc.Fct = resCell.Operation;
            }
            opCount++;
        }
    }
}

TreeCell *CKParameterManager::FindOrCreateGuidCell(TreeCell *&cells, int &cellCount, CKGUID &guid) {
    int pos = DichotomicSearch(0, cellCount - 1, cells, guid);
    if (pos >= 0) return &cells[pos];

    pos = -(pos + 1);
    TreeCell *newCells = new TreeCell[cellCount + 1];
    if (!newCells) return nullptr;

    if (cells) {
        if (pos > 0) memcpy(newCells, cells, sizeof(TreeCell) * pos);
        if (pos < cellCount) memcpy(newCells + pos + 1, cells + pos, sizeof(TreeCell) * (cellCount - pos));
        delete[] cells;
    }

    cells = newCells;
    cellCount++;
    cells[pos].Guid = guid;
    cells[pos].ChildCount = 0;
    cells[pos].Children = nullptr;
    return &cells[pos];
}

CKBOOL CKParameterManager::IsParameterTypeToBeShown(CKParameterType type) {
    if (type < 0 || type >= m_RegisteredTypes.Size()) return FALSE;
    CKParameterTypeDesc *desc = m_RegisteredTypes[type];
    return desc ? !(desc->dwFlags & CKPARAMETERTYPE_HIDDEN) : FALSE;
}

CKBOOL CKParameterManager::IsParameterTypeToBeShown(CKGUID guid) {
    CKParameterType type = ParameterGuidToType(guid);
    return IsParameterTypeToBeShown(type);
}

void CKParameterManager::UpdateParameterEnum() {
    XString enumString;
    XClassArray<XString> paramNames;

    for (int i = 0; i < GetParameterTypesCount(); ++i) {
        CKParameterTypeDesc *typeDesc = GetParameterTypeDescription(i);
        if (!typeDesc || !typeDesc->Valid || (typeDesc->dwFlags & CKPARAMETERTYPE_HIDDEN))
            continue;

        XString formattedName(typeDesc->TypeName);
        formattedName << "=" << i;
        paramNames.PushBack(formattedName);
    }
    paramNames.Sort();

    for (int i = 0; i < paramNames.Size(); ++i) {
        enumString << paramNames[i] << (i < paramNames.Size() - 1 ? "," : "");
    }

    ChangeEnumDeclaration(CKPGUID_PARAMETERTYPE, enumString.Str());
    m_ParameterTypeEnumUpToDate = TRUE;
}

CKParameterManager::CKParameterManager(CKContext *Context) : CKBaseManager(Context, PARAMETER_MANAGER_GUID, "Parameter Manager") {
    m_NbOperations = 0;
    m_NbAllocatedOperations = 0;
    m_OperationTree = nullptr;
    m_DerivationMasksUpToDate = FALSE;
    m_NbFlagsDefined = 0;
    m_Flags = nullptr;
    m_NbStructDefined = 0;
    m_Structs = nullptr;
    m_NbEnumsDefined = 0;
    m_Enums = nullptr;
    m_ParameterTypeEnumUpToDate = FALSE;
    m_Context->RegisterNewManager(this);
    CKInitializeParameterTypes(this);
}

CKParameterManager::~CKParameterManager() {
    RemoveAllOperations();
    RemoveAllParameterTypes();
}

CKBOOL CKParameterManager::CheckParamTypeValidity(CKParameterType type) {
    return type >= 0 && type < m_RegisteredTypes.Size() && m_RegisteredTypes[type] != nullptr;
}

CKBOOL CKParameterManager::CheckOpCodeValidity(CKOperationType type) {
    return type >= 0 && type < m_NbOperations && m_OperationTree && m_OperationTree[type].IsActive;
}

CKBOOL CKParameterManager::GetParameterGuidParentGuid(CKGUID child, CKGUID &parent) {
    CKParameterType type = ParameterGuidToType(child);
    if (CheckParamTypeValidity(type)) {
        parent = m_RegisteredTypes[type]->DerivedFrom;
        if (parent.IsValid()) return TRUE;
    }
    return FALSE;
}

void CKParameterManager::UpdateDerivationTables() {
    for (int i = 0; i < m_RegisteredTypes.Size(); ++i) {
        CKParameterTypeDesc *child = m_RegisteredTypes[i];
        if (!child) continue;
        child->DerivationMask.Clear();
        for (int j = 0; j < m_RegisteredTypes.Size(); ++j) {
            if (m_RegisteredTypes[j] && IsDerivedFromIntern(i, j)) {
                child->DerivationMask.Set(j);
            }
        }
    }
    m_DerivationMasksUpToDate = TRUE;
}

void CKParameterManager::RecurseDeleteParam(TreeCell *cell, CKGUID param) {
    if (!cell || cell->ChildCount <= 0) return;

    for (int i = 0; i < cell->ChildCount; ++i) {
        TreeCell &child = cell->Children[i];
        if (child.Guid == param) {
            RecurseDelete(&child);
            const int newCount = cell->ChildCount - 1;
            TreeCell *newChildren = newCount > 0 ? new TreeCell[newCount] : nullptr;
            if (newChildren) {
                if (i > 0) memcpy(newChildren, cell->Children, i * sizeof(TreeCell));
                if (i < newCount) memcpy(newChildren + i, cell->Children + i + 1, (newCount - i) * sizeof(TreeCell));
            }
            delete[] cell->Children;
            cell->Children = newChildren;
            cell->ChildCount = newCount;
            --i; // Re-evaluate current index
        } else {
            RecurseDeleteParam(&child, param);
        }
    }
}

int CKParameterManager::DichotomicSearch(int start, int end, TreeCell *tab, CKGUID key) {
    if (!tab) return -1;
    while (start <= end) {
        int mid = start + (end - start) / 2;
        CKGUID &current = tab[mid].Guid;
        if (current == key) return mid;
        if (current > key) end = mid - 1;
        else start = mid + 1;
    }
    return -(start + 1);
}

void CKParameterManager::RecurseDelete(TreeCell *cell) {
    if (cell && cell->ChildCount > 0) {
        for (int i = 0; i < cell->ChildCount; ++i) {
            RecurseDelete(&cell->Children[i]);
        }
        delete[] cell->Children;
        cell->Children = nullptr;
        cell->ChildCount = 0;
    }
}

CKBOOL CKParameterManager::IsDerivedFromIntern(int childIdx, int parentIdx) {
    if (childIdx == parentIdx) return TRUE;

    CKGUID parentGuid = ParameterTypeToGuid(parentIdx);
    CKParameterTypeDesc *childDesc = GetParameterTypeDescription(childIdx);
    if (!childDesc || !GetParameterTypeDescription(parentIdx)) return FALSE;

    if (childDesc->DerivedFrom == parentGuid) return TRUE;

    // Termination conditions: null GUID or CKPGUID_NONE
    if (!childDesc->DerivedFrom.IsValid() || childDesc->DerivedFrom == CKPGUID_NONE)
        return FALSE;

    return IsDerivedFromIntern(ParameterGuidToType(childDesc->DerivedFrom), parentIdx);
}

CKBOOL CKParameterManager::RemoveAllParameterTypes() {
    m_ParamGuids.Clear();

    if (m_Enums) {
        for (int i = 0; i < m_NbEnumsDefined; ++i) {
            for (int j = 0; j < m_Enums[i].NbData; ++j) delete[] m_Enums[i].Desc[j];
            delete[] m_Enums[i].Desc;
            delete[] m_Enums[i].Vals;
        }
        delete[] m_Enums;
        m_Enums = nullptr;
        m_NbEnumsDefined = 0;
    }
    if (m_Flags) {
        for (int i = 0; i < m_NbFlagsDefined; ++i) {
            for (int j = 0; j < m_Flags[i].NbData; ++j) delete[] m_Flags[i].Desc[j];
            delete[] m_Flags[i].Desc;
            delete[] m_Flags[i].Vals;
        }
        delete[] m_Flags;
        m_Flags = nullptr;
        m_NbFlagsDefined = 0;
    }
    if (m_Structs) {
        for (int i = 0; i < m_NbStructDefined; ++i) {
            for (int j = 0; j < m_Structs[i].NbData; ++j) delete[] m_Structs[i].Desc[j];
            delete[] m_Structs[i].Desc;
            delete[] m_Structs[i].Guids;
        }
        delete[] m_Structs;
        m_Structs = nullptr;
        m_NbStructDefined = 0;
    }

    for (int i = 0; i < m_RegisteredTypes.Size(); ++i) delete m_RegisteredTypes[i];
    m_RegisteredTypes.Clear();

    return TRUE;
}

CKBOOL CKParameterManager::RemoveAllOperations() {
    if (m_OperationTree) {
        for (CKOperationType i = 0; i < m_NbOperations; ++i) {
            UnRegisterOperationType(i);
        }
        delete[] m_OperationTree;
        m_OperationTree = nullptr;
    }
    m_NbOperations = 0;
    m_NbAllocatedOperations = 0;
    m_OpGuids.Clear();
    return TRUE;
}

int CKParameterManager::BuildGuidHierarchy(CKGUID guid, CKGUID *buffer, int max) {
    int count = 0;
    buffer[count++] = guid;
    CKGUID parent;

    while (count < max && GetParameterGuidParentGuid(guid, parent)) {
        buffer[count++] = parent;
        guid = parent;
    }
    return count;
}

CKStructHelper::CKStructHelper(CKParameter *Param) {
    m_Context = Param ? Param->GetCKContext() : nullptr;
    if (m_Context) {
        m_StructDescription = m_Context->GetParameterManager()->GetStructDescByType(Param->GetType());
        m_SubIDS = (CK_ID *) Param->GetReadDataPtr();
    } else {
        m_StructDescription = nullptr;
        m_SubIDS = nullptr;
    }
}

CKStructHelper::CKStructHelper(CKContext *ctx, CKGUID PGuid, CK_ID *Data) {
    m_Context = ctx;
    if (ctx) {
        CKParameterManager *pm = ctx->GetParameterManager();
        CKParameterType type = pm->ParameterGuidToType(PGuid);
        m_StructDescription = ctx->GetParameterManager()->GetStructDescByType(type);
    } else {
        m_StructDescription = nullptr;
    }
    m_SubIDS = Data;
}

CKStructHelper::CKStructHelper(CKContext *ctx, CKParameterType PType, CK_ID *Data) {
    m_Context = ctx;
    if (ctx) {
        m_StructDescription = ctx->GetParameterManager()->GetStructDescByType(PType);
    } else {
        m_StructDescription = nullptr;
    }
    m_SubIDS = Data;
}

char *CKStructHelper::GetMemberName(int Pos) {
    if (m_StructDescription && Pos >= 0 && Pos < m_StructDescription->NbData) {
        return m_StructDescription->Desc[Pos];
    }
    return nullptr;
}

CKGUID CKStructHelper::GetMemberGUID(int Pos) {
    if (m_StructDescription && Pos >= 0 && Pos < m_StructDescription->NbData) {
        return m_StructDescription->Guids[Pos];
    }
    return CKGUID();
}

int CKStructHelper::GetMemberCount() {
    if (m_StructDescription) {
        return m_StructDescription->NbData;
    }
    return 0;
}
