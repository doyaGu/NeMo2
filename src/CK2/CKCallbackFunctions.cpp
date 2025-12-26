#include "CKCallbackFunctions.h"

#include "VxMath.h"
#include "CKStateChunk.h"
#include "CKAttributeManager.h"
#include "CKParameterManager.h"
#include "CKMessageManager.h"
#include "CKInterfaceManager.h"
#include "CK2dCurve.h"

void CKInitializeParameterTypes(CKParameterManager *man) {
    CKParameterTypeDesc noneTypeDesc;
    noneTypeDesc.Guid = CKPGUID_NONE;
    noneTypeDesc.TypeName = "None";
    noneTypeDesc.Valid = TRUE;
    man->RegisterParameterType(&noneTypeDesc);

    CKParameterTypeDesc pointerTypeDesc;
    pointerTypeDesc.Guid = CKPGUID_POINTER;
    pointerTypeDesc.TypeName = "Pointer";
    pointerTypeDesc.Valid = TRUE;
    pointerTypeDesc.DefaultSize = sizeof(void *);
    pointerTypeDesc.CopyFunction = CK_ParameterCopier_Nope;
    pointerTypeDesc.SaveLoadFunction = CKPointerSaver;
    pointerTypeDesc.dwFlags = CKPARAMETERTYPE_HIDDEN;
    man->RegisterParameterType(&pointerTypeDesc);

    CKParameterTypeDesc bufferTypeDesc;
    bufferTypeDesc.Guid = CKPGUID_VOIDBUF;
    bufferTypeDesc.TypeName = "Buffer";
    bufferTypeDesc.Valid = TRUE;
    bufferTypeDesc.DefaultSize = 10;
    bufferTypeDesc.CopyFunction = CK_ParameterCopier_SetValue;
    bufferTypeDesc.dwFlags = CKPARAMETERTYPE_VARIABLESIZE | CKPARAMETERTYPE_HIDDEN | CKPARAMETERTYPE_NOENDIANCONV;
    man->RegisterParameterType(&bufferTypeDesc);

    CKParameterTypeDesc copyDependenciesTypeDesc;
    copyDependenciesTypeDesc.Guid = CKPGUID_COPYDEPENDENCIES;
    copyDependenciesTypeDesc.TypeName = "Copy Dependencies";
    copyDependenciesTypeDesc.Valid = TRUE;
    copyDependenciesTypeDesc.DefaultSize = sizeof(CKDWORD);
    copyDependenciesTypeDesc.CreateDefaultFunction = CKDependenciesCreator;
    copyDependenciesTypeDesc.DeleteFunction = CKDependenciesDestructor;
    copyDependenciesTypeDesc.SaveLoadFunction = CKDependenciesSaver;
    copyDependenciesTypeDesc.CopyFunction = CKDependenciesCopier;
    man->RegisterParameterType(&copyDependenciesTypeDesc);

    CKParameterTypeDesc deleteDependenciesTypeDesc = copyDependenciesTypeDesc;
    deleteDependenciesTypeDesc.Guid = CKPGUID_DELETEDEPENDENCIES;
    deleteDependenciesTypeDesc.TypeName = "Delete Dependencies";
    man->RegisterParameterType(&deleteDependenciesTypeDesc);

    CKParameterTypeDesc replaceDependenciesTypeDesc = copyDependenciesTypeDesc;
    replaceDependenciesTypeDesc.Guid = CKPGUID_REPLACEDEPENDENCIES;
    replaceDependenciesTypeDesc.TypeName = "Replace Dependencies";
    man->RegisterParameterType(&replaceDependenciesTypeDesc);

    CKParameterTypeDesc saveDependenciesTypeDesc = copyDependenciesTypeDesc;
    saveDependenciesTypeDesc.Guid = CKPGUID_SAVEDEPENDENCIES;
    saveDependenciesTypeDesc.TypeName = "Save Dependencies";
    man->RegisterParameterType(&saveDependenciesTypeDesc);

    CKParameterTypeDesc stateChunkTypeDesc;
    stateChunkTypeDesc.Guid = CKPGUID_STATECHUNK;
    stateChunkTypeDesc.TypeName = "State Chunk";
    stateChunkTypeDesc.Valid = TRUE;
    stateChunkTypeDesc.DefaultSize = sizeof(CKStateChunk *);
    stateChunkTypeDesc.CreateDefaultFunction = CKStateChunkCreator;
    stateChunkTypeDesc.DeleteFunction = CKStateChunkDestructor;
    stateChunkTypeDesc.SaveLoadFunction = CKStateChunkSaver;
    stateChunkTypeDesc.CopyFunction = CKStateChunkCopier;
    man->RegisterParameterType(&stateChunkTypeDesc);

    CKParameterTypeDesc floatTypeDesc;
    floatTypeDesc.Guid = CKPGUID_FLOAT;
    floatTypeDesc.TypeName = "Float";
    floatTypeDesc.Valid = TRUE;
    floatTypeDesc.DefaultSize = sizeof(float);
    floatTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    floatTypeDesc.StringFunction = CKFloatStringFunc;
    man->RegisterParameterType(&floatTypeDesc);

    CKParameterTypeDesc percentageTypeDesc;
    percentageTypeDesc.Guid = CKPGUID_PERCENTAGE;
    percentageTypeDesc.DerivedFrom = CKPGUID_FLOAT;
    percentageTypeDesc.TypeName = "Percentage";
    percentageTypeDesc.Valid = TRUE;
    percentageTypeDesc.DefaultSize = sizeof(float);
    percentageTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    percentageTypeDesc.StringFunction = CKPercentageStringFunc;
    man->RegisterParameterType(&percentageTypeDesc);

    CKParameterTypeDesc angleTypeDesc;
    angleTypeDesc.Guid = CKPGUID_ANGLE;
    angleTypeDesc.DerivedFrom = CKPGUID_FLOAT;
    angleTypeDesc.TypeName = "Angle";
    angleTypeDesc.Valid = TRUE;
    angleTypeDesc.DefaultSize = sizeof(float);
    angleTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    angleTypeDesc.StringFunction = CKAngleStringFunc;
    man->RegisterParameterType(&angleTypeDesc);

    CKParameterTypeDesc intTypeDesc;
    intTypeDesc.Guid = CKPGUID_INT;
    intTypeDesc.TypeName = "Integer";
    intTypeDesc.Valid = TRUE;
    intTypeDesc.DefaultSize = sizeof(int);
    intTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    intTypeDesc.StringFunction = CKIntStringFunc;
    man->RegisterParameterType(&intTypeDesc);

    CKParameterTypeDesc timeTypeDesc;
    timeTypeDesc.Guid = CKPGUID_TIME;
    timeTypeDesc.DerivedFrom = CKPGUID_FLOAT;
    timeTypeDesc.TypeName = "Time";
    timeTypeDesc.Valid = TRUE;
    timeTypeDesc.DefaultSize = sizeof(float);
    timeTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    timeTypeDesc.StringFunction = CKTimeStringFunc;
    man->RegisterParameterType(&timeTypeDesc);

    CKParameterTypeDesc boolTypeDesc;
    boolTypeDesc.Guid = CKPGUID_BOOL;
    boolTypeDesc.DerivedFrom = CKPGUID_INT;
    boolTypeDesc.TypeName = "Boolean";
    boolTypeDesc.Valid = TRUE;
    boolTypeDesc.DefaultSize = sizeof(int);
    boolTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    boolTypeDesc.StringFunction = CKBoolStringFunc;
    man->RegisterParameterType(&boolTypeDesc);

    CKParameterTypeDesc stringTypeDesc;
    stringTypeDesc.Guid = CKPGUID_STRING;
    stringTypeDesc.TypeName = "String";
    stringTypeDesc.Valid = TRUE;
    stringTypeDesc.DefaultSize = 64;
    stringTypeDesc.CopyFunction = CK_ParameterCopier_SetValue;
    stringTypeDesc.StringFunction = CKStringStringFunc;
    stringTypeDesc.dwFlags = CKPARAMETERTYPE_VARIABLESIZE | CKPARAMETERTYPE_NOENDIANCONV;
    man->RegisterParameterType(&stringTypeDesc);

    CKParameterTypeDesc vectorTypeDesc;
    vectorTypeDesc.Guid = CKPGUID_VECTOR;
    vectorTypeDesc.TypeName = "Vector";
    vectorTypeDesc.Valid = TRUE;
    vectorTypeDesc.DefaultSize = sizeof(VxVector);
    vectorTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    vectorTypeDesc.StringFunction = CKVectorStringFunc;
    man->RegisterParameterType(&vectorTypeDesc);

    CKParameterTypeDesc quaternionTypeDesc;
    quaternionTypeDesc.Guid = CKPGUID_QUATERNION;
    quaternionTypeDesc.TypeName = "Quaternion";
    quaternionTypeDesc.Valid = TRUE;
    quaternionTypeDesc.DefaultSize = sizeof(VxQuaternion);
    quaternionTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    quaternionTypeDesc.StringFunction = CKQuaternionStringFunc;
    man->RegisterParameterType(&quaternionTypeDesc);

    CKParameterTypeDesc eulerAnglesTypeDesc;
    eulerAnglesTypeDesc.Guid = CKPGUID_EULERANGLES;
    eulerAnglesTypeDesc.DerivedFrom = CKPGUID_VECTOR;
    eulerAnglesTypeDesc.TypeName = "Euler Angles";
    eulerAnglesTypeDesc.Valid = TRUE;
    eulerAnglesTypeDesc.DefaultSize = sizeof(VxVector);
    eulerAnglesTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    eulerAnglesTypeDesc.StringFunction = CKEulerStringFunc;
    man->RegisterParameterType(&eulerAnglesTypeDesc);

    CKParameterTypeDesc rectTypeDesc;
    rectTypeDesc.Guid = CKPGUID_RECT;
    rectTypeDesc.TypeName = "Rectangle";
    rectTypeDesc.Valid = TRUE;
    rectTypeDesc.DefaultSize = sizeof(VxRect);
    rectTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    rectTypeDesc.StringFunction = CKRectStringFunc;
    man->RegisterParameterType(&rectTypeDesc);

    CKParameterTypeDesc vector2DTypeDesc;
    vector2DTypeDesc.Guid = CKPGUID_2DVECTOR;
    vector2DTypeDesc.TypeName = "Vector 2D";
    vector2DTypeDesc.Valid = TRUE;
    vector2DTypeDesc.DefaultSize = sizeof(Vx2DVector);
    vector2DTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    vector2DTypeDesc.StringFunction = CK2DVectorStringFunc;
    man->RegisterParameterType(&vector2DTypeDesc);

    CKParameterTypeDesc matrixTypeDesc;
    matrixTypeDesc.Guid = CKPGUID_MATRIX;
    matrixTypeDesc.TypeName = "Matrix";
    matrixTypeDesc.Valid = TRUE;
    matrixTypeDesc.DefaultSize = sizeof(VxMatrix);
    matrixTypeDesc.CreateDefaultFunction = CKMatrixCreator;
    matrixTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    matrixTypeDesc.StringFunction = CKMatrixStringFunc;
    man->RegisterParameterType(&matrixTypeDesc);

    CKParameterTypeDesc colorTypeDesc;
    colorTypeDesc.Guid = CKPGUID_COLOR;
    colorTypeDesc.TypeName = "Color";
    colorTypeDesc.Valid = TRUE;
    colorTypeDesc.DefaultSize = sizeof(VxColor);
    colorTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    colorTypeDesc.StringFunction = CKColorStringFunc;
    man->RegisterParameterType(&colorTypeDesc);

    CKParameterTypeDesc boxTypeDesc;
    boxTypeDesc.Guid = CKPGUID_BOX;
    boxTypeDesc.TypeName = "Box";
    boxTypeDesc.Valid = TRUE;
    boxTypeDesc.DefaultSize = sizeof(VxBbox);
    boxTypeDesc.CopyFunction = CK_ParameterCopier_Memcpy;
    boxTypeDesc.StringFunction = CKBoxStringFunc;
    man->RegisterParameterType(&boxTypeDesc);

    CKParameterTypeDesc collectionTypeDesc;
    collectionTypeDesc.Guid = CKPGUID_OBJECTARRAY;
    collectionTypeDesc.TypeName = "Collection";
    collectionTypeDesc.Valid = TRUE;
    collectionTypeDesc.DefaultSize = sizeof(CKDWORD);
    collectionTypeDesc.CreateDefaultFunction = CKCollectionCreator;
    collectionTypeDesc.DeleteFunction = CKCollectionDestructor;
    collectionTypeDesc.SaveLoadFunction = CKCollectionSaver;
    collectionTypeDesc.CheckFunction = CKCollectionCheck;
    collectionTypeDesc.CopyFunction = CKCollectionCopy;
    collectionTypeDesc.StringFunction = CKCollectionStringFunc;
    man->RegisterParameterType(&collectionTypeDesc);

    CKParameterTypeDesc messageTypeDesc;
    messageTypeDesc.Guid = CKPGUID_MESSAGE;
    messageTypeDesc.TypeName = "Message";
    messageTypeDesc.Valid = TRUE;
    messageTypeDesc.DefaultSize = sizeof(CKDWORD);
    messageTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    messageTypeDesc.StringFunction = CKMessageStringFunc;
    messageTypeDesc.Saver_Manager = MESSAGE_MANAGER_GUID;
    man->RegisterParameterType(&messageTypeDesc);

    CKParameterTypeDesc attributeTypeDesc;
    attributeTypeDesc.Guid = CKPGUID_ATTRIBUTE;
    attributeTypeDesc.TypeName = "Attribute";
    attributeTypeDesc.Valid = TRUE;
    attributeTypeDesc.DefaultSize = sizeof(CKDWORD);
    attributeTypeDesc.CopyFunction = CK_ParameterCopier_Dword;
    attributeTypeDesc.StringFunction = CKAttributeStringFunc;
    attributeTypeDesc.Saver_Manager = ATTRIBUTE_MANAGER_GUID;
    man->RegisterParameterType(&attributeTypeDesc);

    CKParameterTypeDesc curve2DTypeDesc;
    curve2DTypeDesc.Guid = CKPGUID_2DCURVE;
    curve2DTypeDesc.TypeName = "2D Curve";
    curve2DTypeDesc.Valid = TRUE;
    curve2DTypeDesc.DefaultSize = sizeof(CKDWORD);
    curve2DTypeDesc.CreateDefaultFunction = CK2dCurveCreator;
    curve2DTypeDesc.DeleteFunction = CK2dCurveDestructor;
    curve2DTypeDesc.SaveLoadFunction = CK2dCurveSaver;
    curve2DTypeDesc.CopyFunction = CK2dCurveCopier;
    man->RegisterParameterType(&curve2DTypeDesc);

    // Object parameter template - used for class parameter types below
    CKParameterTypeDesc objectParamTemplate;
    objectParamTemplate.Valid = TRUE;
    objectParamTemplate.DefaultSize = sizeof(CKDWORD);
    objectParamTemplate.CheckFunction = CKObjectCheckFunc;
    objectParamTemplate.CopyFunction = CK_ParameterCopier_Dword;
    objectParamTemplate.StringFunction = CKObjectStringFunc;
    // Note: This template is NOT registered directly

    CKParameterTypeDesc scriptTypeDesc(objectParamTemplate);
    scriptTypeDesc.Guid = CKPGUID_SCRIPT;
    scriptTypeDesc.DerivedFrom = CKPGUID_OBJECT;
    scriptTypeDesc.TypeName = "Script";
    scriptTypeDesc.UICreatorFunction = nullptr;
    scriptTypeDesc.Cid = CKCID_BEHAVIOR;
    man->RegisterParameterType(&scriptTypeDesc);

    man->RegisterNewEnum(CKPGUID_DIRECTION, "Direction", "X=1,-X=2,Y=3,-Y=4,Z=5,-Z=6");
    man->RegisterNewEnum(CKPGUID_BLENDMODE, "Texture Blend Mode",
                         "Decal=1,Modulate=2,DecalAlpha=3,ModulateAlpha=4,DecalMask=5,ModulateMask=6,Copy=7,Add=8");
    man->RegisterNewEnum(CKPGUID_FILTERMODE, "Filter Mode",
                         "Nearest=1,Linear=2,MipNearest=3,MipLinear=4,LinearMipNearest=5,LinearMipLinear=6,Anisotropic=7");
    man->RegisterNewEnum(CKPGUID_BLENDFACTOR, "Blend Factor",
                         "Zero=1,One=2,Source Color=3,Inverse Source Color=4,Source Alpha=5,Inverse Source Alpha=6,Destination Alpha=7,Inverse Destination Alpha=8,Destination Color=9,Inverse Destination Color=10,Source Alpha Saturation=11,Both Source Alpha=12,Both Inverse Source Alpha=13");
    man->RegisterNewEnum(CKPGUID_FILLMODE, "Fill Mode", "Point=1,WireFrame=2,Solid=3,OnlyZ=4");
    man->RegisterNewEnum(CKPGUID_LITMODE, "Lit Mode", "Prelit=0,Lit=1");
    man->RegisterNewEnum(CKPGUID_SHADEMODE, "Shade Mode", "Flat=1,Gouraud=2");
    man->RegisterNewEnum(CKPGUID_GLOBALEXMODE, "Global Shade Mode", "Wireframe=0,Flat=1,Gouraud=2,Material Default=4");
    man->RegisterNewEnum(CKPGUID_ZFUNC, "Z Compare Function",
                         "Never=1,Less=2,Equal=3,Less Equal=4,Greater=5,Not Equal=6,Greater Equal=7,Always=8");
    man->RegisterNewEnum(CKPGUID_ADDRESSMODE, "Address Mode", "Wrap=1,Mirror=2,Clamp=3,Border=4");
    man->RegisterNewEnum(CKPGUID_WRAPMODE, "Wrap Mode", "None=0,V Looping=1,U Looping=2,UV Looping=3");
    man->RegisterNewEnum(CKPGUID_3DSPRITEMODE, "3DSprite Mode", "Billboard=0,XRotate=1,YRotate=2,Orientable=3");
    man->RegisterNewEnum(CKPGUID_FOGMODE, "Fog Mode", "None=0,Exponential=1,Exponential Squared=2,Linear=3");
    man->RegisterNewEnum(CKPGUID_LIGHTTYPE, "Light Type", "Point=1,Spot=2,Directional=3");
    man->RegisterNewEnum(CKPGUID_COMPOPERATOR, "Comparison Operator",
                         "Equal=1,Not Equal=2,Less than=3,Less or equal=4,Greater than=5,Greater or equal=6");
    man->RegisterNewEnum(CKPGUID_BINARYOPERATOR, "Binary Operator", "+=1,-=2,*=3,/=4");
    man->RegisterNewEnum(CKPGUID_SETOPERATOR, "Sets Operator", "Union=1,Intersection=2,Subtraction=3");
    man->RegisterNewEnum(CKPGUID_SPRITETEXTALIGNMENT, "Sprite Text Alignment",
                         "Center=1,Left=2,Right=4,Top=8,Bottom=16,VCenter=32,HCenter=64");
    man->RegisterNewEnum(CKPGUID_ARRAYTYPE, "Array Column Type", "Integer=1,Float=2,String=3,Object=4,Parameter=5");
    man->RegisterNewFlags(
        CKPGUID_FILTER, "Filter",
        "1=1,2=2,3=4,4=8,5=16,6=32,7=64,8=128,9=256,10=512,11=1024,12=2048,13=4096,14=8192,15=16384,16=32768,17=65536,18=131072,19=262144,20=524288,21=1048576,22=2097152,23=4194304,24=8388608,25=16777216,26=33554432,27=67108864,28=134217728,29=268435456,30=536870912,31=1073741824,32=2147483648");
    man->RegisterNewEnum(CKPGUID_SCENEACTIVITYFLAGS, "Scene Activity Options",
                         "Scene Defaults=0,Force activate=1,Force deactivate=2,Do nothing=3");
    man->RegisterNewEnum(CKPGUID_SCENERESETFLAGS, "Scene Reset Options", "Scene Defaults=0,Force Reset=1,Do nothing=2");
    man->RegisterNewFlags(CKPGUID_RENDEROPTIONS, "Render Options",
                          "Background Sprites=1,Foreground Sprites=2,Use Camera Ratio=8,Clear ZBuffer=16,Clear Back Buffer=32,Clear Stencil Buffer=64,Buffer Swapping=128,Clear Only Viewport=256,Vertical Sync=512,Disable 3D=1024");


    XString enumData;
    const int classCount = CKGetClassCount();
    XClassArray<XString> classNames(classCount);

    for (CK_CLASSID classId = CKCID_OBJECT; classId < classCount; ++classId) {
        CKClassDesc *classDesc = CKGetClassDesc(classId);
        if (!classDesc || !classDesc->NameFct) continue;

        // Get class name
        XString className(classDesc->NameFct());

        if (classDesc->Parameter != CKGUID(0, 0)) {
            // Reuse objectParamTemplate - set class-specific fields
            objectParamTemplate.Cid = classId;
            objectParamTemplate.TypeName = className;
            objectParamTemplate.Guid = classDesc->Parameter;
            objectParamTemplate.DerivedFrom = CKGUID(0, 0);

            CKClassDesc *parentDesc = classDesc;
            while (parentDesc->Parent != 0) {
                parentDesc = CKGetClassDesc(parentDesc->Parent);
                if (!parentDesc)
                    break;
                if (parentDesc->Parameter != CKGUID(0, 0)) {
                    objectParamTemplate.DerivedFrom = parentDesc->Parameter;
                    break;
                }
            }

            man->RegisterParameterType(&objectParamTemplate);
        }

        className << "=" << (int)classId;
        classNames.PushBack(className);
    }

    classNames.Sort();

    for (int i = 0; i < classNames.Size(); ++i) {
        enumData << classNames[i];
        if (i < classNames.Size() - 1) {
            enumData << ",";
        }
    }

    if (enumData.Length() == 0) {
        enumData = "";
    }

    man->RegisterNewEnum(CKPGUID_CLASSID, "Registered Class", enumData.Str());
    man->RegisterNewEnum(CKPGUID_PARAMETERTYPE, "Registered Parameter", "NA=0");
    CKParameterTypeDesc *desc = man->GetParameterTypeDescription(CKPGUID_PARAMETERTYPE);
    if (desc) {
        desc->CreateDefaultFunction = CKParameterTypesEnumCreator;
    }
}

void CK_ParameterCopier_SaveLoad(CKParameter *dest, CKParameter *src) {
    CKStateChunk *stateChunk = nullptr;
    src->m_ParamType->SaveLoadFunction(src, &stateChunk, FALSE);
    dest->m_ParamType->SaveLoadFunction(dest, &stateChunk, TRUE);
    if (stateChunk) {
        delete stateChunk;
    }
}

void CK_ParameterCopier_SetValue(CKParameter *dest, CKParameter *src) {
    dest->SetValue(src->m_Buffer, src->m_Size);
}

void CK_ParameterCopier_Dword(CKParameter *dest, CKParameter *src) {
    CKDWORD *srcValue = (CKDWORD *) src->m_Buffer;
    CKDWORD *destValue = (CKDWORD *) dest->m_Buffer;
    *destValue = *srcValue;
}

void CK_ParameterCopier_Memcpy(CKParameter *dest, CKParameter *src) {
    memcpy(dest->m_Buffer, src->m_Buffer, dest->m_Size);
}

void CK_ParameterCopier_BufferCompatibleWithDWORD(CKParameter *dest, CKParameter *src) {
    if (src->GetDataSize() == sizeof(CKDWORD)) {
        dest->SetValue(src->GetReadDataPtr(), sizeof(CKDWORD));
    } else {
        memcpy(dest->GetWriteDataPtr(), src->GetReadDataPtr(), src->GetDataSize());
    }
}

void CK_ParameterCopier_Nope(CKParameter *dest, CKParameter *src) {
    // Do nothing
}

CKERROR CKParameterTypesEnumCreator(CKParameter *param) {
    CKParameterManager *pm = param->m_Context->GetParameterManager();
    if (!pm->m_ParameterTypeEnumUpToDate) {
        pm->UpdateParameterEnum();
    }

    int valueToSet = 23;
    param->SetValue(&valueToSet, 0);
    return CK_OK;
}

CKERROR CKDependenciesCreator(CKParameter *param) {
    CK_DEPENDENCIES_OPMODE opMode = CK_DEPENDENCIES_COPY;
    CKGUID paramGuid = param->GetGUID();
    if (paramGuid == CKPGUID_COPYDEPENDENCIES) {
        opMode = CK_DEPENDENCIES_COPY;
    } else if (paramGuid == CKPGUID_DELETEDEPENDENCIES) {
        opMode = CK_DEPENDENCIES_DELETE;
    } else if (paramGuid == CKPGUID_REPLACEDEPENDENCIES) {
        opMode = CK_DEPENDENCIES_REPLACE;
    } else if (paramGuid == CKPGUID_SAVEDEPENDENCIES) {
        opMode = CK_DEPENDENCIES_SAVE;
    }

    CKDependencies *dependencies = new CKDependencies();

    CKCopyDefaultClassDependencies(*dependencies, opMode);

    param->SetValue(&dependencies, 0);

    return CK_OK;
}

void CKDependenciesDestructor(CKParameter *param) {
    CKDependencies *dependencies = nullptr;
    param->GetValue(&dependencies, FALSE);
    CKDependencies *nullDependencies = nullptr;
    param->SetValue(&nullDependencies, 0);
    if (dependencies) {
        delete dependencies;
    }
}

CK_DEPENDENCIES_OPMODE GetOperationModeFromGUID(CKParameter *param) {
    const CKGUID parameterGuid = param->GetGUID();
    if (parameterGuid == CKPGUID_COPYDEPENDENCIES) return CK_DEPENDENCIES_COPY;
    if (parameterGuid == CKPGUID_DELETEDEPENDENCIES) return CK_DEPENDENCIES_DELETE;
    if (parameterGuid == CKPGUID_REPLACEDEPENDENCIES) return CK_DEPENDENCIES_REPLACE;
    if (parameterGuid == CKPGUID_SAVEDEPENDENCIES) return CK_DEPENDENCIES_SAVE;

    return CK_DEPENDENCIES_COPY; // Default
}

void CKDependenciesSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load) {
    CKDependencies *dependencies = nullptr;
    param->GetValue(&dependencies, FALSE);
    if (!dependencies)
        return;

    if (!load) {
        // Save mode
        CKStateChunk *saveChunk = CreateCKStateChunk(nullptr);
        saveChunk->StartWrite();

        // Write appropriate identifier based on dependencies flags
        switch (dependencies->m_Flags) {
        case CK_DEPENDENCIES_NONE:
            saveChunk->WriteIdentifier(0x12248766);
            break;
        case CK_DEPENDENCIES_FULL:
            saveChunk->WriteIdentifier(0x12248767);
            break;
        case CK_DEPENDENCIES_CUSTOM: {
            saveChunk->WriteIdentifier(0x12248768);

            // Determine operation mode from parameter GUID
            const CK_DEPENDENCIES_OPMODE operationMode = GetOperationModeFromGUID(param);
            CKDependencies *defaultDependencies = CKGetDefaultClassDependencies(operationMode);

            // Write differences from default dependencies
            const int dependencyCount = dependencies->Size();
            for (int i = 1; i < dependencyCount; ++i) {
                if ((*dependencies)[i] != (*defaultDependencies)[i]) {
                    saveChunk->WriteInt(i);
                    saveChunk->WriteDword((*dependencies)[i]);
                }
            }
            break;
        }
        }

        saveChunk->CloseChunk();
        *chunk = saveChunk;
    } else {
        // Load mode
        CKStateChunk *loadChunk = *chunk;
        if (!loadChunk) return;

        // Read dependency type identifier
        if (loadChunk->SeekIdentifier(0x12248766)) {
            dependencies->m_Flags = CK_DEPENDENCIES_NONE;
        } else if (loadChunk->SeekIdentifier(0x12248767)) {
            dependencies->m_Flags = CK_DEPENDENCIES_FULL;
        } else {
            if (loadChunk->SeekIdentifier(0x12248768))
                dependencies->m_Flags = CK_DEPENDENCIES_CUSTOM;

            // Read custom dependencies data
            const int customDataSizeBytes = loadChunk->SeekIdentifierAndReturnSize(0x12248768);
            if (customDataSizeBytes > 0) {
                dependencies->m_Flags = CK_DEPENDENCIES_CUSTOM;
                const int entryCount = customDataSizeBytes / 8;  // Each entry is 2 DWORDs (8 bytes)
                for (int i = 0; i < entryCount; ++i) {
                    const int dependencyIndex = loadChunk->ReadInt();
                    (*dependencies)[dependencyIndex] = loadChunk->ReadInt();
                }
            }
        }
    }
}

void CKDependenciesCopier(CKParameter *dest, CKParameter *src) {
    CKDependencies *srcDeps = nullptr;
    CKDependencies *destDeps = nullptr;
    src->GetValue(&srcDeps, TRUE);
    dest->GetValue(&destDeps, TRUE);
    *destDeps = *srcDeps;
    destDeps->m_Flags = srcDeps->m_Flags;
}

CKERROR CKStateChunkCreator(CKParameter *param) {
    CKStateChunk *chunk = CreateCKStateChunk(nullptr);
    if (!chunk) {
        return CKERR_OUTOFMEMORY;
    }
    param->SetValue(&chunk, 0);
    return CK_OK;
}

void CKStateChunkDestructor(CKParameter *param) {
    CKStateChunk *chunk = nullptr;
    param->GetValue(&chunk, FALSE);
    CKStateChunk *null = nullptr;
    param->SetValue(&null, 0);
    DeleteCKStateChunk(chunk);
}

void CKStateChunkSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load) {
    CKStateChunk *stateChunk = nullptr;
    param->GetValue(&stateChunk, FALSE);
    if (load) {
        if (stateChunk) {
            stateChunk->Clone(*chunk);
        }
    } else {
        CKStateChunk *newChunk = new CKStateChunk(stateChunk);
        *chunk = newChunk;
    }
}

void CKStateChunkCopier(CKParameter *dest, CKParameter *src) {
    CKStateChunk *srcChunk = nullptr;
    CKStateChunk *destChunk = nullptr;
    src->GetValue(&srcChunk, TRUE);
    dest->GetValue(&destChunk, TRUE);
    destChunk->Clone(srcChunk);
}

CKERROR CKCollectionCreator(CKParameter *param) {
    XObjectArray *objectArray = new XObjectArray();
    if (!objectArray) {
        return CKERR_OUTOFMEMORY;
    }
    param->SetValue(&objectArray, 0);
    return CK_OK;
}

void CKCollectionDestructor(CKParameter *param) {
    XObjectArray *objectArray = nullptr;
    param->GetValue(&objectArray, FALSE);
    if (objectArray) {
        delete objectArray;
    }
    objectArray = nullptr;
    param->SetValue(&objectArray, 0);
}

void CKCollectionSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load) {
    XObjectArray *objectArray = nullptr;
    param->GetValue(&objectArray, FALSE);
    if (load) {
        CKStateChunk *loadChunk = *chunk;
        if (loadChunk && loadChunk->SeekIdentifier(0x50)) {
            const XObjectArray &loadArray = loadChunk->ReadXObjectArray();
            *objectArray = loadArray;
        }
    } else {
        CKStateChunk *saveChunk = nullptr;
        if (objectArray) {
            saveChunk = new CKStateChunk(CKCID_OBJECTARRAY, nullptr);
            saveChunk->SetDynamic(TRUE);
            saveChunk->StartWrite();
            saveChunk->WriteIdentifier(0x50);
            objectArray->Save(saveChunk, param->m_Context);
            saveChunk->CloseChunk();
        }

        *chunk = saveChunk;
    }
}

void CKCollectionCheck(CKParameter *param) {
    XObjectArray *objectArray = nullptr;
    param->GetValue(&objectArray, FALSE);
    int i = 0;
    CK_ID *begin = objectArray->Begin();
    if (objectArray->End() - objectArray->Begin() >= 0 && objectArray->Size() != 0) {
        do {
            if (!param->m_Context->GetObject(begin[i])) {
                objectArray->Begin()[i] = 0;
            }
            ++i;
            begin = objectArray->Begin();
        } while (i < objectArray->End() - objectArray->Begin());
    }
}

void CKCollectionCopy(CKParameter *dest, CKParameter *src) {
    XObjectArray *destArray = nullptr;
    dest->GetValue(&destArray, FALSE);
    XObjectArray *srcArray = nullptr;
    src->GetValue(&srcArray, FALSE);
    *destArray = *srcArray;
}

CKERROR CKMatrixCreator(CKParameter *param) {
    if (!param)
        return CKERR_INVALIDPARAMETER;

    VxMatrix matrix = VxMatrix::Identity();
    param->SetValue(&matrix, 0);
    return CK_OK;
}

CKERROR CKStructCreator(CKParameter *param) {
    if (!param)
        return CKERR_INVALIDPARAMETER;

    CKContext *context = param->m_Context;
    CKParameterManager *pm = context->GetParameterManager();

    CKParameterType type = param->GetType();
    CKStructStruct *desc = pm->GetStructDescByType(type);
    if (!desc)
        return CKERR_INVALIDPARAMETER;

    int memberCount = desc->NbData;
    CK_ID *memberIDs = new CK_ID[memberCount];

    for (int i = 0; i < memberCount; ++i) {
        CKParameterOut *member = context->CreateCKParameterOut(desc->Desc[i], desc->Guids[i], param->IsDynamic());

        if (member) {
            memberIDs[i] = member->GetID();
        } else {
            memberIDs[i] = 0;
        }
    }

    param->SetValue(memberIDs, sizeof(CK_ID) * memberCount);

    delete[] memberIDs;
    return CK_OK;
}

void CKStructDestructor(CKParameter *param) {
    if (!param)
        return;

    CKContext *context = param->m_Context;
    if (context->IsInClearAll())
        return;

    CKParameterManager *pm = context->GetParameterManager();
    CKStructStruct *desc = pm->GetStructDescByType(param->GetType());

    if (desc) {
        CK_ID *memberIDs = (CK_ID *) param->GetReadDataPtr();

        for (int i = 0; i < desc->NbData; ++i) {
            context->DestroyObject(memberIDs[i], CK_DESTROY_NONOTIFY);
            memberIDs[i] = 0;
        }
    }
}

void CKStructSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load) {
    if (!param || !chunk)
        return;

    CKContext *context = param->m_Context;
    CKParameterManager *pm = context->GetParameterManager();
    CKParameterType type = param->GetType();
    CKStructStruct *structDesc = pm->GetStructDescByType(type);

    if (!structDesc)
        return;

    CK_ID *memberIds = (CK_ID *) param->GetReadDataPtr();
    if (!memberIds)
        return;

    if (load) {
        // Loading from chunk
        CKStateChunk *loadChunk = *chunk;
        if (!loadChunk)
            return;

        if (loadChunk->SeekIdentifier(0x12348765)) {
            int count = loadChunk->ReadInt();
            if (count != structDesc->NbData) {
                context->OutputToConsole("Structure member count mismatch!");
                return;
            }

            for (int i = 0; i < structDesc->NbData; ++i) {
                CKStateChunk *subChunk = loadChunk->ReadSubChunk();
                if (subChunk) {
                    CKObject *member = context->GetObject(memberIds[i]);
                    if (member) {
                        member->Load(subChunk, nullptr);
                    }
                    delete subChunk;
                }
            }
        }
    } else {
        // Saving to chunk
        CKStateChunk *saveChunk = CreateCKStateChunk(nullptr);
        saveChunk->StartWrite();

        // Write structure header
        saveChunk->WriteIdentifier(0x12348765);
        saveChunk->WriteInt(structDesc->NbData);

        // Save each member
        for (int i = 0; i < structDesc->NbData; ++i) {
            CKObject *member = context->GetObject(memberIds[i]);
            if (member) {
                CKStateChunk *memberChunk = member->Save(nullptr, CK_STATESAVE_ALL);
                if (memberChunk) {
                    saveChunk->WriteSubChunk(memberChunk);
                    delete memberChunk;
                }
            }
        }

        saveChunk->CloseChunk();
        *chunk = saveChunk;
    }
}

void CKStructCopier(CKParameter *dest, CKParameter *src) {
    CKContext *context = dest->m_Context;
    CKParameterManager *pm = context->GetParameterManager();

    CKParameterType type = dest->GetType();
    CKStructStruct *desc = pm->GetStructDescByType(type);
    if (!desc)
        return;

    CK_ID *srcMembers = (CK_ID *) src->GetReadDataPtr();
    CK_ID *destMembers = (CK_ID *) dest->GetWriteDataPtr();

    for (int i = 0; i < desc->NbData; ++i) {
        CKParameterOut *srcParam = (CKParameterOut *) context->GetObject(srcMembers[i]);
        CKParameterOut *destParam = (CKParameterOut *) context->GetObject(destMembers[i]);
        if (destParam) {
            destParam->CopyValue(srcParam, FALSE);
        }
    }
}

CKERROR CK2dCurveCreator(CKParameter *param) {
    CK2dCurve *curve = new CK2dCurve();
    if (!curve) {
        return CKERR_OUTOFMEMORY;
    }
    param->SetValue(&curve, 0);
    return CK_OK;
}

void CK2dCurveDestructor(CKParameter *param) {
    if (!param)
        return;
    CK2dCurve *curve = nullptr;
    param->GetValue(&curve, FALSE);
    if (curve) {
        delete curve;
    }
    curve = nullptr;
    param->SetValue(&curve, 0);
}

void CK2dCurveCopier(CKParameter *dest, CKParameter *src) {
    CK2dCurve *srcCurve = nullptr;
    src->GetValue(&srcCurve, FALSE);
    CK2dCurve *destCurve = nullptr;
    dest->GetValue(&destCurve, TRUE);
    *destCurve = *srcCurve;
}

void CK2dCurveSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load) {
    if (!param)
        return;

    CK2dCurve *curve = nullptr;
    param->GetValue(&curve, FALSE);
    if (load) {
        CKStateChunk *loadChunk = *chunk;
        if (loadChunk && loadChunk->SeekIdentifier(CKCHNK_2DCURVE)) {
            CKStateChunk *curveChunk = loadChunk->ReadSubChunk();
            if (curve)
                curve->Read(curveChunk);
            if (curveChunk) {
                DeleteCKStateChunk(curveChunk);
            }
        }
    } else {
        CKStateChunk *saveChunk = CreateCKStateChunk(CKCHNK_2DCURVE, nullptr);
        CKStateChunk *curveChunk = nullptr;
        if (curve)
            curveChunk = curve->Dump();
        saveChunk->StartWrite();
        saveChunk->WriteIdentifier(CKCHNK_2DCURVE);
        saveChunk->WriteSubChunk(curveChunk);
        saveChunk->CloseChunk();
        if (curveChunk) {
            DeleteCKStateChunk(curveChunk);
        }
        *chunk = saveChunk;
    }
}

void CKPointerSaver(CKParameter *param, CKStateChunk **chunk, CKBOOL load) {
    if (!load && chunk && *chunk)
        *chunk = nullptr;
}

void CKObjectCheckFunc(CKParameter *param) {
    CK_ID *pid = (CK_ID *) param->GetWriteDataPtr();
    if (pid) {
        CKContext *context = param->m_Context;
        CKObject *obj = context->GetObject(*pid);
        if (!obj) {
            *pid = 0;
        }
    }
}

int CKMatrixStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    VxMatrix matrix;

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "[%f,%f,%f,%f][%f,%f,%f,%f][%f,%f,%f,%f][%f,%f,%f,%f]",
                   &matrix[0][0], &matrix[0][1], &matrix[0][2], &matrix[0][3],
                   &matrix[1][0], &matrix[1][1], &matrix[1][2], &matrix[1][3],
                   &matrix[2][0], &matrix[2][1], &matrix[2][2], &matrix[2][3],
                   &matrix[3][0], &matrix[3][1], &matrix[3][2], &matrix[3][3]);
            param->SetValue(&matrix, 0);
        }
        return 0;
    }

    param->GetValue(&matrix, FALSE);

    // Apply epsilon check for each element (CK2.dll behavior)
    float m00 = fabs(matrix[0][0]) >= EPSILON ? matrix[0][0] : 0.0f;
    float m01 = fabs(matrix[0][1]) >= EPSILON ? matrix[0][1] : 0.0f;
    float m02 = fabs(matrix[0][2]) >= EPSILON ? matrix[0][2] : 0.0f;
    float m03 = fabs(matrix[0][3]) >= EPSILON ? matrix[0][3] : 0.0f;
    float m10 = fabs(matrix[1][0]) >= EPSILON ? matrix[1][0] : 0.0f;
    float m11 = fabs(matrix[1][1]) >= EPSILON ? matrix[1][1] : 0.0f;
    float m12 = fabs(matrix[1][2]) >= EPSILON ? matrix[1][2] : 0.0f;
    float m13 = fabs(matrix[1][3]) >= EPSILON ? matrix[1][3] : 0.0f;
    float m20 = fabs(matrix[2][0]) >= EPSILON ? matrix[2][0] : 0.0f;
    float m21 = fabs(matrix[2][1]) >= EPSILON ? matrix[2][1] : 0.0f;
    float m22 = fabs(matrix[2][2]) >= EPSILON ? matrix[2][2] : 0.0f;
    float m23 = fabs(matrix[2][3]) >= EPSILON ? matrix[2][3] : 0.0f;
    float m30 = fabs(matrix[3][0]) >= EPSILON ? matrix[3][0] : 0.0f;
    float m31 = fabs(matrix[3][1]) >= EPSILON ? matrix[3][1] : 0.0f;
    float m32 = fabs(matrix[3][2]) >= EPSILON ? matrix[3][2] : 0.0f;
    float m33 = fabs(matrix[3][3]) >= EPSILON ? matrix[3][3] : 0.0f;
    
    char source[256];
    sprintf(source, "[%g,%g,%g,%g][%g,%g,%g,%g][%g,%g,%g,%g][%g,%g,%g,%g]",
            m00, m01, m02, m03,
            m10, m11, m12, m13,
            m20, m21, m22, m23,
            m30, m31, m32, m33);

    if (ValueString)
        strcpy(ValueString, source);

    return strlen(source) + 1;
}

int CKQuaternionStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    // CK2.dll initialization order: z=0, y=0, x=0, w=1
    VxQuaternion quat;
    quat.z = 0.0f;
    quat.y = 0.0f;
    quat.x = 0.0f;
    quat.w = 1.0f;

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "%f,%f,%f,%f", &quat.x, &quat.y, &quat.z, &quat.w);
            param->SetValue(&quat, 0);
        }
        return 0;
    }

    param->GetValue(&quat, FALSE);
    
    char source[256];
    sprintf(source, "%g,%g,%g,%g", quat.x, quat.y, quat.z, quat.w);

    if (ValueString)
        strcpy(ValueString, source);

    return strlen(source) + 1;
}

int CKObjectStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    CKContext *context = param->m_Context;

    if (ReadFromString) {
        if (ValueString) {
            const CKParameterType parameterType = param->GetType();
            CKParameterManager *parameterManager = context->GetParameterManager();
            const CKDWORD classId = parameterManager->TypeToClassID(parameterType);
            
            CKObject *object = context->GetObjectByNameAndParentClass(ValueString, classId, NULL);
            const CK_ID objectId = object ? object->GetID() : 0;
            param->SetValue(&objectId, 0);
        }
        return 0;
    }

    CK_ID objID = 0;
    param->GetValue(&objID, FALSE);
    
    CKObject *object = context->GetObject(objID);
    if (!object || !object->GetName()) {
        if (ValueString)
            *ValueString = '\0';
        return 0;
    }

    if (ValueString)
        strcpy(ValueString, object->GetName());
    
    return strlen(object->GetName()) + 1;
}

int CKStringStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;
    if (ReadFromString) {
        if (ValueString) {
            const size_t valueLength = strlen(ValueString);
            param->SetValue(ValueString, valueLength + 1);
        }
        return 0;
    }
    if (ValueString)
        param->GetValue(ValueString, FALSE);
    return param->GetDataSize();
}

int CKAttributeStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    CKContext *m_Context = param->m_Context;
    
    if (ReadFromString) {
        if (ValueString) {
            CKAttributeManager *AttributeManager = m_Context->GetAttributeManager();
            CKAttributeType attrType = AttributeManager->GetAttributeTypeByName(ValueString);
            param->SetValue(&attrType, 0);
        }
        return 0;
    }
    
    CKAttributeType attrType = -1;
    param->GetValue(&attrType, FALSE);
    
    CKAttributeManager *AttributeManager = m_Context->GetAttributeManager();
    const char *AttributeNameByType = AttributeManager->GetAttributeNameByType(attrType);
    if (!AttributeNameByType)
        return 0;
    
    if (ValueString)
        strcpy(ValueString, AttributeNameByType);
    
    return strlen(AttributeNameByType) + 1;
}

int CKMessageStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    CKContext *m_Context = param->m_Context;
    
    if (ReadFromString) {
        if (ValueString) {
            CKMessageManager *MessageManager = m_Context->GetMessageManager();
            CKMessageType msgType = MessageManager->AddMessageType(ValueString);
            param->SetValue(&msgType, 0);
        }
        return 0;
    }
    
    CKMessageType msgType = -1;
    param->GetValue(&msgType, FALSE);
    
    CKMessageManager *MessageManager = m_Context->GetMessageManager();
    const char *MessageTypeName = MessageManager->GetMessageTypeName(msgType);
    if (!MessageTypeName)
        return 0;
    
    if (ValueString)
        strcpy(ValueString, MessageTypeName);
    
    return strlen(MessageTypeName) + 1;
}

int CKCollectionStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    CKContext *context = param->m_Context;
    
    if (ReadFromString)
        return 0;
    
    XObjectArray *objectArray = NULL;
    param->GetValue(&objectArray, FALSE);
    
    if (!objectArray)
        return 0;
    
    int formattedLength;
    if (ValueString) {
        formattedLength = sprintf(ValueString, "%d Elements", objectArray->Size());
    } else {
        char *stringBuffer = context->GetStringBuffer(40);
        formattedLength = sprintf(stringBuffer, "%d Elements", objectArray->Size());
    }
    
    return formattedLength + 1;
}

int CKStructStringFunc(CKParameter *param, char *valueStr, CKBOOL readFromStr) {
    const char DELIMITER = ';';
    const int MAX_MEMBER_LENGTH = 256;
    const int MAX_STRUCT_LENGTH = 2048;

    if (!param)
        return 0;

    CKContext *context = param->m_Context;
    CKParameterManager *pm = context->GetParameterManager();
    CKParameterType type = param->GetType();

    CKStructStruct *desc = pm->GetStructDescByType(type);
    if (!desc)
        return -1;

    CK_ID *memberIDs = (CK_ID *) param->GetReadDataPtr();
    if (!memberIDs)
        return -1;

    if (readFromStr) {
        if (!valueStr)
            return 0;

        const char *current = valueStr;
        for (int i = 0; i < desc->NbData; ++i) {
            CKParameterOut *member = (CKParameterOut *) context->GetObject(memberIDs[i]);
            if (!member)
                continue;

            // Find next delimiter
            size_t len = strcspn(current, &DELIMITER);
            if (len > 0) {
                char buffer[MAX_MEMBER_LENGTH] = {0};
                strncpy(buffer, current, XMin(len, sizeof(buffer) - 1));
                member->SetStringValue(buffer);
            }

            // Move to next member data
            current += len;
            if (*current == DELIMITER) current++;
        }
        return 0;
    } else {
        // Convert structure members to string
        char output[MAX_STRUCT_LENGTH] = {0};
        int totalLength = 0;

        for (int i = 0; i < desc->NbData; ++i) {
            CKParameterOut *member = (CKParameterOut *) context->GetObject(memberIDs[i]);

            char memberStr[MAX_MEMBER_LENGTH] = {0};
            if (member && member->GetStringValue(memberStr, sizeof(memberStr))) {
                // Check buffer space: existing length + member + delimiter + null
                if (totalLength + strlen(memberStr) + 1 >= sizeof(output))
                    break;

                strcat(output, memberStr);
                totalLength += strlen(memberStr);
            } else {
                strcat(output, "NULL");
                totalLength += 4;
            }

            // Add delimiter (will remove last one later)
            strcat(output, &DELIMITER);
            totalLength++;
        }

        // Save original length before removing trailing delimiter
        int originalLength = totalLength;
        
        // Remove trailing delimiter if any data was added
        if (totalLength > 0) {
            output[totalLength - 1] = '\0';
        }

        // Copy to output buffer if provided
        if (valueStr) {
            strcpy(valueStr, output);
        }

        // Return original length + 1 (matches CK2.dll behavior)
        return originalLength + 1;
    }
}

int CKEnumStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    CKContext *context = param->m_Context;
    CKParameterManager *paramManager = context->GetParameterManager();

    CKParameterType paramType = param->GetType();
    CKEnumStruct *enumDesc = paramManager->GetEnumDescByType(paramType);

    // Fallback to integer conversion if not an enum type
    if (!enumDesc) {
        return CKIntStringFunc(param, ValueString, ReadFromString);
    }

    int enumValue = 0;
    
    if (ReadFromString) {
        // Parse string to enum value
        if (ValueString) {
            int i = 0;
            if (enumDesc->NbData > 0) {
                while (!enumDesc->Desc[i] || strcmp(ValueString, enumDesc->Desc[i]) != 0) {
                    if (++i >= enumDesc->NbData)
                        goto CHECK_NOT_FOUND;
                }
                enumValue = enumDesc->Vals[i];
            CHECK_NOT_FOUND:
                if (i == enumDesc->NbData) {
                    return CKIntStringFunc(param, ValueString, ReadFromString);
                }
            }
            param->SetValue(&enumValue, 0);
        }
        return 0;
    } else {
        // Convert enum value to string
        char enumText[256];
        strcpy(enumText, "Invalid Enum Value");
        memset(&enumText[19], 0, 237);
        
        int i = 0;
        param->GetValue(&enumValue, FALSE);
        
        if (enumDesc->NbData > 0) {
            while (enumDesc->Vals[i] != enumValue) {
                ++i;
                if (i >= enumDesc->NbData)
                    goto CHECK_FALLBACK;
            }
            if (enumDesc->Desc[i])
                strcpy(enumText, enumDesc->Desc[i]);
        }
        
    CHECK_FALLBACK:
        if (i == enumDesc->NbData)
            return CKIntStringFunc(param, ValueString, FALSE);
        
        if (ValueString)
            strcpy(ValueString, enumText);
        
        return strlen(enumText) + 1;
    }
}

int CKFlagsStringFunc(CKParameter *param, CKSTRING ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    CKContext *context = param->m_Context;
    CKParameterType paramType = param->GetType();
    CKParameterManager *paramManager = context->GetParameterManager();
    CKFlagsStruct *flagsDesc = paramManager->GetFlagsDescByType(paramType);

    // Fallback to integer conversion if not a flags type
    if (!flagsDesc) {
        return CKIntStringFunc(param, ValueString, ReadFromString);
    }

    int flagsValue = 0;
    
    if (ReadFromString) {
        while (ValueString) {
            if (!*ValueString)
                break;
            
            char *commaPos = strchr(ValueString, ',');
            int tokenLen = commaPos ? (int)(commaPos - ValueString) : (int)strlen(ValueString);
            
            if (tokenLen <= 0) {
                ValueString = nullptr;
                continue;
            }
            
            char token[256];
            strncpy(token, ValueString, tokenLen);
            token[tokenLen] = '\0';
            
            int i = 0;
            if (flagsDesc->NbData > 0) {
                while (!flagsDesc->Desc[i] || _stricmp(token, flagsDesc->Desc[i]) != 0) {
                    if (++i >= flagsDesc->NbData)
                        goto NEXT_TOKEN;
                }
                flagsValue |= flagsDesc->Vals[i];
            }
            
        NEXT_TOKEN:
            if (commaPos)
                ValueString += tokenLen + 1;
            else
                ValueString = nullptr;
        }

        param->SetValue(&flagsValue, 0);
        return 0;
    } else {
        char text[1024];
        text[0] = '\0';
        memset(&text[1], 0, 1020);
        
        int i = 0;
        param->GetValue(&flagsValue, FALSE);
        
        if (flagsDesc->NbData > 0) {
            do {
                if ((flagsValue & flagsDesc->Vals[i]) != 0 && flagsDesc->Desc[i]) {
                    strcat(text, flagsDesc->Desc[i]);
                    strcat(text, ",");
                }
                ++i;
            } while (i < flagsDesc->NbData);
        }

        // Save original length before removing trailing comma
        int originalLen = (int)strlen(text);
        int resultLen = originalLen;
        
        if (originalLen > 0) {
            text[originalLen - 1] = '\0';
        }
        
        if (ValueString) {
            strcpy(ValueString, text);
        }

        return resultLen + 1;
    }
}

int CKIntStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param)
        return 0;

    int value = 0;
    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "%d", &value);
            param->SetValue(&value, 0);
        }
        return 0;
    } else {
        param->GetValue(&value, FALSE);
        
        char Source[64];
        sprintf(Source, "%d", value);
        
        if (ValueString) {
            strcpy(ValueString, Source);
        }

        return strlen(Source) + 1;
    }
}

int CKAngleStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float angleRadians = 0.0f;
    
    if (ReadFromString) {
        int cycles = 0;
        float degreesRemainder = 0.0f;
        
        if (ValueString) {
            sscanf(ValueString, "%d:%f", &cycles, &degreesRemainder);
            angleRadians = ((float)cycles + degreesRemainder) * 0.017453292f;  // deg2rad constant from CK2.dll
            param->SetValue(&angleRadians, 0);
        }
        return 0;
    } else {
        param->GetValue(&angleRadians, FALSE);
        
        double degrees = angleRadians * 57.295776;  // rad2deg constant from CK2.dll
        int cycles = (int)(0.0027777778 * degrees);  // 1/360 constant from CK2.dll
        
        char source[64];
        sprintf(source, "%d:%g", cycles, degrees - (double)cycles * 360.0);
        
        if (ValueString) {
            strcpy(ValueString, source);
        }

        return strlen(source) + 1;
    }
}

int CKEulerStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float anglesRadians[3] = {0.0f, 0.0f, 0.0f};  // radians array
    
    if (ReadFromString) {
        int cycleX = 0, cycleY = 0, cycleZ = 0;  // cycles
        float remainderX = 0.0f, remainderY = 0.0f, remainderZ = 0.0f;  // remainders
        
        if (ValueString) {
            sscanf(ValueString, "%d:%f,%d:%f,%d:%f", &cycleX, &remainderX, &cycleY, &remainderY, &cycleZ, &remainderZ);
            
            // Convert to radians (using CK2.dll constants)
            anglesRadians[0] = ((float)cycleX + remainderX) * 0.017453292f;
            anglesRadians[1] = ((float)cycleY + remainderY) * 0.017453292f;
            anglesRadians[2] = ((float)cycleZ + remainderZ) * 0.017453292f;
            
            param->SetValue(anglesRadians, 0);
        }
        return 0;
    } else {
        param->GetValue(anglesRadians, FALSE);
        
        int cycleX = 0, cycleY = 0, cycleZ = 0;
        float remainderX = 0.0f, remainderY = 0.0f, remainderZ = 0.0f;
        
        for (int i = 0; i < 3; ++i) {
            double degrees = anglesRadians[i] * 57.295776;  // rad2deg
            int cycles = (int)(0.0027777778 * degrees);  // 1/360
            
            if (i == 0) { cycleX = cycles; remainderX = (float)(degrees - (double)cycles * 360.0); }
            else if (i == 1) { cycleY = cycles; remainderY = (float)(degrees - (double)cycles * 360.0); }
            else { cycleZ = cycles; remainderZ = (float)(degrees - (double)cycles * 360.0); }
        }
        
        char source[64];
        sprintf(source, "%d:%g,%d:%g,%d:%g", cycleX, remainderX, cycleY, remainderY, cycleZ, remainderZ);
        
        if (ValueString)
            strcpy(ValueString, source);
        
        return strlen(source) + 1;
    }
}

int CK2DVectorStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float x = 0.0f;
    float y = 0.0f;

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "%f,%f", &x, &y);
            param->SetValue(&x, 0);
        }
        return 0;
    } else {
        param->GetValue(&x, FALSE);
        
        char source[64];
        sprintf(source, "%g,%g", x, y);
        
        if (ValueString)
            strcpy(ValueString, source);

        return strlen(source) + 1;
    }
}

int CKVectorStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float x = 0.0f, y = 0.0f, z = 0.0f;

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "%f,%f,%f", &x, &y, &z);
            param->SetValue(&x, 0);
        }
        return 0;
    } else {
        param->GetValue(&x, FALSE);
        
        char source[64];
        sprintf(source, "%g,%g,%g", x, y, z);
        
        if (ValueString)
            strcpy(ValueString, source);

        return strlen(source) + 1;
    }
}

int CKRectStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    VxRect rect = {0.0f, 0.0f, 0.0f, 0.0f};

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "(%f,%f),(%f,%f)", &rect.left, &rect.top, &rect.right, &rect.bottom);
            param->SetValue(&rect, 0);
        }
        return 0;
    } else {
        param->GetValue(&rect, FALSE);
        
        char source[64];
        sprintf(source, "(%g,%g),(%g,%g)", rect.left, rect.top, rect.right, rect.bottom);
        
        if (ValueString)
            strcpy(ValueString, source);

        return strlen(source) + 1;
    }
}

int CKBoxStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    VxBbox bbox;
    // CK2.dll calls VxBbox constructor

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "(%f,%f,%f),(%f,%f,%f)", 
                   &bbox.Min.x, &bbox.Min.y, &bbox.Min.z, 
                   &bbox.Max.x, &bbox.Max.y, &bbox.Max.z);
            param->SetValue(&bbox, 0);
        }
        return 0;
    } else {
        param->GetValue(&bbox, FALSE);
        
        char source[64];
        sprintf(source, "(%g,%g,%g),(%g,%g,%g)", 
                bbox.Min.x, bbox.Min.y, bbox.Min.z, 
                bbox.Max.x, bbox.Max.y, bbox.Max.z);
        
        if (ValueString)
            strcpy(ValueString, source);

        return strlen(source) + 1;
    }
}

int CKColorStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;

    if (ReadFromString) {
        if (ValueString) {
            int r255, g255, b255, a255;
            sscanf(ValueString, "%d,%d,%d,%d", &r255, &g255, &b255, &a255);
            
            // Use CK2.dll constant 0.0039215689 = 1/255
            r = (float)r255 * 0.0039215689f;
            g = (float)g255 * 0.0039215689f;
            b = (float)b255 * 0.0039215689f;
            a = (float)a255 * 0.0039215689f;
            
            param->SetValue(&r, 0);
        }
        return 0;
    } else {
        param->GetValue(&r, FALSE);
        
        char source[64];
        sprintf(source, "%d,%d,%d,%d",
                (unsigned int)(int)(r * 255.0f),
                (unsigned int)(int)(g * 255.0f),
                (unsigned int)(int)(b * 255.0f),
                (unsigned int)(int)(a * 255.0f));
        
        if (ValueString)
            strcpy(ValueString, source);
        
        return strlen(source) + 1;
    }
}

int CKBoolStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    CKBOOL value = FALSE;

    if (ReadFromString) {
        if (ValueString) {
            char token[8];
            strncpy(token, ValueString, 5);
            token[5] = '\0';
            value = (_stricmp(token, "TRUE") == 0) ? TRUE : FALSE;
        }
        param->SetValue(&value, 0);
        return 0;
    } else {
        param->GetValue(&value, FALSE);
        
        char text[64];
        if (value)
            strcpy(text, "TRUE");
        else
            strcpy(text, "FALSE");
        
        if (ValueString)
            strcpy(ValueString, text);
        
        return strlen(text) + 1;
    }
}

int CKFloatStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float value = 0.0f;

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "%f", &value);
            param->SetValue(&value, 0);
        }
        return 0;
    } else {
        param->GetValue(&value, FALSE);
        
        char source[64];
        sprintf(source, "%g", value);
        
        if (ValueString)
            strcpy(ValueString, source);
        
        return strlen(source) + 1;
    }
}

int CKPercentageStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float value = 0.0f;

    if (ReadFromString) {
        if (ValueString) {
            sscanf(ValueString, "%f%%", &value);
            float normalized = value * 0.01f;  // CK2.dll uses 0.0099999998
            param->SetValue(&normalized, 0);
        }
        return 0;
    } else {
        param->GetValue(&value, FALSE);
        double percentage = value * 100.0;
        
        char source[64];
        sprintf(source, "%g%%", percentage);
        
        if (ValueString)
            strcpy(ValueString, source);
        
        return strlen(source) + 1;
    }
}

int CKTimeStringFunc(CKParameter *param, char *ValueString, CKBOOL ReadFromString) {
    if (!param) return 0;

    float timeMs = 0.0f;

    if (ReadFromString) {
        if (ValueString) {
            float minutes = 0.0f, seconds = 0.0f, milliseconds = 0.0f;
            sscanf(ValueString, "%fm %fs %fms", &minutes, &seconds, &milliseconds);
            timeMs = (minutes * 60.0f + seconds) * 1000.0f + milliseconds;
            param->SetValue(&timeMs, 0);
        }
        return 0;
    } else {
        param->GetValue(&timeMs, FALSE);
        
        div_t minuteSplit = div((int)timeMs, 60000);
        int minutes = minuteSplit.quot;
        div_t secondSplit = div(minuteSplit.rem, 1000);
        
        char source[64];
        sprintf(source, "%.2dm %.2ds %.3dms", minutes, secondSplit.quot, secondSplit.rem);
        
        if (ValueString)
            strcpy(ValueString, source);
        
        return strlen(source) + 1;
    }
}

CK_PARAMETERUICREATORFUNCTION GetUICreatorFunction(CKContext *context, CKParameterTypeDesc *desc) {
    CKInterfaceManager *interfaceManager = (CKInterfaceManager *) context->GetManagerByGuid(INTERFACE_MANAGER_GUID);
    if (interfaceManager) {
        return interfaceManager->GetEditorFunctionForParameterType(desc);
    }
    return nullptr;
}
