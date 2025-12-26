#include "CKGlobals.h"

#include <time.h>

#include <miniz.h>

#include "VxMath.h"
#include "CKContext.h"
#include "CKPluginManager.h"
#include "CKObject.h"
#include "CKParameterIn.h"
#include "CKParameter.h"
#include "CKParameterOut.h"
#include "CKParameterLocal.h"
#include "CKParameterOperation.h"
#include "CKBehaviorLink.h"
#include "CKBehaviorIO.h"
#include "CKSceneObject.h"
#include "CKSynchroObject.h"
#include "CKInterfaceObjectManager.h"
#include "CKBeObject.h"
#include "CKBehavior.h"
#include "CKScene.h"
#include "CKLevel.h"
#include "CKGroup.h"
#include "CKSound.h"
#include "CKDataArray.h"
#include "CKWaveSound.h"
#include "CKMidiSound.h"
#include "CKObjectDeclaration.h"
#include "CKBehaviorPrototype.h"
#include "CKStateChunk.h"

extern INSTANCE_HANDLE g_CKModule;

XString g_PluginPath;
XString g_StartPath;
XArray<CKContext *> g_Contextes;
XClassInfoArray g_CKClassInfo;
CKDependencies g_DefaultCopyDependencies;
CKDependencies g_DefaultReplaceDependencies;
CKDependencies g_DefaultDeleteDependencies;
CKDependencies g_DefaultSaveDependencies;
CKStats g_MainStats;
int g_MaxClassID = CKCID_MAXCLASSID;
ProcessorsType g_TheProcessor;
CKPluginManager g_ThePluginManager;
VxImageDescEx Default32Desc;
CKBOOL WarningForOlderVersion = FALSE;

typedef XHashTable<CKObjectDeclaration *, CKGUID> XObjDeclHashTable;
XObjDeclHashTable g_PrototypeDeclarationList;

void CKInitStartPath() {
    char buffer[260];
    if (VxGetModuleFileName(g_CKModule, buffer, 260)) {
        CKPathSplitter ps(buffer);
        g_StartPath = ps.GetDrive();
        g_StartPath << ps.GetDir();
        VxAddLibrarySearchPath(g_StartPath.Str());

        char pluginsPath[260];
        VxMakePath(pluginsPath, g_StartPath.Str(), "Plugins");
        g_PluginPath = pluginsPath;
    }
}

CKERROR CKStartUp() {
    CKInitStartPath();

    srand((unsigned int)time(nullptr));

    Default32Desc.BitsPerPixel = 32;
    Default32Desc.RedMask = R_MASK;
    Default32Desc.GreenMask = G_MASK;
    Default32Desc.BlueMask = B_MASK;
    Default32Desc.AlphaMask = A_MASK;

    g_CKClassInfo.Reserve(CKCID_MAXCLASSID + 1);

    g_TheProcessor = GetProcessorType();

    g_ThePluginManager.AddCategory(CKPLUGIN_BITMAP);
    g_ThePluginManager.AddCategory(CKPLUGIN_SOUND);
    g_ThePluginManager.AddCategory(CKPLUGIN_MODEL);
    g_ThePluginManager.AddCategory(CKPLUGIN_MANAGER);
    g_ThePluginManager.AddCategory(CKPLUGIN_BEHAVIOR);
    g_ThePluginManager.AddCategory(CKPLUGIN_RENDERENGINE);
    g_ThePluginManager.AddCategory(CKPLUGIN_MOVIE);
    g_ThePluginManager.AddCategory(CKPLUGIN_EXTENSIONS);

    CKCLASSREGISTER(CKObject, CKObject);
    CKCLASSREGISTER(CKParameterIn, CKObject);
    CKCLASSREGISTER(CKParameter, CKObject);
    CKCLASSREGISTER(CKParameterOut, CKParameter);
    CKCLASSREGISTER(CKParameterLocal, CKParameter);
    CKCLASSREGISTER(CKParameterOperation, CKObject);
    CKCLASSREGISTER(CKBehaviorLink, CKObject);
    CKCLASSREGISTER(CKBehaviorIO, CKObject);
    CKCLASSREGISTER(CKSceneObject, CKObject);
    CKCLASSREGISTER(CKSynchroObject, CKObject);
    CKCLASSREGISTER(CKStateObject, CKObject);
    CKCLASSREGISTER(CKCriticalSectionObject, CKObject);
    CKCLASSREGISTER(CKInterfaceObjectManager, CKObject);
    CKCLASSREGISTER(CKBeObject, CKSceneObject);
    CKCLASSREGISTER(CKBehavior, CKSceneObject);
    CKCLASSREGISTER(CKScene, CKBeObject);
    CKCLASSREGISTER(CKLevel, CKBeObject);
    CKCLASSREGISTER(CKGroup, CKBeObject);
    CKCLASSREGISTER(CKSound, CKBeObject);
    CKCLASSREGISTER(CKDataArray, CKBeObject);
    CKCLASSREGISTER(CKWaveSound, CKSound);
    CKCLASSREGISTER(CKMidiSound, CKSound);

    CKBuildClassHierarchyTable();

    return CK_OK;
}

CKERROR CKShutdown() {
    for (XObjDeclHashTableIt it = g_PrototypeDeclarationList.Begin(); it != g_PrototypeDeclarationList.End(); ++it) {
        CKObjectDeclaration *decl = *it;
        if (decl) {
            if (decl->m_Proto)
                delete decl->m_Proto;
            delete decl;
        }
    }

    g_ThePluginManager.ReleaseAllPlugins();
    g_StartPath = "";
    g_PluginPath = "";
    g_CKClassInfo.Clear();
    g_Contextes.Clear();
    g_PrototypeDeclarationList.Clear();

    return CK_OK;
}

CKContext *GetCKContext(int pos) {
    if (pos < 0 || pos >= g_Contextes.Size())
        return nullptr;
    return g_Contextes[pos];
}

CKObject *CKGetObject(CKContext *iCtx, CK_ID iID) {
    return iCtx ? iCtx->GetObject(iID) : nullptr;
}

CKERROR CKCreateContext(CKContext **iContext, WIN_HANDLE iWin, int iRenderEngine, CKDWORD Flags) {
    if (iRenderEngine < 0 || iRenderEngine >= g_ThePluginManager.GetPluginCount(CKPLUGIN_RENDERENGINE_DLL))
        iRenderEngine = 0;

    CKContext *ctx = new CKContext(iWin, iRenderEngine, Flags);
    if (!ctx) return CKERR_OUTOFMEMORY;

    g_Contextes.PushBack(ctx);
    g_ThePluginManager.InitializePlugins(ctx);
    memset(&g_MainStats, 0, sizeof(g_MainStats));
    ctx->m_InitManagerOnRegister = TRUE;
    ctx->ExecuteManagersOnCKInit();
    *iContext = ctx;
    return CK_OK;
}

CKERROR CKCloseContext(CKContext *ctx) {
    if (!ctx)
        return CKERR_INVALIDPARAMETER;
    g_Contextes.Remove(ctx);
    CKERROR err = ctx->ClearAll();
    delete ctx;
    return err;
}

CKSTRING CKGetStartPath() {
    return g_StartPath.Str();
}

CKSTRING CKGetPluginsPath() {
    return g_PluginPath.Str();
}

void CKDestroyObject(CKObject *o, CKDWORD Flags, CKDependencies *dep) {
    if (o && o->m_Context)
        o->m_Context->DestroyObject(o, Flags, dep);
}

CKDWORD CKGetVersion() {
    return CKVERSION;
}

void ComputeParentsTable(CK_CLASSID cid) {
    if (cid <= 0 || cid >= g_CKClassInfo.Size()) return;

    CKClassDesc& info = g_CKClassInfo[cid];
    if (info.Done) return;

    info.Done = TRUE; // Mark as processed to prevent infinite recursion

    // Root class
    if (info.Parent == 0) {
        info.Parents.Set(cid);
        info.DerivationLevel = 0;
        return;
    }

    ComputeParentsTable(info.Parent);

    CKClassDesc &parentInfo = g_CKClassInfo[info.Parent];
    info.Parents = parentInfo.Parents;
    info.Parents.Set(cid);
    info.DerivationLevel = parentInfo.DerivationLevel + 1;
}

void ComputeParentsNotifyTable(CK_CLASSID cid) {
    if (cid <= 0 || cid >= g_CKClassInfo.Size()) return;

    CKClassDesc& info = g_CKClassInfo[cid];
    if (info.Done) return;

    info.Done = TRUE; // Mark as processed

    if (info.Parent != 0) {
        ComputeParentsNotifyTable(info.Parent);
        CKClassDesc &parentInfo = g_CKClassInfo[info.Parent];
        info.CommonToBeNotify.Or(parentInfo.CommonToBeNotify);
    }
    info.CommonToBeNotify.Or(info.ToBeNotify);
}

void CKBuildClassHierarchyTable() {
    const int classCount = g_CKClassInfo.Size();

    // 1. Initialize class information structures
    for (CK_CLASSID i = 0; i < classCount; ++i) {
        CKClassDesc &info = g_CKClassInfo[i];
        info.Done = FALSE;
        info.CommonToBeNotify.Clear();
        info.ToBeNotify.Clear();
        info.Parameter = CKGUID();
        info.Parents.Clear();
        info.Children.Clear();
    }

    // 2. Compute parent-child relationships
    for (CK_CLASSID i = 0; i < classCount; ++i) {
        ComputeParentsTable(i);
    }

    // 3. Build inverse child relationships
    for (CK_CLASSID cls = 0; cls < classCount; ++cls) {
        g_CKClassInfo[cls].Children.Set(cls); // A class is considered its own "child" for type matching

        for (CK_CLASSID parent = 0; parent < classCount; ++parent) {
            if (g_CKClassInfo[cls].Parents.IsSet(parent)) {
                g_CKClassInfo[parent].Children.Set(cls);
            }
        }
    }

    // 4. Call registration functions
    for (CK_CLASSID i = 0; i < classCount; ++i) {
        if (g_CKClassInfo[i].RegisterFct) {
            g_CKClassInfo[i].RegisterFct();
        }
    }

    // 5. Initialize global dependencies
    g_DefaultCopyDependencies.Resize(classCount);
    g_DefaultCopyDependencies.m_Flags = CK_DEPENDENCIES_CUSTOM;

    g_DefaultDeleteDependencies.Resize(classCount);
    g_DefaultDeleteDependencies.m_Flags = CK_DEPENDENCIES_NONE;

    g_DefaultReplaceDependencies.Resize(classCount);
    g_DefaultSaveDependencies.Resize(classCount);

    // 6. Copy dependency information
    for (CK_CLASSID i = 0; i < classCount; ++i) {
        g_DefaultCopyDependencies[i] = g_CKClassInfo[i].DefaultCopyDependencies;
        g_DefaultDeleteDependencies[i] = g_CKClassInfo[i].DefaultDeleteDependencies;
        g_DefaultReplaceDependencies[i] = g_CKClassInfo[i].DefaultReplaceDependencies;
        g_DefaultSaveDependencies[i] = g_CKClassInfo[i].DefaultSaveDependencies;
    }

    // Reset 'Done' flag before next computation
    for (CK_CLASSID i = 0; i < classCount; ++i) {
        g_CKClassInfo[i].Done = FALSE;
    }

    // 7. Compute notification tables
    for (CK_CLASSID i = 0; i < classCount; ++i) {
        ComputeParentsNotifyTable(i);
    }

    // 8. Build common notification lists
    for (CK_CLASSID cls = 0; cls < classCount; ++cls) {
        for (CK_CLASSID target = 0; target < classCount; ++target) {
            if (g_CKClassInfo[cls].CommonToBeNotify.IsSet(target)) {
                // Only add if parent is valid
                if (g_CKClassInfo[cls].Parent != 0 && g_CKClassInfo[cls].Parent < classCount) {
                    g_CKClassInfo[cls].ToNotify.Insert(g_CKClassInfo[cls].Parent, target);
                }
            }
        }
    }
}

CKPluginManager *CKGetPluginManager() {
    return &g_ThePluginManager;
}

int CKGetPrototypeDeclarationCount() {
    return g_PrototypeDeclarationList.Size();
}

CKObjectDeclaration *CKGetPrototypeDeclaration(int n) {
    int i = 0;
    for (auto it = g_PrototypeDeclarationList.Begin(); it != g_PrototypeDeclarationList.End(); ++it) {
        if (i++ == n)
            return *it;
    }
    return nullptr;
}

XObjDeclHashTableIt CKGetPrototypeDeclarationStartIterator() {
    return g_PrototypeDeclarationList.Begin();
}

XObjDeclHashTableIt CKGetPrototypeDeclarationEndIterator() {
    return g_PrototypeDeclarationList.End();
}

CKObjectDeclaration *CKGetObjectDeclarationFromGuid(CKGUID guid) {
    return g_PrototypeDeclarationList[guid];
}

CKBehaviorPrototype *CKGetPrototypeFromGuid(CKGUID guid) {
    XObjDeclHashTableIt it = g_PrototypeDeclarationList.Find(guid);
    if (!it)
        return nullptr;

    CKBehaviorPrototype *proto = (*it)->m_Proto;
    if (proto)
        return proto;

    CKDLL_CREATEPROTOFUNCTION creationFct = (*it)->m_CreationFunction;
    if (!creationFct)
        return nullptr;

    if (creationFct(&proto) != CK_OK)
        return nullptr;

    proto->SetGuid(guid);
    proto->SetApplyToClassID((*it)->m_CompatibleClassID);
    (*it)->m_Proto = proto;
    return proto;
}

CKERROR CKRemovePrototypeDeclaration(CKObjectDeclaration *decl) {
    if (!decl)
        return CKERR_INVALIDPARAMETER;

    CKGUID declGuid = decl->GetGuid();
    g_PrototypeDeclarationList.Remove(declGuid);

    CKBehaviorPrototype *prototype = decl->m_Proto;
    if (prototype) {
        CKGUID protoGuid = prototype->GetGuid();

        for (CKContext **contextIter = g_Contextes.Begin();
             contextIter != g_Contextes.End();
             ++contextIter) {
            CKContext *ctx = *contextIter;
            if (!ctx) continue;

            int behaviorCount = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
            CK_ID *behaviorIds = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);

            for (int i = 0; i < behaviorCount; ++i) {
                CKBehavior *behavior = (CKBehavior *) ctx->GetObject(behaviorIds[i]);
                if (!behavior)
                    continue;

                CKGUID behaviorGuid = behavior->GetPrototypeGuid();

                if (behaviorGuid == protoGuid) {
                    // Reset behavior configuration
                    behavior->SetFunction(nullptr);
                    behavior->SetCallbackFunction(nullptr);

                    ctx->OutputToConsoleExBeep(
                        "Warning : Behavior %s has no more a valid prototype !",
                        behavior->GetName());
                }
            }
        }

        delete prototype;
    }

    delete decl;
    return CK_OK;
}

CKObjectDeclaration *CreateCKObjectDeclaration(CKSTRING Name) {
    return new CKObjectDeclaration(Name);
}

CKBehaviorPrototype *CreateCKBehaviorPrototype(CKSTRING Name) {
    return new CKBehaviorPrototype(Name);
}

CKBehaviorPrototype *CreateCKBehaviorPrototypeRunTime(CKSTRING Name) {
    return new CKBehaviorPrototype(Name);
}

int CKGetClassCount() {
    return g_CKClassInfo.Size();
}

CKClassDesc *CKGetClassDesc(CK_CLASSID cid) {
    return &g_CKClassInfo[cid];
}

CKSTRING CKClassIDToString(CK_CLASSID cid) {
    if (cid >= 0 && cid < g_CKClassInfo.Size()) {
        CKSTRING name = g_CKClassInfo[cid].NameFct();
        if (name)
            return name;
    }
    return "Invalid Class Identifier";
}

CK_CLASSID CKStringToClassID(CKSTRING classname) {
    if (!classname)
        return 0;

    for (CK_CLASSID i = 0; i < g_CKClassInfo.Size(); ++i) {
        CKSTRING name = g_CKClassInfo[i].NameFct();
        if (name && !strcmp(name, classname))
            return i;
    }

    return 0;
}

CKBOOL CKIsChildClassOf(CK_CLASSID child, CK_CLASSID parent) {
    // Special case: every class is its own child
    if (child == parent)
        return TRUE;

    // Special case: class 0 is not a valid class (it's a sentinel for "no parent")
    if (child == 0 || parent == 0)
        return FALSE;

    // Bounds check
    if (child >= g_CKClassInfo.Size() || parent >= g_CKClassInfo.Size())
        return FALSE;

    // Use the bitset to check ancestry
    return g_CKClassInfo[child].Parents.IsSet(parent) != 0;
}

CKBOOL CKIsChildClassOf(CKObject *obj, CK_CLASSID parent) {
    // Handle null object
    if (!obj)
        return FALSE;

    // Special case: class 0 is not a valid class
    if (parent == 0)
        return FALSE;

    // Bounds check
    if (parent >= g_CKClassInfo.Size())
        return FALSE;

    CK_CLASSID child = obj->GetClassID();

    // Special case: every class is its own child
    if (child == parent)
        return TRUE;

    // Use the bitset to check ancestry
    return g_CKClassInfo[child].Parents.IsSet(parent) != 0;
}

CK_CLASSID CKGetParentClassID(CK_CLASSID child) {
    if (child >= 0 && child < g_CKClassInfo.Size())
        return g_CKClassInfo[child].Parent;
    return 0;
}

CK_CLASSID CKGetParentClassID(CKObject *obj) {
    if (obj)
        return g_CKClassInfo[obj->GetClassID()].Parent;
    return 0;
}

CK_CLASSID CKGetCommonParent(CK_CLASSID cid1, CK_CLASSID cid2) {
    if (cid1 <= 0 || cid2 <= 0 || cid1 >= g_CKClassInfo.Size() || cid2 >= g_CKClassInfo.Size()) {
        return 0;
    }

    // Quick check: if one is a child of the other, return the parent
    if (CKIsChildClassOf(cid1, cid2)) {
        return cid2;
    }
    if (CKIsChildClassOf(cid2, cid1)) {
        return cid1;
    }

    // Use the precomputed parent bit arrays for an efficient intersection test
    XBitArray commonParents = g_CKClassInfo[cid1].Parents;
    commonParents.And(g_CKClassInfo[cid2].Parents);

    // Find the most derived common parent (highest derivation level)
    int maxLevel = -1;
    CK_CLASSID commonParent = 0;
    for (int i = 0; i < g_CKClassInfo.Size(); ++i) {
        if (commonParents.IsSet(i)) {
            if (g_CKClassInfo[i].DerivationLevel > maxLevel) {
                maxLevel = g_CKClassInfo[i].DerivationLevel;
                commonParent = i;
            }
        }
    }
    return commonParent;
}

CKObjectArray *CreateCKObjectArray() {
    return new CKObjectArray(nullptr);
}

void DeleteCKObjectArray(CKObjectArray *obj) {
    if (obj) {
        obj->Clear();
        delete obj;
    }
}

CKStateChunk *CKSaveObjectState(CKObject *obj, CKDWORD Flags) {
    if (!obj)
        return nullptr;
    return obj->Save(nullptr, Flags);
}

CKERROR CKReadObjectState(CKObject *obj, CKStateChunk *chunk) {
    if (!obj || !chunk)
        return CKERR_INVALIDPARAMETER;
    return obj->Load(chunk, nullptr);
}

BITMAP_HANDLE CKLoadBitmap(CKSTRING filename) {
    return nullptr;
}

CKBOOL CKSaveBitmap(CKSTRING filename, BITMAP_HANDLE bm) {
    return 0;
}

CKBOOL CKSaveBitmap(CKSTRING filename, VxImageDescEx &desc) {
    return 0;
}

void CKConvertEndianArray32(void *buf, int DwordCount) {
    // EMPTY
}

void CKConvertEndianArray16(void *buf, int DwordCount) {
    // EMPTY
}

CKDWORD CKConvertEndian32(CKDWORD dw) {
    return dw;
}

CKWORD CKConvertEndian16(CKWORD w) {
    return w;
}

CKDWORD CKComputeDataCRC(char *data, int size, CKDWORD PreviousCRC) {
    return adler32(PreviousCRC, (const Bytef *) data, size);
}

char *CKPackData(char *Data, int size, int &NewSize, int compressionLevel) {
    char *buffer = new char[size];
    if (!buffer)
        return nullptr;

    if (compress2((Bytef *) buffer, (uLongf *) &NewSize, (const Bytef *) Data, size, compressionLevel) == Z_OK) {
        char *buffer2 = new char[NewSize];
        if (buffer2) {
            memcpy(buffer2, buffer, NewSize);
            delete[] buffer;
            return buffer2;
        }
    }

    delete[] buffer;
    return nullptr;
}

char *CKUnPackData(int DestSize, char *SrcBuffer, int SrcSize) {
    char *buffer = new char[DestSize];
    if (!buffer)
        return nullptr;

    if (uncompress((Bytef *) buffer, (uLongf *) &DestSize, (const Bytef *) SrcBuffer, SrcSize) == Z_OK) {
        return buffer;
    }

    delete[] buffer;
    return nullptr;
}

CKSTRING CKStrdup(CKSTRING string) {
    if (!string)
        return nullptr;

    size_t len = strlen(string);
    CKSTRING str = new char[len + 1];
    strcpy(str, string);
    return str;
}

CKSTRING CKStrupr(CKSTRING string) {
    if (!string)
        return nullptr;

    for (char *pch = string; *pch; ++pch) {
        char ch = *pch;
        if (ch >= 'a' && ch <= 'z')
            *pch = ch - 0x20;
    }
    return string;
}

CKSTRING CKStrlwr(CKSTRING string) {
    if (!string)
        return nullptr;

    for (char *pch = string; *pch; ++pch) {
        char ch = *pch;
        if (ch >= 'A' && ch <= 'Z')
            *pch = ch + 0x20;
    }
    return string;
}

CKBitmapProperties *CKCopyBitmapProperties(CKBitmapProperties *bp) {
    if (!bp)
        return nullptr;
    if (bp->m_Size <= 0 || bp->m_Size >= 256)
        return nullptr;
    CKBitmapProperties *nbp = (CKBitmapProperties *) new CKBYTE[bp->m_Size];
    memcpy(nbp, bp, bp->m_Size);
    return nbp;
}

void CKCopyDefaultClassDependencies(CKDependencies &deps, CK_DEPENDENCIES_OPMODE mode) {
    switch (mode) {
    case CK_DEPENDENCIES_COPY:
        deps = g_DefaultCopyDependencies;
        break;
    case CK_DEPENDENCIES_DELETE:
        deps = g_DefaultDeleteDependencies;
        break;
    case CK_DEPENDENCIES_REPLACE:
        deps = g_DefaultReplaceDependencies;
        break;
    case CK_DEPENDENCIES_SAVE:
        deps = g_DefaultSaveDependencies;
        break;
    case CK_DEPENDENCIES_BUILD:
        deps = g_DefaultCopyDependencies;
        break;
    default:
        break;
    }
}

CKDependencies *CKGetDefaultClassDependencies(CK_DEPENDENCIES_OPMODE mode) {
    switch (mode) {
    case CK_DEPENDENCIES_COPY:
        return &g_DefaultCopyDependencies;
    case CK_DEPENDENCIES_DELETE:
        return &g_DefaultDeleteDependencies;
    case CK_DEPENDENCIES_REPLACE:
        return &g_DefaultReplaceDependencies;
    case CK_DEPENDENCIES_SAVE:
        return &g_DefaultSaveDependencies;
    case CK_DEPENDENCIES_BUILD:
        return &g_DefaultCopyDependencies;
    default:
        return nullptr;
    }
}

void CKDeletePointer(void *ptr) {
    delete ptr;
}

CKERROR CKCopyAllAttributes(CKBeObject *Src, CKBeObject *Dest) {
    if (!Src || !Dest)
        return CKERR_INVALIDPARAMETER;

    int attributeCount = Src->GetAttributeCount();
    if (attributeCount <= 0)
        return CK_OK;

    for (int i = 0; i < attributeCount; ++i) {
        int attributeType = Src->GetAttributeType(i);

        if (Dest->HasAttribute(attributeType))
            return CKERR_ALREADYPRESENT;

        if (!Dest->SetAttribute(attributeType))
            return CKERR_INVALIDOPERATION;

        CKParameter *sourceParam = Src->GetAttributeParameter(attributeType);
        CKParameter *destParam = Dest->GetAttributeParameter(attributeType);

        if (sourceParam && destParam) {
            destParam->CopyValue(sourceParam);
        }
    }

    return CK_OK;
}

CKERROR CKMoveAllScripts(CKBeObject *Src, CKBeObject *Dest) {
    if (!Src || !Dest)
        return CKERR_INVALIDPARAMETER;

    const int count = Src->GetScriptCount();
    for (int i = 0; i < count; ++i) {
        CKBehavior *beh = Src->GetScript(i);
        if (beh) {
            CKMoveScript(Src, Dest, beh);
        }
    }

    return CK_OK;
}

CKERROR CKMoveScript(CKBeObject *Src, CKBeObject *Dest, CKBehavior *Beh) {
    CKScene *sourceLevelScene = nullptr;
    CKScene *targetLevelScene = nullptr;
    CKBOOL isLevelOperation = FALSE;

    if (CKIsChildClassOf(Src, CKCID_LEVEL) && CKIsChildClassOf(Dest, CKCID_LEVEL)) {
        isLevelOperation = TRUE;
        sourceLevelScene = ((CKLevel *) Src)->GetLevelScene();
        targetLevelScene = ((CKLevel *) Dest)->GetLevelScene();
    }

    XHashTable<CKSceneObjectDesc, CK_ID> sceneDescMap;
    int sceneCount = Beh->GetSceneInCount();
    for (int sceneIdx = 0; sceneIdx < sceneCount; ++sceneIdx) {
        CKScene *currentScene = Beh->GetSceneIn(sceneIdx);
        if (!currentScene)
            continue;

        CKSceneObjectDesc *sceneDesc = currentScene->GetSceneObjectDesc(Beh);
        if (!sceneDesc)
            continue;

        CKSceneObjectDesc desc;
        desc.m_Object = sceneDesc->m_Object;
        desc.m_InitialValue = nullptr;
        desc.m_Flags = sceneDesc->m_Flags;
        if (sceneDesc->m_InitialValue) {
            desc.m_InitialValue = new CKStateChunk(*sceneDesc->m_InitialValue);
        }

        if (isLevelOperation && targetLevelScene && currentScene == sourceLevelScene) {
            sceneDescMap.Insert(targetLevelScene->GetID(), desc);
        } else {
            sceneDescMap.Insert(currentScene->GetID(), desc);
        }
    }

    Src->RemoveScript(Beh->GetID());
    if (Dest->AddScript(Beh) != CK_OK) {
        Src->AddScript(Beh);
    }

    const int updatedSceneCount = Beh->GetSceneInCount();
    for (int sceneIdx = 0; sceneIdx < updatedSceneCount; ++sceneIdx) {
        CKScene *currentScene = Beh->GetSceneIn(sceneIdx);
        if (!currentScene)
            continue;

        CKSceneObjectDesc *sceneDesc = currentScene->GetSceneObjectDesc(Beh);
        if (!sceneDesc)
            continue;

        CKSceneObjectDesc *desc = sceneDescMap.FindPtr(currentScene->GetID());
        if (desc) {
            *sceneDesc = *desc;
        }
    }

    return CK_OK;
}

void CKRemapObjectParameterValue(CKContext *ckContext, CK_ID oldID, CK_ID newID, CK_CLASSID cid, CKBOOL derived) {
    const XObjectPointerArray &params = ckContext->GetObjectListByType(CKCID_PARAMETEROUT, TRUE);
    for (XObjectPointerArray::Iterator it = params.Begin(); it != params.End(); ++it) {
        CKParameter *param = (CKParameter *) *it;
        CK_CLASSID pcid = param->GetParameterClassID();
        if (pcid == cid || (derived && CKIsChildClassOf(pcid, cid))) {
            if (*(CK_ID *) param->GetReadDataPtr(TRUE) == oldID)
                param->SetValue(&newID);
        }
    }
}

void CKStoreDeclaration(XObjectDeclarationArray *reg, CKObjectDeclaration *a) {
    reg->PushBack(a);
}

void CKClassNeedNotificationFrom(CK_CLASSID Cid1, CK_CLASSID Cid2) {
    g_CKClassInfo[Cid1].ToBeNotify.Or(g_CKClassInfo[Cid2].Children);
}

void CKClassRegisterAssociatedParameter(CK_CLASSID Cid, CKGUID pguid) {
    g_CKClassInfo[Cid].Parameter = pguid;
}

void CKClassRegisterDefaultDependencies(CK_CLASSID Cid, CKDWORD depend_Mask, int mode) {
    switch (mode) {
    case CK_DEPENDENCIES_COPY:
        g_CKClassInfo[Cid].DefaultCopyDependencies = depend_Mask;
        break;
    case CK_DEPENDENCIES_DELETE:
        g_CKClassInfo[Cid].DefaultDeleteDependencies = depend_Mask;
        break;
    case CK_DEPENDENCIES_REPLACE:
        g_CKClassInfo[Cid].DefaultReplaceDependencies = depend_Mask;
        break;
    case CK_DEPENDENCIES_SAVE:
        g_CKClassInfo[Cid].DefaultSaveDependencies = depend_Mask;
        break;
    default:
        break;
    }
}

void CKClassRegisterDefaultOptions(CK_CLASSID Cid, CKDWORD options_Mask) {
    g_CKClassInfo[Cid].DefaultOptions = options_Mask;
}

CK_CLASSID CKClassGetNewIdentifier() {
    CK_CLASSID cid = g_CKClassInfo.Size();
    if (cid < CKCID_MAXCLASSID)
        cid = CKCID_MAXCLASSID;
    if (cid >= g_MaxClassID)
        g_MaxClassID = cid + 1;
    return cid;
}

void CKClassRegister(CK_CLASSID Cid, CK_CLASSID ParentCid, CKCLASSREGISTERFCT RegisterFct,
                     CKCLASSCREATIONFCT CreationFct,
                     CKCLASSNAMEFCT NameFct, CKCLASSDEPENDENCIESFCT DependsFct,
                     CKCLASSDEPENDENCIESCOUNTFCT DependsCountFct) {
    if (Cid >= g_CKClassInfo.Size())
        g_CKClassInfo.Resize(Cid + 1);

    CKClassDesc desc;
    desc.Parent = ParentCid != Cid ? ParentCid : 0;
    desc.RegisterFct = RegisterFct;
    desc.CreationFct = CreationFct;
    desc.NameFct = NameFct;
    desc.DependsFct = DependsFct;
    desc.DependsCountFct = DependsCountFct;
    g_CKClassInfo[Cid] = desc;
}

CKSTRING CKErrorToString(CKERROR err) {
    return "Unknown Error";
}
