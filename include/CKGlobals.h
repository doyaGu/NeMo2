#ifndef CKGLOBALS_H
#define CKGLOBALS_H

#include "CKDefines.h"
#include "CKDefines2.h"
#include "XHashTable.h"

typedef XHashTable<CKObjectDeclaration *, CKGUID>::Iterator XObjDeclHashTableIt;

//-----------------------------------------------
// Initializations functions

DLL_EXPORT CKERROR CKStartUp();
DLL_EXPORT CKERROR CKShutdown();

DLL_EXPORT CKContext *GetCKContext(int pos);

DLL_EXPORT CKObject *CKGetObject(CKContext *iCtx, CK_ID iID);

DLL_EXPORT CKERROR CKCreateContext(CKContext **iContext, WIN_HANDLE iWin, int iRenderEngine, CKDWORD Flags);
DLL_EXPORT CKERROR CKCloseContext(CKContext *);

DLL_EXPORT CKSTRING CKGetStartPath();
DLL_EXPORT CKSTRING CKGetPluginsPath();

DLL_EXPORT void CKDestroyObject(CKObject *o, CKDWORD Flags = 0, CKDependencies *dep = NULL);

DLL_EXPORT CKDWORD CKGetVersion();

DLL_EXPORT void CKBuildClassHierarchyTable();

DLL_EXPORT CKPluginManager *CKGetPluginManager();

//----------------------------------------------------------
// Behavior prototype declaration functions

DLL_EXPORT int CKGetPrototypeDeclarationCount();
DLL_EXPORT CKObjectDeclaration *CKGetPrototypeDeclaration(int n);

DLL_EXPORT XObjDeclHashTableIt CKGetPrototypeDeclarationStartIterator();

DLL_EXPORT XObjDeclHashTableIt CKGetPrototypeDeclarationEndIterator();

DLL_EXPORT CKObjectDeclaration *CKGetObjectDeclarationFromGuid(CKGUID guid);
DLL_EXPORT CKBehaviorPrototype *CKGetPrototypeFromGuid(CKGUID guid);
DLL_EXPORT CKERROR CKRemovePrototypeDeclaration(CKObjectDeclaration *objdecl);
DLL_EXPORT CKObjectDeclaration *CreateCKObjectDeclaration(CKSTRING Name);
DLL_EXPORT CKBehaviorPrototype *CreateCKBehaviorPrototype(CKSTRING Name);
DLL_EXPORT CKBehaviorPrototype *CreateCKBehaviorPrototypeRunTime(CKSTRING Name);

#ifdef VIRTOOLS_RUNTIME_VERSION
#define CreateCKBehaviorPrototype CreateCKBehaviorPrototypeRunTime
#endif

/**********************************************
Summary: Helper macro to register a behavior declaration

Arguments:
    reg: A pointer to the XObjectDeclarationArray given in the RegisterBehaviorDeclarations function.
    fct: A function that creates the behavior object declaration (CKObjectDeclaration) and returns it.
See Also:Main Steps of Building Block Creation
*************************************************/
#define RegisterBehavior(reg, fct) \
    CKObjectDeclaration *fct();    \
    CKStoreDeclaration(reg, fct());

//----------------------------------------------------------
// Class Hierarchy Management

DLL_EXPORT int CKGetClassCount();
DLL_EXPORT CKClassDesc *CKGetClassDesc(CK_CLASSID cid);
DLL_EXPORT CKSTRING CKClassIDToString(CK_CLASSID cid);
DLL_EXPORT CK_CLASSID CKStringToClassID(CKSTRING classname);

DLL_EXPORT CKBOOL CKIsChildClassOf(CK_CLASSID child, CK_CLASSID parent);
DLL_EXPORT CKBOOL CKIsChildClassOf(CKObject *obj, CK_CLASSID parent);
DLL_EXPORT CK_CLASSID CKGetParentClassID(CK_CLASSID child);
DLL_EXPORT CK_CLASSID CKGetParentClassID(CKObject *obj);
DLL_EXPORT CK_CLASSID CKGetCommonParent(CK_CLASSID cid1, CK_CLASSID cid2);

//-----------------------------------------------
// Array Creation Functions

DLL_EXPORT CKObjectArray *CreateCKObjectArray();
DLL_EXPORT void DeleteCKObjectArray(CKObjectArray *obj);

//-----------------------------------------------
// StateChunk Creation Functions

DLL_EXPORT CKStateChunk *CreateCKStateChunk(CK_CLASSID id, CKFile *file = NULL);
DLL_EXPORT CKStateChunk *CreateCKStateChunk(CKStateChunk *chunk);
DLL_EXPORT void DeleteCKStateChunk(CKStateChunk *chunk);

DLL_EXPORT CKStateChunk *CKSaveObjectState(CKObject *obj, CKDWORD Flags = CK_STATESAVE_ALL);
DLL_EXPORT CKERROR CKReadObjectState(CKObject *obj, CKStateChunk *chunk);

//-----------------------------------------------
// Bitmap utilities

DLL_EXPORT BITMAP_HANDLE CKLoadBitmap(CKSTRING filename);
DLL_EXPORT CKBOOL CKSaveBitmap(CKSTRING filename, BITMAP_HANDLE bm);
DLL_EXPORT CKBOOL CKSaveBitmap(CKSTRING filename, VxImageDescEx &desc);

//------------------------------------------------
//--- Endian Conversion utilities

DLL_EXPORT void CKConvertEndianArray32(void *buf, size_t DwordCount);
DLL_EXPORT void CKConvertEndianArray16(void *buf, size_t DwordCount);
DLL_EXPORT CKDWORD CKConvertEndian32(CKDWORD dw);
DLL_EXPORT CKWORD CKConvertEndian16(CKWORD w);

//------------------------------------------------
// Compression utilities

DLL_EXPORT CKDWORD CKComputeDataCRC(const CKBYTE *data, size_t size, CKDWORD PreviousCRC = 0);
DLL_EXPORT char *CKPackData(const CKBYTE *Data, size_t size, size_t &NewSize, int compressionLevel);
DLL_EXPORT char *CKUnPackData(size_t DestSize, const CKBYTE *SrcBuffer, size_t SrcSize);

//-------------------------------------------------
// String Utilities

DLL_EXPORT CKSTRING CKStrdup(CKSTRING string);
DLL_EXPORT CKSTRING CKStrupr(CKSTRING string);
DLL_EXPORT CKSTRING CKStrlwr(CKSTRING string);

//-------------------------------------------------
// CKBitmapProperties Utilities

DLL_EXPORT CKBitmapProperties *CKCopyBitmapProperties(CKBitmapProperties *bp);

#define CKDeleteBitmapProperties(bp) CKDeletePointer((void *)bp)

//-------------------------------------------------
// Class Dependencies utilities

DLL_EXPORT void CKCopyDefaultClassDependencies(CKDependencies &deps, CK_DEPENDENCIES_OPMODE mode);
DLL_EXPORT CKDependencies *CKGetDefaultClassDependencies(CK_DEPENDENCIES_OPMODE mode);

DLL_EXPORT void CKDeletePointer(void *ptr);

//-------------------------------------------------------------------

//-------------------------------------------------
// Merge Utilities
DLL_EXPORT CKERROR CKCopyAllAttributes(CKBeObject *Src, CKBeObject *Dest);
DLL_EXPORT CKERROR CKMoveAllScripts(CKBeObject *Src, CKBeObject *Dest);
DLL_EXPORT CKERROR CKMoveScript(CKBeObject *Src, CKBeObject *Dest, CKBehavior *Beh);
DLL_EXPORT void CKRemapObjectParameterValue(CKContext *ckContext, CK_ID oldID, CK_ID newID, CK_CLASSID cid = CKCID_OBJECT, CKBOOL derived = TRUE);

DLL_EXPORT void CKStoreDeclaration(XObjectDeclarationArray *reg, CKObjectDeclaration *a);

//-------- CKClass Registration
#define CKCLASSNOTIFYFROM(cls1, cls2) CKClassNeedNotificationFrom(cls1::m_ClassID, cls2::m_ClassID)
#define CKCLASSNOTIFYFROMCID(cls1, cid) CKClassNeedNotificationFrom(cls1::m_ClassID, cid)
#define CKPARAMETERFROMCLASS(cls, pguid) CKClassRegisterAssociatedParameter(cls::m_ClassID, pguid)
#define CKCLASSDEFAULTCOPYDEPENDENCIES(cls1, depend_Mask) CKClassRegisterDefaultDependencies(cls1::m_ClassID, depend_Mask, CK_DEPENDENCIES_COPY)
#define CKCLASSDEFAULTDELETEDEPENDENCIES(cls1, depend_Mask) CKClassRegisterDefaultDependencies(cls1::m_ClassID, depend_Mask, CK_DEPENDENCIES_DELETE)
#define CKCLASSDEFAULTREPLACEDEPENDENCIES(cls1, depend_Mask) CKClassRegisterDefaultDependencies(cls1::m_ClassID, depend_Mask, CK_DEPENDENCIES_REPLACE)
#define CKCLASSDEFAULTSAVEDEPENDENCIES(cls1, depend_Mask) CKClassRegisterDefaultDependencies(cls1::m_ClassID, depend_Mask, CK_DEPENDENCIES_SAVE)
#define CKCLASSDEFAULTOPTIONS(cls1, options_Mask) CKClassRegisterDefaultOptions(cls1::m_ClassID, options_Mask)

#define CKCLASSREGISTER(cls1, Parent_Class)                                                                                                                                                         \
    {                                                                                                                                                                                               \
        if (cls1::m_ClassID <= 0)                                                                                                                                                                   \
            cls1::m_ClassID = CKClassGetNewIdentifier();                                                                                                                                            \
        CKClassRegister(cls1::m_ClassID, Parent_Class::m_ClassID, cls1::Register, (CKCLASSCREATIONFCT)cls1::CreateInstance, cls1::GetClassName, cls1::GetDependencies, cls1::GetDependenciesCount); \
    }

#define CKCLASSREGISTERCID(cls1, Parent_Class)                                                                                                                                           \
    {                                                                                                                                                                                    \
        if (cls1::m_ClassID <= 0)                                                                                                                                                        \
            cls1::m_ClassID = CKClassGetNewIdentifier();                                                                                                                                 \
        CKClassRegister(cls1::m_ClassID, Parent_Class, cls1::Register, (CKCLASSCREATIONFCT)cls1::CreateInstance, cls1::GetClassName, cls1::GetDependencies, cls1::GetDependenciesCount); \
    }

DLL_EXPORT void CKClassNeedNotificationFrom(CK_CLASSID Cid1, CK_CLASSID Cid2);
DLL_EXPORT void CKClassRegisterAssociatedParameter(CK_CLASSID Cid, CKGUID pguid);
DLL_EXPORT void CKClassRegisterDefaultDependencies(CK_CLASSID Cid, CKDWORD depend_Mask, int mode);
DLL_EXPORT void CKClassRegisterDefaultOptions(CK_CLASSID Cid, CKDWORD options_Mask);
DLL_EXPORT CK_CLASSID CKClassGetNewIdentifier();
DLL_EXPORT void CKClassRegister(CK_CLASSID Cid, CK_CLASSID Parent_Cid,
                     CKCLASSREGISTERFCT registerfct, CKCLASSCREATIONFCT creafct,
                     CKCLASSNAMEFCT NameFct, CKCLASSDEPENDENCIESFCT DependsFct,
                     CKCLASSDEPENDENCIESCOUNTFCT DependsCountFct);

#endif // CKGLOBALS_H
