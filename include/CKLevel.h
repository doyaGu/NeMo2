#ifndef CKLEVEL_H
#define CKLEVEL_H

#include "CKBeObject.h"

/*************************************************
{filename:CKLevel}
Summary: Main composition management

Remarks:
+ A composition must contain an unique CKLevel object accessible through CKContext::GetCurrentLevel
+ The level manages a list of CKScene and provides the methods to switch from one scene to another.
+ All objects used in the composition must be referenced in the level in order to be saved.

+ The level also manages a list of CKRenderContext on which the rendering should be done. It uses this list
to automatically add an object to the rendercontext when it is added to the level.

+ A Level automatically creates a scene it owns. This scene is called the Level Scene and is used to
store the reference of every objects in the composition. This scene contains additional information about
objects such as how do they behave at startup or initial conditions.

+ The class id of CKLevel is CKCID_LEVEL.


See also: CKContext::GetCurrentLevel,CKScene
*************************************************/
class CKLevel : public CKBeObject
{
    friend class CKScene;
    friend class CKContext;

public:
    //-----------------------------------------------------
    //	Object Management
    //  Object are directly inserted or removed from the level Scene
    DLL_EXPORT CKERROR AddObject(CKObject *obj);
    DLL_EXPORT CKERROR RemoveObject(CKObject *obj);
    DLL_EXPORT CKERROR RemoveObject(CK_ID objID);
    DLL_EXPORT void BeginAddSequence(CKBOOL Begin);
    DLL_EXPORT void BeginRemoveSequence(CKBOOL Begin);

    //--------------------------------------------------------
    // Return the list of object ( owned by this level )
    DLL_EXPORT const XObjectPointerArray &ComputeObjectList(CK_CLASSID cid, CKBOOL derived = TRUE);

    //----------------------------------------------------------
    // Place Management
    DLL_EXPORT CKERROR AddPlace(CKPlace *pl);
    DLL_EXPORT CKERROR RemovePlace(CKPlace *pl);
    DLL_EXPORT CKPlace *RemovePlace(int pos);
    DLL_EXPORT CKPlace *GetPlace(int pos);
    DLL_EXPORT int GetPlaceCount();

    //-----------------------------------------------------------
    // Scene Management
    DLL_EXPORT CKERROR AddScene(CKScene *scn);
    DLL_EXPORT CKERROR RemoveScene(CKScene *scn);
    DLL_EXPORT CKScene *RemoveScene(int pos);
    DLL_EXPORT CKScene *GetScene(int pos);
    DLL_EXPORT int GetSceneCount();

    //-----------------------------------------------------------
    // Active Scene
    DLL_EXPORT CKERROR SetNextActiveScene(CKScene *, CK_SCENEOBJECTACTIVITY_FLAGS Active = CK_SCENEOBJECTACTIVITY_SCENEDEFAULT,
                               CK_SCENEOBJECTRESET_FLAGS Reset = CK_SCENEOBJECTRESET_RESET);

    DLL_EXPORT CKERROR LaunchScene(CKScene *, CK_SCENEOBJECTACTIVITY_FLAGS Active = CK_SCENEOBJECTACTIVITY_SCENEDEFAULT,
                        CK_SCENEOBJECTRESET_FLAGS Reset = CK_SCENEOBJECTRESET_RESET);
    DLL_EXPORT CKScene *GetCurrentScene();

    //----------------------------------------------------------
    //	Render Context functions
    DLL_EXPORT void AddRenderContext(CKRenderContext *, CKBOOL Main = FALSE);
    DLL_EXPORT void RemoveRenderContext(CKRenderContext *);
    DLL_EXPORT int GetRenderContextCount();
    DLL_EXPORT CKRenderContext *GetRenderContext(int count);

    //-----------------------------------------------------------
    //	Main Scene for this Level
    DLL_EXPORT CKScene *GetLevelScene();

    //-----------------------------------------------------------
    //	Merge
    DLL_EXPORT CKERROR Merge(CKLevel *mergedLevel, CKBOOL asScene);

#ifndef NO_OLDERVERSION_COMPATIBILITY
    virtual void ApplyPatchForOlderVersion(int NbObject, CKFileObject *FileObjects);
#endif

    //-------------------------------------------------------------------

    DLL_EXPORT CKLevel(CKContext *Context, CKSTRING name = NULL);
    DLL_EXPORT virtual ~CKLevel();
    DLL_EXPORT virtual CK_CLASSID GetClassID();

    DLL_EXPORT virtual void PreSave(CKFile *file, CKDWORD flags);
    DLL_EXPORT virtual CKStateChunk *Save(CKFile *file, CKDWORD flags);
    DLL_EXPORT virtual CKERROR Load(CKStateChunk *chunk, CKFile *file);

    DLL_EXPORT virtual void CheckPreDeletion();
    DLL_EXPORT virtual void CheckPostDeletion();

    DLL_EXPORT virtual size_t GetMemoryOccupation();
    DLL_EXPORT virtual CKBOOL IsObjectUsed(CKObject *o, CK_CLASSID cid);

    //--------------------------------------------
    // Dependencies Functions
    DLL_EXPORT virtual CKERROR PrepareDependencies(CKDependenciesContext &context);

    //--------------------------------------------
    // Class Registering
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKLevel *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    // Dynamic Cast method (returns NULL if the object can't be cast)
    static CKLevel *Cast(CKObject *iO)
    {
        return CKIsChildClassOf(iO, CKCID_LEVEL) ? (CKLevel *)iO : NULL;
    }

    virtual void AddToScene(CKScene *scene, CKBOOL dependencies = TRUE);
    virtual void RemoveFromScene(CKScene *scene, CKBOOL dependencies = TRUE);

    DLL_EXPORT void Reset();
    DLL_EXPORT void ActivateAllScript();
    DLL_EXPORT void CreateLevelScene();
    DLL_EXPORT void PreProcess();

protected:
    XObjectPointerArray m_SceneList;
    XObjectPointerArray m_RenderContextList;
    CK_ID m_CurrentScene;
    CK_ID m_DefaultScene;
    CK_ID m_NextActiveScene;
    CK_SCENEOBJECTACTIVITY_FLAGS m_NextSceneActivityFlags;
    CK_SCENEOBJECTRESET_FLAGS m_NextSceneResetFlags;
    CKBOOL m_IsReseted;
};

#endif // CKLEVEL_H
