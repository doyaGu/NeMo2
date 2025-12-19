#ifndef CKSCENE_H
#define CKSCENE_H

#include "CKBeObject.h"
#include "CKSceneObjectDesc.h"
#include "XHashTable.h"

typedef XHashTable<CKSceneObjectDesc, CK_ID> CKSODHash;
typedef CKSODHash::Iterator CKSODHashIt;

/*************************************************
Summary: Iterators on objects in a scene.

Remarks:
+ Objects in a scene are stored in a Hash table, this class can be used with CKScene::GetObjectIterator
to iterate on objects in a scene.

For example if can be use in following way:

        CKSceneObjectIterator it = scene->GetObjectIterator();
        CKObject* obj;

        while(!it.End()) {
            obj = context->GetObject(it.GetObjectID());
            ....
            ...
            it++;
        }

See Also:CKScene::GetObjectIterator
*************************************************/
class CKSceneObjectIterator
{
public:
    CKSceneObjectIterator() = delete;
    CKSceneObjectIterator(CKSODHashIt it) : m_Iterator(it) {}
    // Summary:Returns the ID of the current object.
    // Return Value
    //	CK_ID of the current object.
    CK_ID GetObjectID() {
        if (!m_Iterator.m_Node)
            return 0;
        return m_Iterator.GetKey();
    }

    CKSceneObjectDesc *GetObjectDesc() {
        if (!m_Iterator.m_Node)
            return nullptr;
        return m_Iterator;
    }
    // Summary:Reset iterator to start of the list.
    void Rewind()
    {
        if (!m_Iterator.m_Table) {
            m_Iterator = CKSODHashIt();
            return;
        }
        m_Iterator = m_Iterator.m_Table->Begin();
    }

    void RemoveAt()
    {
        if (!m_Iterator.m_Table || !m_Iterator.m_Node) {
            m_Iterator = CKSODHashIt();
            return;
        }
        m_Iterator = m_Iterator.m_Table->Remove(m_Iterator);
    }

    // Summary:Indicates if end of list is reached.
    // Return Value
    //	Returns TRUE if the iterator is at the end of the list of objects.
    int End() {
        if (!m_Iterator.m_Table)
            return TRUE;
        return m_Iterator == m_Iterator.m_Table->End();
    }

    CKSceneObjectIterator &operator++(int)
    {
        ++m_Iterator;
        return *this;
    }

    CKSODHashIt m_Iterator;
};

/*****************************************************************************
{filename:CKScene}
Summary: Narrative management.

Remarks:
    + A level in Virtools can counts several scenes or no scene at all.
    A scene is used to described what objects should be present,
    and how should they act (which behaviors should be active).

    + An object in the scene can have an initial value stored (called Initial Conditions)

    + A scene can also stored some render options that can be set when the scene becomes
    active such as background color,fog options or starting camera. Changing these options at runtime
    does not affect the current rendering, they are only use when the scene becomes active.

    + The class id of CKScene is CKCID_SCENE.

See also:CKLevel::LaunchScene
******************************************************************************/
class CKScene : public CKBeObject
{
    friend class CKSceneObject;

public:
    //-----------------------------------------------------------
    // Objects functions
    // Adds an Object to this scene with all dependant objects
    DLL_EXPORT void AddObjectToScene(CKSceneObject *o, CKBOOL dependencies = TRUE);
    DLL_EXPORT void RemoveObjectFromScene(CKSceneObject *o, CKBOOL dependencies = TRUE);
    DLL_EXPORT CKBOOL IsObjectHere(CKObject *o);
    DLL_EXPORT void BeginAddSequence(CKBOOL Begin);
    DLL_EXPORT void BeginRemoveSequence(CKBOOL Begin);

    //-----------------------------------------------------------
    // Object List
    DLL_EXPORT int GetObjectCount();
    DLL_EXPORT void AddObject(CKSceneObject *o);
    DLL_EXPORT void RemoveObject(CKSceneObject *o);
    DLL_EXPORT void RemoveAllObjects();
    DLL_EXPORT const XObjectPointerArray &ComputeObjectList(CK_CLASSID cid, CKBOOL derived = TRUE);
    DLL_EXPORT CKERROR ComputeObjectList(CKObjectArray *array, CK_CLASSID cid, CKBOOL derived = TRUE);

    //-----------------------------------------------------------
    // Object Settings by index in list
    DLL_EXPORT CKSceneObjectIterator GetObjectIterator();
    DLL_EXPORT CKSceneObjectDesc *GetSceneObjectDesc(CKSceneObject *o);

    //---- BeObject and Script Activation/deactivation
    DLL_EXPORT void Activate(CKSceneObject *o, CKBOOL Reset);
    DLL_EXPORT void DeActivate(CKSceneObject *o);

    DLL_EXPORT void ActivateBeObject(CKBeObject *beo, CKBOOL activate, CKBOOL reset);
    DLL_EXPORT void ActivateScript(CKBehavior *beh, CKBOOL activate, CKBOOL reset);

    //-----------------------------------------------------------
    // Object Settings by object
    DLL_EXPORT void SetObjectFlags(CKSceneObject *o, CK_SCENEOBJECT_FLAGS flags);
    DLL_EXPORT CK_SCENEOBJECT_FLAGS GetObjectFlags(CKSceneObject *o);
    DLL_EXPORT CK_SCENEOBJECT_FLAGS ModifyObjectFlags(CKSceneObject *o, CKDWORD Add, CKDWORD Remove);
    DLL_EXPORT CKBOOL SetObjectInitialValue(CKSceneObject *o, CKStateChunk *chunk);
    DLL_EXPORT CKStateChunk *GetObjectInitialValue(CKSceneObject *o);
    DLL_EXPORT CKBOOL IsObjectActive(CKSceneObject *o);

    //-----------------------------------------------------------
    // Render Settings
    DLL_EXPORT void ApplyEnvironmentSettings(XObjectPointerArray *renderContexts = NULL);
    DLL_EXPORT void UseEnvironmentSettings(CKBOOL use = TRUE);
    DLL_EXPORT CKBOOL EnvironmentSettings();

    //-----------------------------------------------------------
    // Ambient Light
    DLL_EXPORT void SetAmbientLight(CKDWORD Color);
    DLL_EXPORT CKDWORD GetAmbientLight();

    //-----------------------------------------------------------
    //	Fog Access
    DLL_EXPORT void SetFogMode(VXFOG_MODE Mode);
    DLL_EXPORT void SetFogStart(float Start);
    DLL_EXPORT void SetFogEnd(float End);
    DLL_EXPORT void SetFogDensity(float Density);
    DLL_EXPORT void SetFogColor(CKDWORD Color);

    DLL_EXPORT VXFOG_MODE GetFogMode();
    DLL_EXPORT float GetFogStart();
    DLL_EXPORT float GetFogEnd();
    DLL_EXPORT float GetFogDensity();
    DLL_EXPORT CKDWORD GetFogColor();

    //---------------------------------------------------------
    // Background
    DLL_EXPORT void SetBackgroundColor(CKDWORD Color);
    DLL_EXPORT CKDWORD GetBackgroundColor();

    DLL_EXPORT void SetBackgroundTexture(CKTexture *texture);
    DLL_EXPORT CKTexture *GetBackgroundTexture();

    //--------------------------------------------------------
    // Active camera
    DLL_EXPORT void SetStartingCamera(CKCamera *camera);
    DLL_EXPORT CKCamera *GetStartingCamera();

    //-----------------------------------------------------------
    // Level functions
    DLL_EXPORT CKLevel *GetLevel();
    DLL_EXPORT void SetLevel(CKLevel *level);

    //-----------------------------------------------------------
    // Merge functions
    DLL_EXPORT CKERROR Merge(CKScene *mergedScene, CKLevel *fromLevel = NULL);

    //-------------------------------------------------------------------

    DLL_EXPORT void Init(XObjectPointerArray &renderContexts, CK_SCENEOBJECTACTIVITY_FLAGS activityFlags, CK_SCENEOBJECTRESET_FLAGS resetFlags);
    DLL_EXPORT void Stop(XObjectPointerArray &renderContexts, CKBOOL reset = TRUE);

    void Launch();

    //-----------------------------------------------------------
    // Virtual functions
    DLL_EXPORT CKScene(CKContext *Context, CKSTRING name = NULL);
    virtual ~CKScene();
    virtual CK_CLASSID GetClassID();

    virtual void PreSave(CKFile *file, CKDWORD flags);
    virtual CKStateChunk *Save(CKFile *file, CKDWORD flags);
    virtual CKERROR Load(CKStateChunk *chunk, CKFile *file);

    virtual void PreDelete();
    virtual void CheckPostDeletion();

    virtual int GetMemoryOccupation();
    virtual int IsObjectUsed(CKObject *o, CK_CLASSID cid);

    //--------------------------------------------
    // Dependencies Functions
    virtual CKERROR PrepareDependencies(CKDependenciesContext &context);
    virtual CKERROR RemapDependencies(CKDependenciesContext &context);
    virtual CKERROR Copy(CKObject &o, CKDependenciesContext &context);

    //--------------------------------------------
    // Class Registering
    DLL_EXPORT static CKSTRING GetClassName();
    DLL_EXPORT static int GetDependenciesCount(int mode);
    DLL_EXPORT static CKSTRING GetDependencies(int i, int mode);
    DLL_EXPORT static void Register();
    DLL_EXPORT static CKScene *CreateInstance(CKContext *Context);
    DLL_EXPORT static CK_CLASSID m_ClassID;

    // Dynamic Cast method (returns NULL if the object can't be casted)
    static CKScene *Cast(CKObject *iO)
    {
        return CKIsChildClassOf(iO, CKCID_SCENE) ? (CKScene *)iO : NULL;
    }

    DLL_EXPORT void CheckSceneObjectList();

    DLL_EXPORT CKSceneObjectDesc *AddObjectDesc(CKSceneObject *o);

protected:
    int m_SceneGlobalIndex;
    CKSODHash m_SceneObjects;
    CKDWORD m_EnvironmentSettings;
    CK_ID m_Level;
    CKDWORD m_BackgroundColor;
    CK_ID m_BackgroundTexture;
    CK_ID m_StartingCamera;
    CKDWORD m_AmbientLightColor;
    CKDWORD m_FogColor;
    VXFOG_MODE m_FogMode;
    float m_FogStart;
    float m_FogEnd;
    float m_FogDensity;
    int m_AddObjectCount;
    int m_RemoveObjectCount;
    XObjectArray m_AddObjectList;
    XObjectArray m_RemoveObjectList;
    XObjectPointerArray m_ObjectList;
};

#endif // CKSCENE_H
