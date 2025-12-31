#ifndef CKGROUP_H
#define CKGROUP_H

#include "CKBeObject.h"

/**************************************************************************
{filename:CKGroup}
Summary: Management of group of objects

Remarks:
    + Behavioral Objects can be grouped together because they share common attributes
    or for better understanding of a level. A group is simply a list of objects that
    can be removed or added.

    + The class id of CKGroup is CKCID_GROUP.
See also: CKBeObject
**************************************************************************/
class CKGroup : public CKBeObject
{
    friend class CKBeObject;

public:
    //---------------------------------------
    // Insertion Removal
    DLL_EXPORT CKERROR AddObject(CKBeObject *o);
    DLL_EXPORT CKERROR AddObjectFront(CKBeObject *o);
    DLL_EXPORT CKERROR InsertObjectAt(CKBeObject *o, int pos);

    DLL_EXPORT CKBeObject *RemoveObject(int pos);
    DLL_EXPORT void RemoveObject(CKBeObject *obj);
    DLL_EXPORT void Clear();

    //---------------------------------------
    // Order
    DLL_EXPORT void MoveObjectUp(CKBeObject *o);
    DLL_EXPORT void MoveObjectDown(CKBeObject *o);

    //---------------------------------------
    // Object Access
    DLL_EXPORT CKBeObject *GetObject(int pos);
    DLL_EXPORT int GetObjectCount();

    DLL_EXPORT CK_CLASSID GetCommonClassID();

    //--------------------------------------------------------
    ////               Private Part

    virtual int CanBeHide();

    //----------------------------------------------------------
    // Object Visibility
    void Show(CK_OBJECT_SHOWOPTION Show = CKSHOW);

    //-------------------------------------------------
    // Internal functions	{secret}
    CKGroup(CKContext *Context, CKSTRING Name = NULL);
    virtual ~CKGroup();
    virtual CK_CLASSID GetClassID();

    virtual void AddToScene(CKScene *scene, CKBOOL dependencies);
    virtual void RemoveFromScene(CKScene *scene, CKBOOL dependencies);

    virtual void PreSave(CKFile *file, CKDWORD flags);
    virtual CKStateChunk *Save(CKFile *file, CKDWORD flags);
    virtual CKERROR Load(CKStateChunk *chunk, CKFile *file);
    virtual void PostLoad();

    virtual void PreDelete();
    virtual void CheckPreDeletion();

    virtual size_t GetMemoryOccupation();
    virtual CKBOOL IsObjectUsed(CKObject *o, CK_CLASSID cid);

    //--------------------------------------------
    // Dependencies Functions {secret}
    virtual CKERROR PrepareDependencies(CKDependenciesContext &context);
    virtual CKERROR RemapDependencies(CKDependenciesContext &context);
    virtual CKERROR Copy(CKObject &o, CKDependenciesContext &context);

    //--------------------------------------------
    // Class Registering {secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKGroup *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    // Dynamic Cast method (returns NULL if the object can't be cast)
    static CKGroup *Cast(CKObject *iO)
    {
        return CKIsChildClassOf(iO, CKCID_GROUP) ? (CKGroup *)iO : NULL;
    }

    void ComputeClassID();

protected:
    XObjectPointerArray m_ObjectArray;
    CK_CLASSID m_CommonClassId;
    CKBOOL m_ClassIdUpdated;
    int m_GroupIndex;
};

#endif // CKGROUP_H
