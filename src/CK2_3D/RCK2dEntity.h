#ifndef RCK2DENTITY_H
#define RCK2DENTITY_H

#include "CKBitmapData.h"
#include "RCKRenderObject.h"
#include "Vx2dVector.h"
#include "VxRect.h"
#include "CK2dEntity.h"
#include "CKContext.h"
#include "CKRenderManager.h"
#include "CKRenderContext.h"
#include "CKStateChunk.h"
#include "CKMaterial.h"
#include "CKFile.h"

class RCK2dEntity : public RCKRenderObject {
    friend class CKRenderedScene;
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CK2dEntity.h"
#undef CK_3DIMPLEMENTATION

    // CKRenderObject overrides
    CKBOOL IsInRenderContext(CKRenderContext *context) override;
    CKBOOL IsRootObject() override;
    CKBOOL IsToBeRendered() override;
    void SetZOrder(int Z) override;
    int GetZOrder() override;

    void GetHomogeneousRelativeRect(VxRect &rect);
    void HierarchySetBackground(CKBOOL back);

    // Internal methods used by RCKRenderContext
    CK2dEntity *Pick(const Vx2DVector &pt, CKBOOL ignoreUnpickable);
    CKBOOL UpdateExtents(CKRenderContext *dev);

    //--------------------------------------------------------
    ////               Private Part

    //----------------------------------------------------------
    // Internal functions
    CKBOOL IsHiddenByParent() override;

    explicit RCK2dEntity(CKContext *Context, CKSTRING name = nullptr);
    ~RCK2dEntity() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;
    void PostLoad() override;

    void PreDelete() override;
    void CheckPreDeletion() override;

    int GetMemoryOccupation() override;

    //--------------------------------------------
    // Dependencies Functions	{Secret}
    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static RCK2dEntity *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    VxRect m_Rect;
    VxRect *m_HomogeneousRect;
    VxRect m_VtxPos;
    VxRect m_SrcRect;
    CKDWORD m_Flags;
    CK2dEntity *m_Parent;
    CKMaterial *m_Material;
    XArray<CK2dEntity *> m_Children;
    VxRect m_SourceRect;
    int m_ZOrder;
};

#endif // RCK2DENTITY_H
