#ifndef RCKGRID_H
#define RCKGRID_H

#include "CKRenderEngineTypes.h"
#include "XObjectArray.h"

#include "RCK3dEntity.h"

class RCKGrid : public RCK3dEntity {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKGrid.h"
#undef CK_3DIMPLEMENTATION

    explicit RCKGrid(CKContext *Context, CKSTRING name = nullptr);
    ~RCKGrid() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;
    void PostLoad() override;
    void Show(CK_OBJECT_SHOWOPTION show = CKSHOW) override;
    void CheckPostDeletion() override;

    int GetMemoryOccupation() override;
    CKBOOL IsObjectUsed(CKObject *obj, CK_CLASSID cid) override;
    void UpdateBox(CKBOOL World = TRUE) override;

    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKGrid *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    XObjectArray m_Layers;
    CKDWORD m_Width;
    CKDWORD m_Length;
    CKDWORD m_Priority;
    CKDWORD m_OrientationMode;
    RCKMesh *m_Mesh;
};

#endif // RCKGRID_H
