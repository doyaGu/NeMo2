#ifndef RCKCURVE_H
#define RCKCURVE_H

#include "RCK3dEntity.h"

class RCKCurvePoint;

class RCKCurve : public RCK3dEntity {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKCurve.h"
#undef CK_3DIMPLEMENTATION

    //--------------------------------------------------------
    ////               Private Part

    //----------------------------------------------------------
    // Internal functions

    explicit RCKCurve(CKContext *Context, CKSTRING name = nullptr);
    ~RCKCurve() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    void CheckPreDeletion() override;

    int GetMemoryOccupation() override;
    int IsObjectUsed(CKObject *o, CK_CLASSID cid) override;

    //--------------------------------------------
    // Dependencies Functions	{Secret}
    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    CKBOOL Render(CKRenderContext *Dev, CKDWORD Flags = CKRENDER_UPDATEEXTENTS) override;

    //--------------------------------------------
    // Class Registering	{Secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKCurve *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    XObjectPointerArray m_ControlPoints;
    CKBOOL m_Opened;
    float m_Length;
    int m_StepCount;
    float m_FittingCoeff;
    CKDWORD m_Color;
    CKBOOL m_Loading; // Non-zero while loading legacy data
};

#endif // RCKCURVE_H
