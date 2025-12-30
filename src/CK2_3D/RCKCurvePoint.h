#ifndef RCKCURVEPOINT_H
#define RCKCURVEPOINT_H

#include "RCK3dEntity.h"

class RCKCurvePoint : public RCK3dEntity {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKCurvePoint.h"
#undef CK_3DIMPLEMENTATION

    //--------------------------------------------------------
    ////               Private Part

    //----------------------------------------------------------
    // Internal functions

    explicit RCKCurvePoint(CKContext *Context, CKSTRING name = nullptr);
    ~RCKCurvePoint() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    //--------------------------------------------
    // Dependencies Functions	{Secret}
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    CKERROR RemapDependencies(CKDependenciesContext &context) override;

    void Rotate(const VxVector *Axis, float Angle, CK3dEntity *Ref = NULL, CKBOOL KeepChildren = FALSE) override;
    void Translate(const VxVector *Vect, CK3dEntity *Ref = NULL, CKBOOL KeepChildren = FALSE) override;
    void AddScale(const VxVector *Scale, CKBOOL KeepChildren = FALSE, CKBOOL Local = TRUE) override;
    void SetPosition(const VxVector *Pos, CK3dEntity *Ref = NULL, CKBOOL KeepChildren = FALSE) override;
    void SetOrientation(const VxVector *Dir, const VxVector *Up, const VxVector *Right = NULL, CK3dEntity *Ref = NULL, CKBOOL KeepChildren = FALSE) override;
    void SetQuaternion(const VxQuaternion *Quat, CK3dEntity *Ref = NULL, CKBOOL KeepChildren = FALSE, CKBOOL KeepScale = FALSE) override;
    void SetScale(const VxVector *Scale, CKBOOL KeepChildren = FALSE, CKBOOL Local = TRUE) override;
    CKBOOL ConstructWorldMatrix(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat) override;
    CKBOOL ConstructWorldMatrixEx(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat, const VxQuaternion *Shear, float Sign) override;
    CKBOOL ConstructLocalMatrix(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat) override;
    CKBOOL ConstructLocalMatrixEx(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat, const VxQuaternion *Shear, float Sign) override;
    void SetLocalMatrix(const VxMatrix &Mat, CKBOOL KeepChildren = FALSE) override;
    void SetWorldMatrix(const VxMatrix &Mat, CKBOOL KeepChildren = FALSE) override;

    void SetCurve(CKCurve *curve);

    // Internal cached data used by CKCurve (matches original engine layout/usage)
    void SetCurveLength(float length);
    void GetReservedVector(VxVector *vector) const;
    void SetReservedVector(const VxVector *vector);
    void GetFittedVector(VxVector *vector) const;
    void SetFittedVector(const VxVector *vector);

    //--------------------------------------------
    // Class Registering	{Secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKCurvePoint *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    CKCurve *m_Curve;
    float m_Tension;
    float m_Continuity;
    float m_Bias;
    float m_Length;
    VxVector m_ReservedVector;
    VxVector m_TangentIn;
    VxVector m_m_TangentOut;
    VxVector m_NotUsedVector;
    CKBOOL m_UseTCB;
    CKBOOL m_Linear;
};

#endif // RCKCURVEPOINT_H
