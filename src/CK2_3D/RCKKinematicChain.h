#ifndef RCKKINEMATICCHAIN_H
#define RCKKINEMATICCHAIN_H

#include "CKRenderEngineTypes.h"
#include "CKKinematicChain.h"

/**
 * @brief Internal structure for IK chain body data.
 * Size: 116 bytes per body part in the chain.
 * Used during IKSetEffectorPos for joint constraint evaluation.
 *
 * Layout based on IDA analysis:
 * - offset 0: CKIkJoint m_RotationJoint (40 bytes)
 * - offset 40: CKBodyPart* m_BodyPart
 * - offset 44: VxMatrix m_LocalTransform (64 bytes)
 * - offset 108: CKDWORD m_IsLocked
 * - offset 112: CKDWORD m_WasAtLimit
 * Total: 116 bytes
 */
struct CKIKChainBodyData {
    CKIkJoint m_RotationJoint;     // offset 0: Rotation joint data (40 bytes)
    CKBodyPart *m_BodyPart;        // offset 40: Pointer to body part
    VxMatrix m_LocalTransform;     // offset 44: Saved local transform matrix (64 bytes)
    CKDWORD m_IsLocked;            // offset 108: Whether this joint is locked
    CKDWORD m_WasAtLimit;          // offset 112: Whether joint was at limit during last solve
};

class RCKKinematicChain : public CKKinematicChain {
public:
    explicit RCKKinematicChain(CKContext *Context, CKSTRING name = nullptr);
    ~RCKKinematicChain() override;
    
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    void CheckPreDeletion() override;
    int GetMemoryOccupation() override;
    CKBOOL IsObjectUsed(CKObject *obj, CK_CLASSID cid) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    // CKKinematicChain virtual methods
    float GetChainLength(CKBodyPart *End = nullptr) override;
    int GetChainBodyCount(CKBodyPart *End = nullptr) override;
    CKBodyPart *GetStartEffector() override;
    CKERROR SetStartEffector(CKBodyPart *start) override;
    CKBodyPart *GetEffector(int pos) override;
    CKBodyPart *GetEndEffector() override;
    CKERROR SetEndEffector(CKBodyPart *end) override;
    CKERROR IKSetEffectorPos(VxVector *pos, CK3dEntity *ref = nullptr, CKBodyPart *body = nullptr) override;

    // Class registration helpers
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKKinematicChain *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

private:
    // IK helper methods
    CKERROR IKRotateToward(VxVector *targetDelta, CKBodyPart *endBody);
    static void SVDDecompose(float **a, int m, int n, float *w, float **v);
    static void SVDSolve(float **u, float *w, float **v, int m, int n, float *b, float *x);

    CKBodyPart *m_StartEffector;   // offset +20 (0x14)
    CKBodyPart *m_EndEffector;     // offset +24 (0x18)
    CKDWORD m_ChainBodyCount;      // offset +28 (0x1C): Number of body parts in chain
    CKIKChainBodyData *m_ChainData; // offset +32 (0x20): Pointer to IK chain body data array
};

#endif // RCKKINEMATICCHAIN_H
