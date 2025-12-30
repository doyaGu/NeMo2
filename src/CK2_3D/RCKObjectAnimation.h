#ifndef RCKOBJECTANIMATION_H
#define RCKOBJECTANIMATION_H

#include "CKRenderEngineTypes.h"
#include "CKObjectAnimation.h"
#include "RCKKeyframeData.h"

class RCKKeyedAnimation;

class RCKObjectAnimation : public CKObjectAnimation {
    friend class RCKKeyedAnimation;  // Allow access to m_ParentKeyedAnimation
public:
    explicit RCKObjectAnimation(CKContext *Context, CKSTRING name = nullptr);
    ~RCKObjectAnimation() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;
    void CheckPreDeletion() override;
    CKBOOL IsObjectUsed(CKObject *obj, CK_CLASSID cid) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;

    // CKObjectAnimation virtual methods
    CKAnimController *CreateController(CKANIMATION_CONTROLLER ControlType) override;
    CKBOOL DeleteController(CKANIMATION_CONTROLLER ControlType) override;
    CKAnimController *GetPositionController() override;
    CKAnimController *GetScaleController() override;
    CKAnimController *GetRotationController() override;
    CKAnimController *GetScaleAxisController() override;
    CKMorphController *GetMorphController() override;
    
    CKBOOL EvaluatePosition(float Time, VxVector &Pos) override;
    CKBOOL EvaluateScale(float Time, VxVector &Scl) override;
    CKBOOL EvaluateRotation(float Time, VxQuaternion &Rot) override;
    CKBOOL EvaluateScaleAxis(float Time, VxQuaternion &ScaleAxis) override;
    CKBOOL EvaluateMorphTarget(float Time, int VertexCount, VxVector *Vertices, CKDWORD VStride, VxCompressedVector *Normals) override;
    CKBOOL EvaluateKeys(float step, VxQuaternion *rot, VxVector *pos, VxVector *scale, VxQuaternion *ScaleRot = NULL) override;
    
    CKBOOL HasMorphNormalInfo() override;
    CKBOOL HasMorphInfo() override;
    CKBOOL HasScaleInfo() override;
    CKBOOL HasPositionInfo() override;
    CKBOOL HasRotationInfo() override;
    CKBOOL HasScaleAxisInfo() override;
    
    void AddPositionKey(float TimeStep, VxVector *pos) override;
    void AddRotationKey(float TimeStep, VxQuaternion *rot) override;
    void AddScaleKey(float TimeStep, VxVector *scl) override;
    void AddScaleAxisKey(float TimeStep, VxQuaternion *sclaxis) override;

    void CheckScaleKeys(VxVector &scale);

    CKBOOL Compare(CKObjectAnimation *anim, float threshold = 0.0f) override;
    CKBOOL ShareDataFrom(CKObjectAnimation *anim) override;
    CKObjectAnimation *Shared() override;
    
    void SetFlags(CKDWORD flags) override;
    CKDWORD GetFlags() override;
    
    void Clear() override;
    void ClearAll() override;
    
    float GetMergeFactor() override;
    void SetMergeFactor(float factor) override;
    CKBOOL IsMerged() override;
    CKObjectAnimation *CreateMergedAnimation(CKObjectAnimation *subanim2, CKBOOL Dynamic = FALSE) override;
    
    void SetLength(float nbframe) override;
    float GetLength() override;
    void GetVelocity(float step, VxVector *vel) override;
    
    CKERROR SetStep(float step, CKKeyedAnimation *anim = NULL) override;
    CKERROR SetFrame(float frame, CKKeyedAnimation *anim = NULL) override;
    float GetCurrentStep() override;
    
    void Set3dEntity(CK3dEntity *ent) override;
    CK3dEntity *Get3dEntity() override;
    
    int GetMorphVertexCount() override;
    
    void CreateTransition(float length, CKObjectAnimation *AnimIn, float StepFrom, CKObjectAnimation *AnimOut, float StepTo, CKBOOL Veloc, CKBOOL DontTurn, CKAnimKey *startingset = NULL) override;
    void Clone(CKObjectAnimation *anim) override;

    // Ensure keyframe data is exclusive to this animation and clear controllers
    void ResetKeyframeData();
    // Set keyframe data length and propagate to all controllers
    void SetKeyframeLength(float length);

    // Static methods for class registration
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKObjectAnimation *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    CKKeyframeData *m_KeyframeData;  // 0x1C
    CKDWORD m_Flags;                 // 0x20
    RCK3dEntity *m_Entity;           // 0x24
    float m_CurrentStep;             // 0x28
    float m_MergeFactor;             // 0x2C - Note: animation length is stored in m_KeyframeData->m_Length
    RCKObjectAnimation *m_Anim1;     // 0x30 - First source animation for merged animations
    RCKObjectAnimation *m_Anim2;     // 0x34 - Second source animation for merged animations
    CKDWORD m_field_38;              // 0x38 - Reserved/unknown field (set to 0 in constructor)
    RCKKeyedAnimation *m_ParentKeyedAnimation; // 0x3C - Parent keyed animation (used in SetStep)
};

#endif // RCKOBJECTANIMATION_H
