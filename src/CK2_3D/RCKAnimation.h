#ifndef RCKANIMATION_H
#define RCKANIMATION_H

#include "CKRenderEngineTypes.h"
#include "CKAnimation.h"

// Flag bit definitions for CKAnimation::m_Flags
// Bit 0  (0x00000001): CKANIMATION_LINKTOFRAMERATE
// Bit 2  (0x00000004): CKANIMATION_CANBEBREAK
// Bit 4  (0x00000010): CKANIMATION_ALIGNORIENTATION
// Bits 8-16  (0x0001FF00): TransitionMode
// Bits 18-23 (0x00EC0000): SecondaryAnimationMode

class RCKAnimation : public CKAnimation {
    friend class RCKCharacter;  // Allow RCKCharacter to access protected members
public:
    explicit RCKAnimation(CKContext *Context, CKSTRING name = nullptr);
    ~RCKAnimation() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    void CheckPreDeletion() override;

    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;

    // CKAnimation virtual methods
    float GetLength() override;
    float GetFrame() override;
    float GetNextFrame(float delta_t) override;
    float GetStep() override;
    void SetFrame(float frame) override;
    void SetStep(float step) override;
    void SetLength(float nbframe) override;
    CKCharacter *GetCharacter() override;
    void LinkToFrameRate(CKBOOL link, float fps = 30.0f) override;
    float GetLinkedFrameRate() override;
    CKBOOL IsLinkedToFrameRate() override;
    void SetTransitionMode(CK_ANIMATION_TRANSITION_MODE mode) override;
    CK_ANIMATION_TRANSITION_MODE GetTransitionMode() override;
    void SetSecondaryAnimationMode(CK_SECONDARYANIMATION_FLAGS mode) override;
    CK_SECONDARYANIMATION_FLAGS GetSecondaryAnimationMode() override;
    void SetCanBeInterrupt(CKBOOL can = TRUE) override;
    CKBOOL CanBeInterrupt() override;
    void SetCharacterOrientation(CKBOOL orient = TRUE) override;
    CKBOOL DoesCharacterTakeOrientation() override;
    void SetFlags(CKDWORD flags) override;
    CKDWORD GetFlags() override;
    CK3dEntity *GetRootEntity() override;
    void SetCurrentStep(float Step) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKAnimation *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    // Object layout: sizeof(RCKAnimation) = 0x34 (52 bytes)
    // Offset 0x00: CKAnimation base (28 bytes)
    RCKCharacter *m_Character;    // Offset 0x1C
    float m_Length;               // Offset 0x20 - Default: 100.0f
    float m_Step;                 // Offset 0x24 - Default: 0.0f (represents normalized frame position)
    RCK3dEntity *m_RootEntity;    // Offset 0x28
    CKDWORD m_Flags;              // Offset 0x2C - Default: CKANIMATION_LINKTOFRAMERATE | CKANIMATION_CANBEBREAK
    float m_FrameRate;            // Offset 0x30 - Default: 30.0f
    // Note: TransitionMode and SecondaryAnimationMode are packed into m_Flags
};

#endif // RCKANIMATION_H
